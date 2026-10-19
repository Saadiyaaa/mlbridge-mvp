"""Short-lived SSH observation channel for MLBridge jobs (observation only).

``ssh_observe`` 让 Agent 轮询 GPU / 系统 / 进程 / 输出 / 日志大小 / worker
阶段：每次观察都重新用 SSH 连一次既有 runtime，事后立即断开。它本身
不创建、不停止、不改变任何 Colab runtime —— 观察失败绝不等于任务失败。

机制
----
1. 从既有的 colab-cli 会话存储（默认 ``~/.config/colab-cli/sessions.json``，
   可被环境变量 ``MLBRIDGE_COLAB_SESSIONS`` 重定向到绝对路径，测试用）中
   取用户指名的 session。目标 session 不存在时，在发起任何连接之前就拒绝。
2. 在 0700 临时目录里构造一个 0600 的隔离 sessions.json，只写入选中
   session 的整个条目（含 token）。无论成功失败都随临时目录删除；token
   永不打印，错误信息中也绝不包含存储内容。
3. 用 OpenSSH 发起一次性命令；ProxyCommand 指名隔离存储内已存在的
   session。colab-cli 0.7.4 在 _session_exists 为真时不会新建实例；
   若会话已终止，get_session 返回失败。不能用不带 -s 的代理调用，
   因为 CLI 的单会话选择提示会写入 stdout、破坏 SSH 字节流。

安全边界（有意为之的设计，勿"顺手加固"）
----------------------------------------
* 绝不调用 ``colab new``，只在隔离存储中已有目标会话时传 ``-s NAME``；
  不对不存在的会话连接，绝不传 ``--rm``（会在断开时杀掉 runtime）。
  这一保护依赖已核对的 google-colab-cli 0.7.4 的会话解析行为。
* SSH 身份：优先使用既有的 MLBridge 专用密钥（配置目录 ssh/id_ed25519），
  否则使用 ``~/.ssh/id_ed25519`` 或 ``id_ecdsa``（或用参数 ``identity``
  显式指定路径）。绝不生成密钥；colab 代理握手
  （``--identity``）与 ssh（``-i``）使用同一把私钥。
* Host key 策略：使用临时 known_hosts 与 ``accept-new``；不建立跨调用的
  持久 host-key 信任，不能等同于服务器身份的端到端验证。代理 WebSocket
  仍依赖 colab-cli 的 TLS、会话 token 和公钥；勿向未信任的实例执行命令。
* 远程命令：默认快照只输出 GPU / CPU / 内存 / 磁盘 / 进程摘要，输出与
  日志只报大小和修改时间，不打印凭据、环境变量、命令参数或日志内容。
  自定义命令经 ``shlex.join`` 生成
  POSIX 安全引用后整体发给远端 shell —— 调用方被视作已显式授权远程执行。
"""
from __future__ import annotations

import contextlib
import json
import os
import shlex
import shutil
import signal
import subprocess
import tempfile
from pathlib import Path

from .colab_runner import SESSION
from .core import BridgeError, config_dir

_DEFAULT_IDENTITY_KEYS = ('id_ed25519', 'id_ecdsa')

# 一次性、人类可读、行数封顶的系统快照。只有大小/相位，不含日志正文。
_DEFAULT_SNAPSHOT_SCRIPT = r'''
echo "== MLBridge 快照 $(date -u +%Y-%m-%dT%H:%M:%SZ) =="
echo '-- GPU --'
if command -v nvidia-smi >/dev/null 2>&1; then
    timeout 5 nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader 2>/dev/null \
        || echo 'nvidia-smi 查询失败（驱动忙或正在启动）'
    timeout 5 nvidia-smi --query-compute-apps=pid,used_memory \
        --format=csv,noheader 2>/dev/null || true
else
    echo '非 GPU runtime（无 nvidia-smi）'
fi
echo '-- CPU / 内存 --'
uptime
free -m 2>/dev/null | head -n 3 || true
echo '-- 磁盘 --'
df -h /content / 2>/dev/null | head -n 5 || true
echo '-- 进程 CPU TOP（前 12 行）--'
ps -eo pid,ppid,pcpu,pmem,rss,stat,etime,comm --sort=-pcpu 2>/dev/null | head -n 12 || true
echo '-- 工作区 / 日志大小与时间（不看内容）--'
for p in /content/mlbridge-workspace /content/mlbridge_job.json \
         /content/mlbridge_run.log; do
    if [ -e "$p" ]; then
        timeout 10 du -sh "$p" 2>/dev/null
        stat -c '%y %n' "$p" 2>/dev/null
    fi
done
echo '-- 配置输出目录（数量和大小；限制扫描）--'
python3 - <<'PY'
import json, os, time
from pathlib import Path
root = Path('/content/mlbridge-workspace').resolve()
try:
    paths = json.loads(Path('/content/mlbridge_job.json').read_text()).get('outputs', [])
except (OSError, ValueError, TypeError):
    paths = []
for relative in paths[:20]:
    if not isinstance(relative, str):
        continue
    target = (root / relative).resolve()
    if not target.is_relative_to(root):
        continue
    count = size = 0
    pending = [target]
    scanned = 0
    truncated = False
    deadline = time.monotonic() + 2
    while pending and scanned < 5000 and time.monotonic() < deadline:
        p = pending.pop()
        scanned += 1
        try:
            if p.is_symlink():
                continue
            if p.is_dir():
                with os.scandir(p) as entries:
                    for entry in entries:
                        if scanned + len(pending) >= 5000 or time.monotonic() >= deadline:
                            truncated = True
                            break
                        pending.append(Path(entry.path))
            elif p.is_file():
                size += p.stat().st_size
                count += 1
        except OSError:
            pass
    print(f'{relative}: files={count} bytes={size} '
          f'partial={truncated or bool(pending)} exists={target.exists()}')
PY
echo '-- worker 阶段 --'
if [ -f /content/mlbridge_state.json ]; then
    grep -o '"phase"[[:space:]]*:[[:space:]]*"[^"]*"' \
        /content/mlbridge_state.json | head -n 1
else
    echo '无 mlbridge_state.json（worker 尚未写入状态）'
fi
'''


def default_snapshot_argv() -> list[str]:
    """默认一次观察命令：走同一条 shlex.join 引用路径的自定义命令。"""
    return ['sh', '-c', _DEFAULT_SNAPSHOT_SCRIPT.strip()]


def sessions_store_path() -> Path:
    """定位 colab-cli 的会话存储；重定向变量必须是绝对路径。"""
    override = os.environ.get('MLBRIDGE_COLAB_SESSIONS')
    if override:
        path = Path(override).expanduser()
        if not path.is_absolute():
            raise BridgeError('MLBRIDGE_COLAB_SESSIONS 必须是绝对路径')
        return path
    return Path.home() / '.config' / 'colab-cli' / 'sessions.json'


def locate_identity(identity: str | None = None) -> Path:
    """找到既有兼容私钥；绝不生成、绝不满意于缺失。"""
    candidates = ([Path(identity).expanduser()] if identity else
                  [config_dir() / 'ssh' / 'id_ed25519',
                   *[Path.home() / '.ssh' / key for key in _DEFAULT_IDENTITY_KEYS]])
    for path in candidates:
        if path.is_file():
            return path
    if identity:
        raise BridgeError(f'指定的 SSH 私钥不存在: {identity}')
    raise BridgeError('找不到既有兼容 SSH 私钥（配置目录 ssh/id_ed25519 或 '
                      '~/.ssh/id_ed25519 / id_ecdsa）；本观察接口绝不生成密钥')


def locate_colab() -> str:
    """colab CLI 的绝对路径；ProxyCommand 里禁止依赖外部 PATH。"""
    found = shutil.which('colab')
    if not found:
        raise BridgeError('找不到 colab 可执行文件；请确认 google-colab-cli 已安装')
    return str(Path(found).resolve())


def load_entry(session: str) -> dict:
    """只读既有存储并取出指名 session 的条目；不满足即在连接前拒绝。"""
    if not SESSION.fullmatch(session):
        raise BridgeError('session ID 不合法')
    path = sessions_store_path()
    if not path.is_file():
        raise BridgeError(f'找不到 colab-cli 会话存储 {path}；从未启动过任务？')
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        # 绝不把文件内容带进异常。
        raise BridgeError(f'colab-cli 会话存储不可读: {path} '
                          f'({type(exc).__name__})') from exc
    if not isinstance(data, dict):
        raise BridgeError(f'colab-cli 会话存储格式不符: {path}')
    if not data:
        raise BridgeError('本地没有任何活跃会话；观察接口绝不自动创建 runtime')
    entry = data.get(session)
    if not isinstance(entry, dict) or entry.get('name') != session or not all(
            isinstance(entry.get(k), str) and entry[k]
            for k in ('token', 'url', 'endpoint')):
        raise BridgeError(
            f'会话 {session} 不在本地会话存储中；先 launch 或 colab new，'
            '观察接口绝不代为创建 runtime')
    return entry


def _isolated_copy(tmpdir: str, session: str, entry: dict) -> Path:
    """被选 session 整条写入 0600 隔离存储；目录 0700。"""
    os.chmod(tmpdir, 0o700)
    config = Path(tmpdir) / 'sessions.json'
    fd = os.open(config, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        json.dump({session: entry}, f)
    return config


def _ssh_argv(colab: str, config: Path, identity: Path, session: str,
              remote_command: str) -> list[str]:
    proxy = (f'{shlex.quote(colab)} --config {shlex.quote(str(config))}'
             ' ssh --proxy-mode -s ' + shlex.quote(session)
             + ' --identity ' + shlex.quote(str(identity)))
    return ['ssh', '-F', '/dev/null',
            '-o', f'ProxyCommand={proxy}',
            '-o', 'BatchMode=yes',
            '-o', 'ConnectTimeout=15',
            '-o', 'StrictHostKeyChecking=accept-new',
            '-o', f'UserKnownHostsFile={config.parent / "known_hosts"}',
            '-o', 'IdentitiesOnly=yes',
            '-i', str(identity),
            '-T', '--', 'root@colab-runtime', remote_command]


def _run(argv: list[str], *, timeout: int) -> subprocess.CompletedProcess:
    """stdout 直通；限时只清理本次 SSH + ProxyCommand 的进程组。"""
    child_env = dict(os.environ)
    child_env.pop('MLBRIDGE_GITHUB_TOKEN', None)
    child = subprocess.Popen(argv, stdin=subprocess.DEVNULL,
                             stderr=subprocess.PIPE, text=True, errors='replace',
                             start_new_session=True, env=child_env)
    try:
        _, stderr = child.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(child.pid, signal.SIGKILL)
        child.communicate()
        raise
    return subprocess.CompletedProcess(argv, child.returncode, None, stderr)


def ssh_observe(session: str, command: list[str] | None = None, *,
                timeout: int = 60, identity: str | None = None) -> int:
    """对既有 Colab runtime 发起一次短命 SSH 观察并返回退出码。

    Args:
      session: 已存在的本地会话名；不存在时在连接前拒绝。
      command: 自定义远端 argv；None 表示内置的系统快照。
      timeout: 整个观察（含建链 + 远端命令）的时间预算上限（秒）。
      identity: 可选私钥路径；默认优先选 MLBridge 专用密钥，再选 ~/.ssh。

    Returns:
      远端命令的退出码（0 = 正常）。

    Raises:
      BridgeError: 目标 session 缺失/存储异常/找不到密钥或 colab/连接或
        远端命令失败/超时。任何失败都不会触碰、停止或创建 runtime。
    """
    if not isinstance(timeout, int) or not 5 <= timeout <= 3600:
        raise BridgeError('timeout 需为 5–3600 秒')
    entry = load_entry(session)                       # 连接前即拒绝缺失会话
    key = locate_identity(identity)
    colab = locate_colab()
    if command is not None and not command:
        raise BridgeError('远端命令不能为空')
    remote_command = shlex.join(command if command is not None else default_snapshot_argv())
    with tempfile.TemporaryDirectory(prefix='mlbridge-observe-') as tmpdir:
        config = _isolated_copy(tmpdir, session, entry)
        try:
            done = _run(_ssh_argv(colab, config, key, session, remote_command),
                        timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise BridgeError(
                f'SSH 观察超时（预算 {timeout}s）：连接或远端命令过慢') from exc
        except FileNotFoundError as exc:
            raise BridgeError('找不到 OpenSSH 客户端 ssh') from exc
        # 临时目录（含含 token 的隔离存储）在此 ALWAYS 销毁，成功失败皆然。
    if done.returncode != 0:
        # 不拼接远端 stderr：它可能包含原项目输出或敏感内容。
        raise BridgeError(f'SSH 观察失败(exit {done.returncode})；请确认实例仍在运行、'
                          '密钥可用及远端命令成功。连接失败不表示任务失败')
    return done.returncode
