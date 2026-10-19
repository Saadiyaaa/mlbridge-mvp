"""对已有 Colab 会话发起一次性 SSH 系统观测或自定义命令。

从 colab-cli 会话存储选择目标 session，并在 0700 临时目录中创建仅包含
该会话的 0600 存储供 ProxyCommand 使用；目录在请求结束后删除。
连接时指定 ``-s NAME``，不使用 ``--rm``，不创建或停止实例；此行为依赖
google-colab-cli 0.7.4 的会话解析机制。连接失败不代表 Worker 失败。

使用已有 SSH 私钥，不自动生成；Host key 使用临时 known_hosts 与
``accept-new``，不保留跨请求的主机身份记录，无法提供持续的主机身份验证。
默认快照只输出系统指标
和文件大小，不读取项目日志正文；自定义命令以远端 root 权限执行。
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
    """找到已有的兼容私钥；未找到则拒绝连接。"""
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
        # 请求结束时清理包含 session token 的临时存储。
    if done.returncode != 0:
        # 不拼接远端 stderr：它可能包含原项目输出或敏感内容。
        raise BridgeError(f'SSH 观察失败(exit {done.returncode})；请确认实例仍在运行、'
                          '密钥可用及远端命令成功。连接失败不表示任务失败')
    return done.returncode
