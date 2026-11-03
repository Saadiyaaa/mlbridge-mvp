"""Lightweight Colab adapter, using only official CLI and local Python stdlib.

The ML process is detached *inside Colab* and does not depend on SSH lifetime.
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import os
import re
import secrets
import stat
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from .core import (BridgeError, default_gpu, default_high_mem,
                   default_max_active_sessions, expand, git_commit,
                   GPU_CHOICES, job_spec, proc, ProcessTimeout, project,
                   repo_url, settings, secret_env, SNAPSHOT, state_dir)

FINAL = {'SUCCEEDED', 'FAILED'}
SESSION = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_-]{0,62}$')

# execution_mode（plan 级，缺省 detached）：仅 'foreground' 在批模式下
# 进入 foreground starter 分支（job 载荷中写 plan['execution_mode']=
# 'foreground'；detached 不写入，legacy 拒绝 foreground）。
EXECUTION_MODES = ('detached', 'foreground')

# foreground cell 内同步执行结束的哨兵行：本地 exec 正常返回（含该哨兵）
# 是“worker 已确定退出”的正面证据；丢失该哨兵则什么都不证明。
FOREGROUND_DONE = 'MLBRIDGE_FOREGROUND_DONE'
# foreground worker 输出统一进 /content/mlbridge_run.log；观测可见性从
# state/receipt 文件与 mlbridge logs（有界尾部）读取，不把整批 stdout
# 塞进内存或 cell 输出。
FOREGROUND_LOG = '/content/mlbridge_run.log'

# foreground 本地 exec 显式超时中的“其余成本余量”：clone/apt-get install/
# restore/setup 这些阶段本身没有远端硬超时（只有逐 unit/发布/manifest
# 超时是远端强制的），在推导本地超时上限时为这部分留出固定余量。
FOREGROUND_EXEC_MARGIN = 900.0


def _foreground_exit_code(cell_output: str) -> int | None:
    """从有界的 cell 输出解析 foreground 哨兵行中的 worker 退出码。

    只认哨兵行：哨兵缺失 ⇒ 返回 None（什么都不证明，绝不猜测退出码）。"""
    for line in (cell_output or '').splitlines():
        if line.startswith(FOREGROUND_DONE):
            rest = line[len(FOREGROUND_DONE):]
            m = re.search(r'worker_exit=(-?\d+)', rest)
            if m:
                return int(m.group(1))
            return None
    return None


def _store_foreground_exit(session: str, worker_exit: int) -> None:
    """写入本地任务记录的 foreground 正面证据（worker 已确定退出）。

    该标记是后续 `watch` / 恢复对账的执行终局证据之一；写入失败只报告，
    绝不虚构为已确认。"""
    try:
        with _job_lock(session):
            record = get_local(session)
            record['foreground_exit'] = {
                'worker_exit': int(worker_exit),
                'observed': datetime.now(timezone.utc).isoformat(),
                'note': 'foreground cell 输出含哨兵行：cell 内同步执行已返回，'
                        'worker 已退出（正面执行终局证据）',
            }
            store_local(session, record)
    except BridgeError as exc:
        print(f'[mlbridge] 无法写入 foreground_exit 标记（后续 watch 请重观测）: '
              f'{exc}', file=sys.stderr, flush=True)


def _note_observation_lost(session: str, timeout: float | None) -> None:
    """本地 foreground exec 超时/传输中断：只记录“观测丢失”，绝不改台账、
    绝不判远端失败、绝不停止实例。"""
    try:
        with _job_lock(session):
            record = get_local(session)
            record['observation_lost'] = {
                'stage': 'foreground-exec',
                'timeout_seconds': None if timeout is None else float(timeout),
                'observed': datetime.now(timezone.utc).isoformat(),
                'note': '本地 exec 连接超时/中断：仅观测丢失，不代表远端任何终态；'
                        '请用 mlbridge watch 按远端事实对账',
            }
            store_local(session, record)
    except BridgeError as exc:
        print(f'[mlbridge] 无法写入观测丢失标记: {exc}', file=sys.stderr, flush=True)

# colab 子进程调用有超时；仅对幂等操作重试，不重试实例分配。
EXEC_TIMEOUT = 300      # 单次 exec：staging 脚本均微小，正常 <10s
UPLOAD_TIMEOUT = 300    # 单次 upload：staging 文件均为 KB 级
ALLOC_TIMEOUT = 900     # new：A100 分配可能排队，但仍须有界
STOP_TIMEOUT = 120
SESSIONS_TIMEOUT = 60   # sessions：只读控制面查询，必须有界
STAGE_ATTEMPTS = 3
STAGE_BACKOFF = 5
# remote_worker 启动后写的状态阶段；出现即说明 worker 已在 VM 上运行。
WORKER_PHASES = frozenset({'PREPARING', 'RESTORING', 'RUNNING', 'PUBLISHING'})

# 私有仓库认证(可选):本地环境变量提供 GitHub 用户名 + fine-grained PAT。
# token 绝不写入 project toml、job plan、argv、git remote URL 或任何日志。
GITHUB_USER_ENV = 'MLBRIDGE_GITHUB_USERNAME'
GITHUB_TOKEN_ENV = 'MLBRIDGE_GITHUB_TOKEN'
# Optional zsh secrets file: read assignments only; NEVER source/eval the file.
_GITHUB_SECRETS_AWK = r'''
/^[[:space:]]*(export[[:space:]]+)?MLBRIDGE_GITHUB_(USERNAME|TOKEN)[[:space:]]*=/ {
    key = $0
    sub(/^[[:space:]]*(export[[:space:]]+)?/, "", key)
    sub(/[[:space:]]*=.*/, "", key)
    value = substr($0, index($0, "=") + 1)
    sub(/^[[:space:]]*/, "", value)
    sub(/[[:space:]]*$/, "", value)
    first = substr(value, 1, 1)
    if ((first == "\"" || first == "\047") && substr(value, length(value), 1) == first)
        value = substr(value, 2, length(value) - 2)
    printf "%s\t%s\n", key, value
}
'''
GIT_AUTH_BASENAME = 'mlbridge_git_auth'
GIT_ASKPASS_BASENAME = 'mlbridge_git_askpass.sh'
_GITHUB_REPO_URL = re.compile(
    r'^https://github\.com/(?P<owner>[A-Za-z0-9][A-Za-z0-9-]*)/'
    r'(?P<repo>[A-Za-z0-9][A-Za-z0-9._-]*)(?:\.git)?$')


def state_path(session: str) -> Path:
    if not SESSION.fullmatch(session):
        raise BridgeError('session ID 不合法')
    return state_dir() / 'jobs' / f'{session}.json'


def store_local(session: str, manifest: dict):
    # 原子替换写入：并发读者（status / watch / stop）永远只会看到完整旧
    # 版本或完整新版本，不会读到半份 JSON；配合 per-job flock 的读改写
    # 合并，rmw + 原子落盘一起保证跨进程不丢失字段。
    path = state_path(session)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'{path.name}.tmp-{os.getpid()}-{secrets.token_hex(4)}')
    try:
        with open(tmp, 'w', encoding='utf-8') as f:
            f.write(json.dumps(manifest, ensure_ascii=False, indent=2))
            f.flush()
            os.fsync(f.fileno())
        tmp.chmod(0o600)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def get_local(session: str) -> dict:
    try:
        return json.loads(state_path(session).read_text())
    except FileNotFoundError as exc:
        raise BridgeError(f'找不到本地任务记录 {session}') from exc


def launch_trace_path(session: str) -> Path:
    """本地 launch/exec/starter 边界的诊断留证文件（JSONL，有界裁剪）。"""
    if not SESSION.fullmatch(session):
        raise BridgeError('session ID 不合法')
    return state_dir() / 'diagnostics' / f'launch-{session}.jsonl'


_TRACE_MAX_BYTES = 512 * 1024
_TRACE_KEEP_BYTES = 64 * 1024


def _trace_launch(session: str, event: dict) -> None:
    """追加一条 JSON 事件到 launch 轨迹文件；失败静默（诊断不砸主流程）。

    只留边界事实（阶段/调用/rc/耗时/有界输出尾部），绝不含凭据或文件内容。
    """
    try:
        path = launch_trace_path(session)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {'observed': datetime.now(timezone.utc).isoformat(), **event}
        data = (json.dumps(record, ensure_ascii=False, default=str) + '\n').encode('utf-8')
        if len(data) > _TRACE_KEEP_BYTES:
            data = (json.dumps({'observed': record['observed'],
                                'event': str(event.get('event', 'unknown'))[:128],
                                'oversize_event_omitted': True,
                                'omitted_bytes': len(data)}) + '\n').encode('utf-8')
        with path.open('ab') as fh:
            fh.write(data)
        path.chmod(0o600)
        if path.stat().st_size > _TRACE_MAX_BYTES:
            # Rotate by bytes, but retain only complete JSONL rows. A raw
            # character tail could cut through a row or a UTF-8 code point.
            with path.open('rb') as fh:
                fh.seek(-_TRACE_KEEP_BYTES, os.SEEK_END)
                fh.readline()  # discard the partial first row
                keep = fh.read()
            tmp = path.with_name(path.name + '.rotate-' + secrets.token_hex(4))
            try:
                with tmp.open('xb') as fh:
                    fh.write(keep)
                tmp.chmod(0o600)
                os.replace(tmp, path)
            finally:
                tmp.unlink(missing_ok=True)
    except OSError:
        pass


def _tail(text: object, limit: int = 4000) -> str:
    if text is None:
        return ''
    if isinstance(text, bytes):
        text = text.decode('utf-8', 'replace')
    return str(text).strip()[-limit:]


def _completion_fields(result) -> dict:
    """Diagnostics tolerate legacy adapters with no captured result.

    Missing fields remain unknown, never a fabricated successful return code.
    """
    return {'rc': getattr(result, 'returncode', None),
            'stdout': _tail(getattr(result, 'stdout', '')),
            'stderr': _tail(getattr(result, 'stderr', ''))}


def _echo(result: subprocess.CompletedProcess) -> None:
    """把捕获的 colab CLI 输出放回控制台，保持原有可见行为。"""
    out = getattr(result, 'stdout', '')
    err = getattr(result, 'stderr', '')
    out = out if isinstance(out, str) else ''
    err = err if isinstance(err, str) else ''
    if out:
        sys.stdout.write(out if out.endswith('\n') else out + '\n')
        sys.stdout.flush()
    if err:
        sys.stderr.write(err if err.endswith('\n') else err + '\n')
        sys.stderr.flush()


def cli(*args, capture=False, timeout=None):
    return proc(['colab', *args], capture=capture, timeout=timeout)


def cli_retry(*args, capture=False, timeout, attempts=STAGE_ATTEMPTS,
              backoff=STAGE_BACKOFF):
    """有界重试的幂等 colab 调用；仅在重试预算内重试，不重新分配实例。"""
    last: Exception | None = None
    for attempt in range(attempts):
        if attempt:
            print(f'[mlbridge] colab {args[0]} 失败，{backoff}s 后重试 '
                  f'({attempt + 1}/{attempts})', file=sys.stderr, flush=True)
            time.sleep(backoff)
        try:
            return cli(*args, capture=capture, timeout=timeout)
        except BridgeError as exc:
            last = exc
    raise BridgeError(
        f'colab {args[0]} 连续 {attempts} 次失败（有界放弃，不会重复分配实例）: '
        f'{last}') from last


# colab sessions 输出：空为 "[colab] No active sessions found on server."；
# 每个活跃 session 一行 "[name] endpoint | Hardware: … | Shape: … | Variant: …"。
_NO_SESSIONS_MARKER = 'no active sessions'
_SESSION_LINE = re.compile(r'^\[([^\]]+)\]\s*(\S+)')
# starter 输出里的 remote worker PID（本地留证用；解析失败不影响流程）。
_STARTER_PID = re.compile(r'MLBridge remote PID:\s*(\d+)')


def list_active_sessions() -> list[str]:
    """向 colab 控制面查询当前活跃 session（服务器权威），fail-closed 解析。

    任何 CLI 失败、超时、或无法识别的输出都抛 BridgeError（视为“拒绝
    分配”），绝不猜测为空、绝不重试、绝不 kill/stop 已有实例。
    """
    r = cli('sessions', capture=True, timeout=SESSIONS_TIMEOUT)
    out = r.stdout or ''
    if _NO_SESSIONS_MARKER in out.lower():
        return []
    names: list[str] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        m = _SESSION_LINE.match(line)
        if not m:
            raise BridgeError(
                f'无法解析 colab sessions 输出（格式可能已变化），按未知处理: {line[:120]}')
        names.append(m.group(1))
    if not names:
        raise BridgeError('colab sessions 输出既非空也无活跃 session 行，按未知处理')
    return names


def check_single_active(own: str, limit: int, active: list[str] | None = None) -> list[str]:
    """在分配前拒绝复用活跃 session，并按 limit 检查其他活跃实例数。

    limit=0 只关闭数量上限；复用活跃 session 仍可能使失败回收停止现有实例。
    """
    fetched = list_active_sessions() if active is None else active
    if own in fetched:
        raise BridgeError(
            f'拒绝分配：session {own} 已在 colab 服务器上活跃。'
            '复用活跃 session 会覆盖本地任务记录，且分配失败时的兜底停止可能'
            '误杀正在生产的实例。请先用 `mlbridge status`/`watch` 观察，'
            '确认任务已结束且实例可释放后手动执行 `mlbridge stop`，'
            '或改用新的 --session 名。不要盲目重试本命令。')
    if limit <= 0:
        return []
    others = [s for s in fetched if s != own]
    if len(others) >= limit:
        raise BridgeError(
            f'单实例策略拒绝分配（max_active_sessions={limit}）：'
            f'已有活跃 session {others}。请先用 `colab sessions` / `mlbridge status` '
            f'确认这些实例是否还在使用；确认后用 `mlbridge stop <session>` 手动释放，'
            f'再重新提交。不要盲目重试本命令；本工具不会代替你停止任何实例。')
    return others


@contextlib.contextmanager
def _job_lock(session: str):
    """单任务跨进程互斥：保护 state/jobs/<session>.json 的读改写合并。

    watch（record_observation / 终态写入）与 stop（shutdown 的 stopped
    标记）仍是逐份读-改-写，若不加锁会互相覆盖（丢字段）。锁粒度是
    单个 job 文件；获取顺序恒为 launch.lock → job lock（launch 持有
    launch.lock 期间的兜底 stop 会短暂持有 job lock），watch/stop 从不
    反向获取 launch.lock，因此不存在死锁环路。锁只在合并窗口内持有，
    不跨任何有界网络调用。
    """
    path = state_path(session).with_suffix('.lock')
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


@contextlib.contextmanager
def _launch_lock():
    """跨进程互斥：同一台机器上的并发 mlbridge run 串行化（检查+分配）。

    只覆盖本工具自身的并发窗口；与其他工具/其他用户直接调用 `colab new`
    之间仍存在 TOCTOU 竞态（见 README 局限说明）。
    """
    path = state_dir() / 'launch.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def get_status(session: str) -> dict:
    with tempfile.TemporaryDirectory(prefix='mlbridge-status-') as t:
        dest = Path(t) / 'state.json'
        try:
            cli('download', '-s', session, '/content/mlbridge_state.json', str(dest),
                capture=True, timeout=75)
            return json.loads(dest.read_text())
        except (BridgeError, ValueError, OSError) as exc:
            return {'phase': 'UNREACHABLE', 'reason': str(exc)}


def fetch_remote_json(session: str, remote_path: str) -> dict:
    """下载远端 JSON 文件（如批逐单元回执）并严格解析。

    任何失败都抛 BridgeError（调用方按“无证据”处理，绝不伪造内容）。
    路径必须形如 /content/...，拒绝 arbitrary 命令注入面。"""
    if not isinstance(remote_path, str) or not remote_path.startswith('/content/'):
        raise BridgeError(f'远端路径必须是 /content/...: {remote_path!r}')
    with tempfile.TemporaryDirectory(prefix='mlbridge-fetch-') as t:
        dest = Path(t) / 'remote.json'
        cli("download", "-s", session, remote_path, str(dest),
            capture=True, timeout=75)
        text = dest.read_text(encoding='utf-8')
    try:
        return json.loads(text)
    except ValueError as exc:
        raise BridgeError(f'远端 JSON 无法解析: {remote_path[:120]} ({exc})') from exc


def restic_snapshots_with_tags(namespace: str, tag: str) -> list[dict]:
    """本地只读 restic snapshots 查询：namespace 中携带给定 tag 的快照。

    - namespace 是计划中的 restic 仓库 URL 字符串（rbclone:* 或本地路径）；
    - 任何 CLI 失败/超时/解析错误都抛 BridgeError（“无法核验”），**绝不**
      返回空列表冒充“缺席已核验”；调用方在拿不到确定性缺席证据时不得
      把 attempt 记为 execution_abandoned；
    - 双保险：即使 restic 的 --tag 过滤语义发生变化，也再按“tags 列表
      字面包含该 tag”过滤一次；只读命令，不写入/不删除/不修复。
    """
    if not isinstance(namespace, str) or not namespace or not isinstance(tag, str) \
            or not tag:
        raise BridgeError('restic snapshots 查询参数非法（拒绝任意命令注入面）')
    r = proc(['restic', '-r', namespace, 'snapshots', '--json', '--tag', tag],
             env=secret_env(settings()), capture=True, timeout=180)
    try:
        data = json.loads(r.stdout)
    except ValueError as exc:
        raise BridgeError(f'restic snapshots --json 输出无法解析: {exc}') from exc
    if not isinstance(data, list):
        raise BridgeError('restic snapshots --json 输出不是快照列表')
    return [s for s in data
            if isinstance(s, dict) and isinstance(s.get('tags'), list)
            and tag in s['tags']]


def record_observation(session: str, status: dict) -> None:
    """把最近一次远端观测写入本地任务记录（last-known state，尽力而为）。

    VM 消失后 /content 不可恢复；本地 last-known 至少保留最后一次看到的
    阶段/时间戳，供 status/watch 判断“消失前在干什么”。绝不把它当作
    远端当前状态。

    瞬时 UNREACHABLE（网络抖动、CLI 超时）绝不覆盖 last_known：否则
    反复观测失败会把最后确认的真实 RUNNING 阶段抹掉，丢失“消失前在
    干什么”的全部信息。已有已确认阶段时直接跳过本次落盘。
    """
    try:
        previous = get_local(session).get('last_known')
    except BridgeError:
        return
    if (status.get('phase') == 'UNREACHABLE' and isinstance(previous, dict)
            and previous.get('phase')):
        # 保留最后一次确认的远端阶段；观测时间戳不变（本次并未真正观测到）。
        return
    try:
        with _job_lock(session):
            record = get_local(session)
            record['last_known'] = {
                **status,
                'observed': datetime.now(timezone.utc).isoformat(),
            }
            store_local(session, record)
    except BridgeError:
        pass


def diagnostic_path(session: str) -> Path:
    return state_dir() / 'diagnostics' / f'{session}.json'


def save_terminal_diagnostic(session: str, status: dict) -> bool:
    """本地保存远端上报的终态（0600）；失败时静默放弃。

    记录不包含凭据，也不验证输出快照完整性。
    """
    if not SESSION.fullmatch(session):
        return False
    try:
        path = diagnostic_path(session)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'session': session,
            'written': datetime.now(timezone.utc).isoformat(),
            'terminal': status,
            'note': '远端自报终态的本地副本；不构成对远端成功/数据完整的独立证明',
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        path.chmod(0o600)
        return True
    except OSError:
        return False


def shutdown(session: str) -> None:
    # stop 幂等（对已停止 session colab 退出码为 0）；有界重试以穿透瞬时
    # 控制面抖动。只有 cli 成功返回后才标记本地 stopped:true。stopped 标记
    # 的读改写在 job 锁内完成，不会覆盖并发 watch 正在写入的其他字段。
    cli_retry('stop', '-s', session, timeout=STOP_TIMEOUT)
    try:
        with _job_lock(session):
            record = get_local(session)
            record['stopped'] = True
            store_local(session, record)
    except BridgeError:
        pass


def logs(session: str, lines: int = 100) -> None:
    if not 1 <= lines <= 2000:
        raise BridgeError('日志行数需在 1–2000 之间')
    code = ("from pathlib import Path\nfrom collections import deque\n"
            "p=Path('/content/mlbridge_run.log')\n"
            f"print(''.join(deque(p.open(errors='replace'), maxlen={lines})) if p.exists() else '日志文件尚未生成')\n")
    put_script(session, code)


def prepare_script() -> str:
    # The script runs in the existing Colab kernel; it only creates /content.
    return 'from pathlib import Path\nPath("/content").mkdir(exist_ok=True)\n'


def starter_script(execution_mode: str = 'detached') -> str:
    """starter 脚本：按 execution_mode 选“分离”或“前台”。

    detached（缺省）：cell 内 Popen(start_new_session)
    分离启动 remote_worker 后 cell 立即结束；worker 独立于 SSH/exec
    生命周期，代价是平台可能回收实例。

    foreground（仅批模式）：worker 在 cell 内**同步**运行，该次 exec 持
    续到批结束为止；代价是本地连接断开/超时时 cell 进程终止可能随之
    终止 worker。worker 的输出全部进 /content/mlbridge_run.log（有界
    尾部可经 `mlbridge logs` 观察），状态/回执周期性落盘供本地 watch
    独立读取；cell 本身只打印有界的一行开始与一行结束哨兵，绝不把整
    批 stdout 塞进输出/内存。两种模式都不承诺平台不回收、不承诺成功。
    脚本不 import mlbridge 包；remote_worker/batch_worker 保持 standalone。
    """
    if execution_mode == 'detached':
        # Explicitly close inherited fds, redirect output and detach process.
        # The launched remote_worker survives closing this colab exec / SSH.
        return '''import os,subprocess,sys\nfrom pathlib import Path\np=Path("/content/mlbridge_run.log")\nwith p.open("ab",buffering=0) as f:\n    proc=subprocess.Popen([sys.executable,"/content/mlbridge_worker.py"],\n        cwd="/content",stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,\n        start_new_session=True,close_fds=True)\nprint("MLBridge remote PID:",proc.pid)\n'''
    if execution_mode == 'foreground':
        # No start_new_session: the worker belongs to the exec process, so a
        # severed connection may terminate it (the documented cost). Output
        # goes to the bounded log file; the cell prints only two lines.
        return (
            'import subprocess,sys,time\n'
            'from pathlib import Path\n'
            'p=Path("/content/mlbridge_run.log")\n'
            'print("MLBridge foreground exec: worker output -> '
            + FOREGROUND_LOG + '", flush=True)\n'
            'started=time.time()\n'
            'with p.open("ab",buffering=0) as f:\n'
            '    r=subprocess.run([sys.executable,"/content/mlbridge_worker.py"],' 
            'cwd="/content",stdin=subprocess.DEVNULL,stdout=f,stderr=f)\n'
            'elapsed=round(time.time()-started,1)\n'
            'print("' + FOREGROUND_DONE + ' worker_exit=%s elapsed=%ss" '
            '% (r.returncode, elapsed), flush=True)\n')
    raise BridgeError(
        f'execution_mode 必须是 {"/".join(EXECUTION_MODES)}: {execution_mode!r}')


def put_script(session: str, contents: str):
    """执行幂等脚本（prepare/askpass/cleanup/logs 等），允许有界重试。

    输出在本地边界处捕获留证（exec 边界 stage/rc/时间/尾部）后回显。
    """
    with tempfile.TemporaryDirectory(prefix='mlbridge-colab-') as t:
        p = Path(t) / 'script.py'
        p.write_text(contents)
        started = time.monotonic()
        try:
            r = cli_retry('exec', '-s', session, '-f', str(p),
                          timeout=EXEC_TIMEOUT, capture=True)
        except BridgeError as exc:
            _trace_launch(session, {'event': 'exec-failed', 'kind': 'put-script',
                                    'seconds': round(time.monotonic() - started, 3),
                                    'error': _tail(str(exc), 2000)})
            raise
        _trace_launch(session, {'event': 'exec', 'kind': 'put-script',
                                'seconds': round(time.monotonic() - started, 3),
                                **_completion_fields(r)})
        _echo(r)


class StarterRun:
    """Result of ONE starter execution.

    ``completed`` is the bounded ``cli exec`` completion; ``remote_pid`` is the
    PID the starter echoed (diagnostics only — an unparseable line is NOT a
    failure and never changes publish semantics).
    """

    __slots__ = ('completed', 'remote_pid')

    def __init__(self, completed, remote_pid: int | None) -> None:
        self.completed = completed
        self.remote_pid = remote_pid


def exec_starter_once(session: str, contents: str, *,
                      timeout: float | None = None,
                      capture: bool = False) -> StarterRun:
    """执行 starter 脚本，且只执行一次（绝不重试）。

    starter 内部启动 remote_worker，不是幂等操作：若 exec 回复丢失后重
    试，会复制出第二个 worker。因此这里只调用一次 cli exec（带界超时；
    foreground 必须显式给出**长**超时），失败时抛错，由调用方按不确定
    性回收（见 _staging_failed），绝不自动重发。

    超时（ProcessTimeout）只表示本地观测中断：远端 worker 可能仍在运行、
    也可能已随连接终止；绝不据此外押任何"远端已失败"的结论，交由
    watch/对账依据远端事实（state/回执/快照 tags）判定。

    输出总在本地边界留证（原始 stdout/stderr 尾部，含 starter 打印的
    remote PID）并回显给用户；返回值同时携带 completion 与解析出的 PID，
    因此 foreground 的哨兵行读取与 launch 诊断共用同一次 exec。
    """
    with tempfile.TemporaryDirectory(prefix='mlbridge-colab-') as t:
        p = Path(t) / 'starter.py'
        p.write_text(contents)
        started = time.monotonic()
        try:
            r = cli('exec', '-s', session, '-f', str(p), capture=True,
                    timeout=EXEC_TIMEOUT if timeout is None else timeout)
        except BridgeError as exc:
            _trace_launch(session, {'event': 'starter-failed',
                                    'seconds': round(time.monotonic() - started, 3),
                                    'error': _tail(str(exc), 2000),
                                    'note': 'starter 只执行一次；传输丢失≠worker 未启动'})
            raise
    _trace_launch(session, {'event': 'starter',
                            'seconds': round(time.monotonic() - started, 3),
                            **_completion_fields(r)})
    _echo(r)
    captured = getattr(r, 'stdout', '')
    remote_pid = None
    for line in (captured if isinstance(captured, str) else '').splitlines():
        matched = _STARTER_PID.search(line)
        if matched:
            remote_pid = int(matched.group(1))
            break
    return StarterRun(r, remote_pid)


def foreground_exec_timeout(content: dict) -> float:
    """foreground 模式的本地 exec 显式长超时（秒）。

    按远端批计划中**强制存在的硬超时**推其界限之和：逐 unit 的
    execution/validator 超时 + manifest 预算超时 + publish_timeout，再
    加上无远端硬超时阶段（clone/apt/restore/setup/排队）的固定余量。
    这只是“观测得足够久”的上限；本地超时仍只作为观测中断处理，绝不
    作为远端失败证据。"""
    from .batch_worker import DEFAULT_MANIFEST_TIMEOUT, DEFAULT_PUBLISH_TIMEOUT
    if not isinstance(content, dict) or not isinstance(content.get('units'), list):
        raise BridgeError('foreground 需要 unit_batch 内容以推导 exec 超时')
    units = content['units']
    total = FOREGROUND_EXEC_MARGIN
    manifest_timeout = DEFAULT_MANIFEST_TIMEOUT
    limits = content.get('manifest_limits')
    if isinstance(limits, dict) and limits.get('timeout_seconds') is not None:
        manifest_timeout = float(limits['timeout_seconds'])
    publish_timeout = float(
        content.get('publish_timeout_seconds') or DEFAULT_PUBLISH_TIMEOUT)
    for item in units:
        if not isinstance(item, dict):
            raise BridgeError('foreground exec 超时推导遇到非法 unit')
        execution = item.get('execution') or {}
        spec = item.get('spec') or {}
        validator = spec.get('validator') or {}
        try:
            total += float(execution['timeout_seconds'])
            total += float(validator['timeout_seconds'])
        except (KeyError, TypeError, ValueError) as exc:
            raise BridgeError(
                'foreground exec 超时推导失败（unit 缺硬超时字段）') from exc
        total += manifest_timeout + publish_timeout
    return total


def cleanup_script() -> str:
    """删除 stage() 已上传的临时凭据；在 launch 失败路径上尽力执行。"""
    return ("import os\n"
            "for name in ('mlbridge_rclone.conf', 'mlbridge_restic_password', "
            "'" + GIT_AUTH_BASENAME + "', '" + GIT_ASKPASS_BASENAME + "'):\n"
            "    try:\n"
            "        os.remove('/content/' + name)\n"
            "    except OSError:\n"
            "        pass\n"
            "print('MLBridge cleanup: temporary credential files removed\\n')\n")


def _github_secrets_path() -> Path:
    return Path.home() / '.config/zsh/private/secrets.env'


def _github_auth_from_file() -> tuple[str, str] | None:
    """用 awk 按键读取纯文本赋值；不执行整份 zsh 文件。"""
    path = _github_secrets_path()
    try:
        mode = path.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise BridgeError('无法检查 GitHub 凭据文件') from exc
    if not stat.S_ISREG(mode.st_mode) or mode.st_uid != os.getuid() or mode.st_mode & 0o077:
        raise BridgeError('GitHub 凭据文件必须属于当前用户、是普通文件且权限不高于 0600')
    child_env = dict(os.environ)
    child_env.pop(GITHUB_TOKEN_ENV, None)
    try:
        result = subprocess.run(['awk', _GITHUB_SECRETS_AWK, str(path)],
                                capture_output=True, text=True, timeout=5,
                                check=True, env=child_env)
    except (OSError, subprocess.SubprocessError) as exc:
        raise BridgeError('无法用 awk 读取 GitHub 凭据文件') from exc
    found: dict[str, str] = {}
    for line in result.stdout.splitlines():
        key, separator, value = line.partition('\t')
        if not separator or key not in (GITHUB_USER_ENV, GITHUB_TOKEN_ENV) or key in found:
            raise BridgeError('GitHub 凭据文件包含重复或无效的赋值')
        found[key] = value
    if not found:
        return None
    return found.get(GITHUB_USER_ENV, ''), found.get(GITHUB_TOKEN_ENV, '')


def github_auth() -> tuple[str, str] | None:
    """显式环境变量成对优先；未设置时按需读取本机的私有文件。"""
    if GITHUB_USER_ENV in os.environ or GITHUB_TOKEN_ENV in os.environ:
        user = os.environ.get(GITHUB_USER_ENV, '').strip()
        token = os.environ.get(GITHUB_TOKEN_ENV, '').strip()
    else:
        from_file = _github_auth_from_file()
        if from_file is None:
            return None
        user, token = from_file
    if '\n' in token or '\r' in token or not re.fullmatch(r'[A-Za-z0-9_.-]+', token):
        raise BridgeError(f'{GITHUB_TOKEN_ENV} 必须是非空的单行 GitHub 令牌')
    if not user:
        raise BridgeError(f'{GITHUB_USER_ENV} 与 {GITHUB_TOKEN_ENV} 必须同时设置')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,99}', user):
        raise BridgeError(f'{GITHUB_USER_ENV} 格式不合法')
    return user, token


def require_github_https_url(repository: str) -> str:
    """私有仓库仅接受干净的 https://github.com/<owner>/<repo>[.git]。

    拒绝内嵌凭据、查询或片段的 URL;错误信息不回显 URL,以防用户
    误写的凭据进入日志。
    """
    if not _GITHUB_REPO_URL.fullmatch(repository or ''):
        raise BridgeError(
            '私有仓库认证仅支持 https://github.com/<owner>/<repo>[.git],'
            'URL 不得内嵌凭据、查询或片段')
    return repository


def _askpass_source(auth_path: str) -> str:
    """GIT_ASKPASS 可执行脚本:从 0600 凭据文件按提示读用户名/token。

    token 只存在于凭据文件中,不进入 argv、环境变量或本脚本内容。
    """
    return ('#!/bin/sh\n'
            'case "$1" in\n'
            f"  *[Uu]sername*) sed -n '1p' {auth_path} ;;\n"
            f"  *) sed -n '2p' {auth_path} ;;\n"
            'esac\n')


def _github_setup_script() -> str:
    """在 Colab kernel 内执行:收紧凭据权限并立刻生成 askpass 脚本。"""
    auth = '/content/' + GIT_AUTH_BASENAME
    askpass = '/content/' + GIT_ASKPASS_BASENAME
    return ('import os\n'
            f'os.chmod({auth!r}, 0o600)\n'
            f'with open({askpass!r}, "w") as f:\n'
            '    f.write(' + repr(_askpass_source(auth)) + ')\n'
            f'os.chmod({askpass!r}, 0o700)\n')


def _stage_github_auth(session: str, user: str, token: str) -> None:
    """把凭据经 0600 本地临时文件上传到 Colab,随后删除本地副本。

    必须在其他 staging 之后调用;上传后立即注入收紧权限的 setup 脚本。
    token 只出现在临时文件与 VM 上的凭据文件里。
    """
    fd, path = tempfile.mkstemp(prefix='mlbridge-git-auth-')
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(f'{user}\n{token}\n')
        os.chmod(path, 0o600)
        stage(session, Path(path), GIT_AUTH_BASENAME)
        put_script(session, _github_setup_script())
    finally:
        os.unlink(path)


def stage(session: str, src: Path, remote_basename: str):
    # 覆盖同一远端路径，重试是幂等的。捕获 CLI 输出：边界处本地留证
    # （stage/rc/时间/输出尾部），控制台行为不变（回显），失败不丢根错误。
    started = time.monotonic()
    try:
        r = cli_retry('upload', '-s', session, str(src), f'/content/{remote_basename}',
                      timeout=UPLOAD_TIMEOUT, capture=True)
    except BridgeError as exc:
        _trace_launch(session, {'event': 'stage-failed', 'remote': remote_basename,
                                'seconds': round(time.monotonic() - started, 3),
                                'error': _tail(str(exc), 2000),
                                'note': '仅记 CLI 错误消息；源文件内容不入轨迹'})
        raise
    _trace_launch(session, {'event': 'stage', 'remote': remote_basename,
                            'seconds': round(time.monotonic() - started, 3),
                            **_completion_fields(r)})
    _echo(r)


def _staging_failed(session: str, starter_sent: bool) -> None:
    """处理 staging 失败：starter 前尝试清理并停止，starter 后保留实例。

    starter 的 exec 回复丢失时 Worker 可能已经启动；记录探测结果，
    由用户确认后接管或停止实例。
    """
    try:
        with _job_lock(session):
            record = get_local(session)
            record['staging_failed'] = True
            store_local(session, record)
    except BridgeError:
        pass
    if starter_sent:
        status = get_status(session)
        phase = status.get('phase', 'UNKNOWN')
        try:
            with _job_lock(session):
                record = get_local(session)
                record['staging_probe_phase'] = phase
                store_local(session, record)
        except BridgeError:
            pass
        if phase in WORKER_PHASES:
            _trace_launch(session, {'event': 'staging-probe', 'phase': phase,
                                    'status_tail': _tail(status, 800),
                                    'note': 'worker 已确认启动：停止任何清理/停止动作；'
                                            '接管交由 watch'})
            print(f'[mlbridge] 远端 worker 已启动 (phase={phase})；不停止实例。'
                  f'请用 mlbridge watch {session} 接管。', file=sys.stderr, flush=True)
            return
        # UNREACHABLE/UNKNOWN：传输未知 ≠ worker 未启动。starter 可能已在
        # 回复丢失前 spawn 了 worker；此时清理凭据或 stop 都可能误杀活任务。
        _trace_launch(session, {'event': 'staging-probe', 'phase': phase,
                                'status_tail': _tail(status, 800),
                                'note': 'state 缺失只记 UNREACHABLE；不断言 VM 死亡；'
                                        '不清理凭据、不停止实例'})
        print(f'[mlbridge] 状态探测不可达 (phase={phase})；不把传输未知当作 '
              'worker 未启动：starter 可能已在丢回复前启动 worker，'
              '因此不清理凭据、不停止实例。', file=sys.stderr, flush=True)
        print(f'[mlbridge] 请先观察: mlbridge watch {session} 或 '
              f'mlbridge status {session}；确认任务确未运行后，再手动执行 '
              f'mlbridge stop {session} 以免持续计费。', file=sys.stderr, flush=True)
        return
    try:
        put_script(session, cleanup_script())
        print('[mlbridge] 已尽力删除 VM 上的临时凭据副本。', file=sys.stderr)
    except BridgeError:
        print('[mlbridge] 无法删除 VM 上的临时凭据（通道失败属预期）；'
              '实例停止后凭据随 VM 销毁。', file=sys.stderr)
    try:
        shutdown(session)
        print(f'[mlbridge] staging 失败，实例 {session} 已停止，避免持续计费。',
              file=sys.stderr, flush=True)
    except BridgeError as exc:
        print(f'[mlbridge] 无法确认实例已停止 ({exc})；请尽快手动执行 '
              f'mlbridge stop {session} 以免持续计费。', file=sys.stderr, flush=True)


def launch(name: str, job: str, inputs: list[str], gpu: str | None = None,
           *, high_mem: bool | None = None, allow_dirty: bool = False,
           session: str | None = None, allow_empty_input: bool = False,
           max_active_sessions: int | None = None,
           unit_batch: dict | None = None,
           on_assign=None, on_assign_abort=None) -> str:
    cfg = project(name)
    store = settings()
    # Preflight: do not start a GPU for missing credentials or a malformed
    # private-repository URL. Fail before any Colab allocation.
    gh_auth = github_auth()
    if gh_auth:
        require_github_https_url(cfg['repository'])
    secret_env(store)
    if unit_batch is not None:
        # 批模式（opt-in）：计划已由 CLI 层在分配前完整预校验；这里只
        # 接受完全解析后的绑定（inputs/setup/commit 使用批计划解析），并确保 batch_worker.py 存在（缺文件 fail closed，在
        # 任何分配之前拒绝）。
        if not isinstance(unit_batch, dict) or not isinstance(
                unit_batch.get('unit_batch'), dict):
            raise BridgeError('unit_batch 必须是完整的批计划绑定 dict')
        if not Path(__file__).with_name('batch_worker.py').is_file():
            raise BridgeError('批模式需要 batch_worker.py 与remote_worker '
                              '同目录 stage；文件缺失，拒绝分配（fail closed）')
        spec = {'setup': unit_batch['setup']}
        commit = unit_batch['commit']
        inputs = unit_batch['inputs']
        mode = unit_batch.get('execution_mode', 'detached')
        if mode not in EXECUTION_MODES:
            raise BridgeError(
                f'execution_mode 必须是 {"/".join(EXECUTION_MODES)}: {mode!r}')
        foreground = mode == 'foreground'
    else:
        foreground = False
        spec = job_spec(cfg, job)
        for snapshot in inputs:
            if not SNAPSHOT.fullmatch(snapshot):
                raise BridgeError('输入必须是明确的 snapshot ID，不能是 latest')
        root = expand(cfg['workspace'])
        commit = git_commit(root, allow_dirty, repository=cfg['repository'])
        if not inputs and not allow_empty_input:
            raise BridgeError('需要至少一个 --input 快照；无输入任务请显式使用 --allow-empty-input')
    gpu_norm = (default_gpu() if gpu is None else gpu).upper()
    if gpu_norm not in GPU_CHOICES:
        raise BridgeError('GPU 参数无效，支持: T4, L4, A100, H100, G4, CPU (或 none)')
    use_high_mem = default_high_mem() if high_mem is None else high_mem
    if use_high_mem and gpu_norm == 'L4':
        raise BridgeError('Colab 的 L4 仅提供单一内存机型，不能请求 --high-mem；'
                          '请改用 A100，或指定 --no-high-mem')
    session = session or f'mlb-{name[:30]}-{secrets.token_hex(4)}'
    if not SESSION.fullmatch(session):
        raise BridgeError('session ID 不合法或太长')
    plan = dict(project=name, job_id=session, session=session,
                repository=cfg['repository'], commit=commit,
                restic_repository=repo_url(name, store),
                inputs=inputs, **spec)
    if foreground:
        # detached（缺省）不写入该键。
        plan['execution_mode'] = 'foreground'
    if unit_batch is not None:
        # 批 JSON 契约：随 job.json 之上的 plan['unit_batch']（字段名与
        # batch_worker.validate_batch 一致），包含绑定 session/job_id 之前
        # 已预校验的全部逐单元定义。
        plan['unit_batch'] = unit_batch['unit_batch']
    request = ['new', '-s', session]
    if gpu_norm in ('CPU', 'NONE'):
        print(f'[mlbridge] Colab 分配中: {session} (CPU)', flush=True)
    else:
        print(f'[mlbridge] Colab 分配中: {session} ({gpu_norm})', flush=True)
        request.extend(('--gpu', gpu_norm))
    if use_high_mem:
        request.append('--high-mem')
    # 先查询活跃实例并拒绝复用，再写本地记录、分配实例；锁内串行化同机提交。
    # 记录可用于分配失败后的排查和停止操作。
    limit = (default_max_active_sessions() if max_active_sessions is None
             else max_active_sessions)
    if max_active_sessions is not None and max_active_sessions < 0:
        raise BridgeError('--max-active-sessions 必须是非负整数（0=关闭并发上限检查）')
    with _launch_lock():
        check_single_active(session, limit)
        local_record = {'project': name, 'job': job, 'session': session,
                        'git_commit': commit, 'inputs': inputs,
                        'stopped': False,
                        'execution_mode': 'foreground' if foreground
                        else 'detached'}
        if unit_batch is not None and isinstance(unit_batch.get('unit_batch'), dict):
            # 批模式：本地任务记录携带完整远端计划，供后续 `watch` / 批
            # 终态对账（无需重传 JSON）。legacy 记录字段保持不变。
            local_record['unit_batch_plan'] = plan
        store_local(session, local_record)
        alloc_started = time.monotonic()
        try:
            r = cli(*request, timeout=ALLOC_TIMEOUT, capture=True)
            _trace_launch(session, {'event': 'allocated',
                                    'seconds': round(time.monotonic() - alloc_started, 3),
                                    **_completion_fields(r)})
            _echo(r)
        except BridgeError as exc:
            # 分配失败不重试 new；对已确认未活跃的 session 尝试停止。
            # 查询与分配之间仍可能发生外部并发变更。
            _trace_launch(session, {'event': 'alloc-failed',
                                    'seconds': round(time.monotonic() - alloc_started, 3),
                                    'error': _tail(str(exc), 2000),
                                    'note': '不重试 new，不自动停止任何非本 session 实例'})
            try:
                shutdown(session)
            except BridgeError:
                pass
            print(f'[mlbridge] 分配失败或超时；若实例实际已创建请用 colab sessions '
                  f'检查并 mlbridge stop {session}。', file=sys.stderr, flush=True)
            raise
        if on_assign is not None:
            # 账本对齐窗口：`colab new` 已成功、任何 staging（含真正
            # 上计费网络/凭据上传）开始之前，把 ledger attempt 写入。
            # 失败则不进入 staging（无 worker、无凭据副本）、停止刚分配
            # 的实例；若错误发生在原子替换之后（台账可能已提交），按
            # 报告与 units/status 人工对账，本层不静默回滚。
            # 捕获 BaseException：KeyboardInterrupt/OSError 等也同样必须
            # 先止损（停止刚分配实例）再重新抛出原始异常，否则会带着
            # 未回收的计费实例跳出本函数。
            try:
                on_assign(session)
            except BaseException:
                print('[mlbridge] 分配成功但台账 attempt 对齐失败：'
                      '不再 staging，正在停止刚分配的实例。',
                      file=sys.stderr, flush=True)
                stopped = False
                try:
                    shutdown(session)
                    stopped = True
                    print(f'[mlbridge] 实例 {session} 已停止；'
                          '本窗口内已写入的 attempt 须用 --retry-failed '
                          '才能开新 attempt（见 mlbridge units）。',
                          file=sys.stderr, flush=True)
                except BaseException as stop_exc:
                    print(f'[mlbridge] 无法确认实例已停止 '
                          f'(instance_stopped=false: {stop_exc!r})；'
                          f'session {session} 保留在本地任务记录中，请手动 '
                          f'mlbridge stop {session} 以免持续计费。',
                          file=sys.stderr, flush=True)
                if stopped and on_assign_abort is not None:
                    # 实例已确认停止：由调用方为对齐窗口内已写入的
                    # attempt 写控制器本地事实证据（见
                    # batch_control.record_alignment_abort）。证据写入
                    # 失败只报告，绝不掩盖原始异常、绝不停在这里。
                    try:
                        on_assign_abort(session)
                    except BaseException as abort_exc:
                        print(f'[mlbridge] 实例停止后写入对齐中止证据失败 '
                              f'({abort_exc!r})；请用 mlbridge units '
                              '人工对账。', file=sys.stderr, flush=True)
                raise
    try:
        # These are tiny files, not the large dataset.
        starter_sent = False
        timeout = None
        with tempfile.TemporaryDirectory(prefix='mlbridge-job-') as t:
            job_json = Path(t) / 'job.json'
            job_json.write_text(json.dumps(plan))
            job_json.chmod(0o600)
            worker = Path(__file__).with_name('remote_worker.py')
            put_script(session, prepare_script())
            stage(session, worker, 'mlbridge_worker.py')
            stage(session, job_json, 'mlbridge_job.json')
            if unit_batch is not None:
                # 批模式显式 stage batch_worker（同目录 sibling）：远端
                # worker 的 unit_batch 分支按需加载。批 JSON 即
                # plan['unit_batch']，随 mlbridge_job.json 上传，字段与
                # batch_worker.validate_batch 契约一致。而 legacy 不
                # stage、不改变上传集合。
                stage(session, Path(__file__).with_name('batch_worker.py'),
                      'batch_worker.py')
            stage(session, expand(store['rclone_config']), 'mlbridge_rclone.conf')
            stage(session, expand(store['password_file']), 'mlbridge_restic_password')
            if gh_auth:
                # 私有仓库凭据最后上传;上传后立刻在 VM 上收紧权限并生成 askpass。
                # 上传到克隆之间，同一 OS 用户可读取凭据（见 README）。
                _stage_github_auth(session, *gh_auth)
            starter_sent = True
            if foreground:
                print('[mlbridge] foreground 执行模式：worker 在 cell 内同步运行；'
                      '本地连接断开/超时可能随 cell 终止执行，detached 则可能被'
                      '平台回收；任一模式都不承诺成功。', flush=True)
                print('[mlbridge] worker 输出写入 /content/mlbridge_run.log'
                      '（`mlbridge logs` 看有界尾部）；状态/回执周期落盘，'
                      '供本地 watch 独立读取。', flush=True)
                timeout = foreground_exec_timeout(unit_batch['unit_batch'])
                print(f'[mlbridge] 本地 exec 显式超时 {timeout:.0f}s（观测上限：'
                      '超时仅作为观测中断处理，不判定远端失败；交 watch/对账）。',
                      flush=True)
                started_run = exec_starter_once(
                    session, starter_script('foreground'),
                    timeout=timeout, capture=True)
                out = started_run.completed.stdout or ''
                if out:
                    print(out, end='' if out.endswith('\n') else '\n', flush=True)
                exit_code = _foreground_exit_code(out)
                if exit_code is not None:
                    # 正面证据：哨兵行出现 ⇒ cell 内同步执行已返回 ⇒
                    # worker 已退出。缺失哨兵 ⇒ 什么都不证明（不记录）。
                    _store_foreground_exit(session, exit_code)
                    print(f'[mlbridge] foreground 执行已确认结束 '
                          f'(worker_exit={exit_code})；后续对账可用 '
                          f'mlbridge watch {session}。', flush=True)
                else:
                    _note_observation_lost(session, timeout)
            else:
                # detached：短超时；CLI 超时/错误沿既有 _staging_failed 路径。
                started_run = exec_starter_once(session, starter_script())
            # remote PID 只是诊断线索（来自 starter 回显），两条路径都记录；
            # 记录失败绝不改变发布语义。
            if started_run.remote_pid is not None:
                _trace_launch(session, {'event': 'starter-pid',
                                        'remote_pid': started_run.remote_pid})
                try:
                    with _job_lock(session):
                        record = get_local(session)
                        record['remote_pid'] = started_run.remote_pid
                        store_local(session, record)
                except BridgeError:
                    pass
            else:
                _trace_launch(session, {'event': 'starter-pid',
                                        'remote_pid': None,
                                        'note': 'starter 未回显可解析 PID'
                                                '（不作为失败）'})
    except Exception as exc:
        # starter 前失败尝试停止；starter 后失败保留实例供检查。
        if foreground and isinstance(exc, ProcessTimeout):
            # foreground exec 超时 = 观测中断（交 watch/对账判定）：
            # 不停止实例、不删凭据、不判失败；仅记录“观测丢失”。
            _note_observation_lost(session, timeout)
            print(f'[mlbridge] foreground exec 超时（>{timeout:.0f}s）：'
                  '仅作为观测中断处理——远端 worker 可能仍在运行，也可能已随'
                  '连接终止；不判定远端失败。请用 mlbridge watch '
                  f'{session} 观测并按远端事实对账。', file=sys.stderr,
                  flush=True)
            return session
        _staging_failed(session, starter_sent)
        raise
    return session


def _finish_success(session: str, status: dict, auto_stop: bool) -> dict:
    if not auto_stop:
        return {**status, 'instance_stopped': False}
    try:
        shutdown(session)
    except BridgeError as exc:
        # The result is already saved locally. Keep it available for a later
        # watch to retry stopping, even if the VM becomes unreachable.
        print(f'[mlbridge] 无法确认 Colab 已停止 ({exc})；请重试 watch 或手动检查/停止 {session}。',
              file=sys.stderr, flush=True)
        return {**status, 'instance_stopped': False, 'stop_error': str(exc)}
    print('[mlbridge] 输出已发布，Colab runtime 已释放。', flush=True)
    return {**status, 'instance_stopped': True}


def _batch_terminal_result(session: str, status: dict, batch_plan: dict,
                           auto_stop: bool) -> dict:
    """批终态逐单元对账（委托 batch_control.settle_terminal，延迟导入
    避免环）；决定是否释放实例。

    stop 规则：只在批内【全部】unit 已为确定成功（台账已提交成果）且
    auto_stop 时；任何未决（PUBLISH_UNKNOWN / RECEIPT_FAILED /
    NOT_STARTED / 无证据 / 未跟踪 attempt）都保留实例不自动 stop。"""
    from .batch_control import settle_terminal
    summary = settle_terminal(session, batch_plan, status)
    for uid, info in summary['units'].items():
        print(f"[mlbridge] 批逐单元 {uid[:12]}: {info['disposition']} "
              f"— {info['detail']}", file=sys.stderr, flush=True)
    result = {'phase': status.get('phase'), 'batch_reconcile': summary}
    if summary['all_resolved_succeeded']:
        if not auto_stop:
            return {**result, 'instance_stopped': False}
        st = _finish_success(session, status, auto_stop)
        return {**st, 'batch_reconcile': summary}
    print('[mlbridge] 批终态未对全部单元形成确定台账结论；不自动 stop；'
          '稍后重 watch 会继续对账（PUBLISH_UNKNOWN 请用 restic tags '
          '人工对账）。', file=sys.stderr, flush=True)
    return {**result, 'instance_stopped': False}


def _execution_ended_proof(record: object) -> dict | None:
    """从本地任务记录中提取“远端执行已确定结束”的控制器证据；无则 None。

    只认两种正面事实（都不是观测猜测）：
    - 实例已确认停止（`colab stop` 成功返回后才写入的 stopped 标记）→
      VM 已释放，执行随实例终止；
    - foreground 哨兵标记（cell 内同步执行已返回，worker 已退出）。

    观测失联（UNREACHABLE）、连续 UNKNOWN、传输失败都**不是**执行终局
    证据；有它们也不写 execution_abandoned。"""
    if not isinstance(record, dict):
        return None
    if record.get('stopped'):
        return {
            'source': 'controller/stop-confirmed',
            'detail': '实例已确认停止（stop 成功返回后写入的标记）：'
                      'VM 已释放，执行随实例终止；本地记录不再有持久化状态可见',
        }
    exit_info = record.get('foreground_exit')
    if (isinstance(exit_info, dict)
            and isinstance(exit_info.get('worker_exit'), int)):
        return {
            'source': 'controller/foreground-exec-exit',
            'detail': f'foreground cell 内同步执行已确认退出 '
                      f'(worker_exit={exit_info["worker_exit"]})：'
                      '执行随 cell 返回而结束',
        }
    return None


def _interrupted_batch_result(session: str, batch_plan: dict,
                              status: dict, proof: dict,
                              auto_stop: bool) -> dict:
    """执行中断对账（批模式）：确认执行结束后逐单元诚实收尾。

    - 常规对账照旧（成功/真实失败证据照常 reconcile/commit）；
    - 只有从未进入发布不确定区（无 PUBLISH_UNKNOWN/RECEIPT_FAILED）
      且 restic tags 核验确认无快照（因而无回执）的 attempt 才写
      execution_abandoned；其余一律保持未决（保留实例不 stop）。
    本函数绝不把观测失联写成任何终态。"""
    from .batch_control import settle_interrupted_execution
    summary = settle_interrupted_execution(session, batch_plan, status,
                                           proof=proof)
    for uid, info in summary['units'].items():
        print(f"[mlbridge] 批逐单元 {uid[:12]}: {info['disposition']} "
              f"— {info['detail']}", file=sys.stderr, flush=True)
    result = {'phase': status.get('phase'),
              'executed_ended': proof['source'],
              'batch_reconcile': summary}
    if summary['all_resolved_succeeded']:
        # 台账全部提交成功（中断后对账发现其实已全部成功）：与终态
        # 路径同一停止规则。
        if not auto_stop:
            return {**result, 'instance_stopped': False}
        return {**_finish_success(session, status, auto_stop),
                'batch_reconcile': summary}
    abandoned = sum(1 for info in summary['units'].values()
                    if info['disposition'] == 'reconciled_abandoned')
    print(f'[mlbridge] 执行中断对账完成（execution_abandoned='
          f'{abandoned}）；未提交/未解决的单元需人工对账后显式 '
          '--retry-failed 才能重跑；不自动 stop。', file=sys.stderr,
          flush=True)
    return {**result, 'instance_stopped': False}


def watch(session: str, *, interval: int = 20, auto_stop: bool = True,
          batch_plan: dict | None = None) -> dict:
    if not 5 <= interval <= 3600:
        raise BridgeError('轮询间隔需在 5–3600 秒之间')
    if batch_plan is not None and not isinstance(batch_plan, dict):
        raise BridgeError('batch_plan 必须是批计划 dict')
    try:
        record = get_local(session)
        stored = record.get('result')
        if stored and stored.get('phase') in FINAL:
            if batch_plan is None:
                if stored['phase'] == 'FAILED' or record.get('stopped'):
                    return {**stored, 'instance_stopped': bool(record.get('stopped'))}
                # A successful run may have been observed before stop failed (or
                # watch --no-stop was used). Retry stop without requiring the VM
                # to remain reachable for a second status download.
                return _finish_success(session, stored, auto_stop)
            # 批模式：终态只是控制面事实；必须按逐单元对账后才能决定停止。
            return _batch_terminal_result(session, stored, batch_plan, auto_stop)
    except BridgeError:
        pass
    count = 0
    while True:
        status = get_status(session)
        record_observation(session, status)
        phase = status.get('phase', 'UNKNOWN')
        print(f'[mlbridge] {session}: {phase}', flush=True)
        if batch_plan is not None and phase not in FINAL:
            # 批模式：先看是否存在执行终局证据（确认停止/ foreground
            # 哨兵）。有且当前状态非终态 ⇒ 运行已中断，交诚实对账。
            try:
                local_probe = get_local(session)
            except BridgeError:
                local_probe = None
            proof = _execution_ended_proof(local_probe)
            if proof is not None:
                return _interrupted_batch_result(session, batch_plan, status,
                                                 proof, auto_stop)
        if phase in FINAL:
            try:
                with _job_lock(session):
                    record = get_local(session)
                    record['result'] = status
                    store_local(session, record)
            except BridgeError:
                pass
            save_terminal_diagnostic(session, status)
            if batch_plan is not None:
                return _batch_terminal_result(session, status, batch_plan,
                                              auto_stop)
            if phase == 'SUCCEEDED':
                return _finish_success(session, status, auto_stop)
            print('[mlbridge] 运行失败；保留 Colab 供排查。请手动停止以免持续计费。', flush=True)
            return {**status, 'instance_stopped': False}
        # Never equate transport errors with death of the remote worker.
        count = count + 1 if phase == 'UNREACHABLE' else 0
        if count >= 6:
            if batch_plan is not None:
                # 连续不可达 + 实例确认停抚 → 执行终局证据成立，交中断
                # 对账；仅有观测失联（无停抚证据）仍绝不裁决。
                try:
                    local_probe = get_local(session)
                except BridgeError:
                    local_probe = None
                proof = _execution_ended_proof(local_probe)
                if proof is not None:
                    return _interrupted_batch_result(session, batch_plan,
                                                     status, proof, auto_stop)
            print('[mlbridge] 连续 6 次无法访问 Colab；任务状态未知，未执行 stop。', flush=True)
            print(f'[mlbridge] 网络恢复后执行: mlbridge watch {session}', flush=True)
            return {'phase': 'UNKNOWN', 'reason': 'consecutive transport failures'}
        time.sleep(interval)
