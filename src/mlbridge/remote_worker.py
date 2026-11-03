"""Uploaded unmodified to Colab; executes a job independent of SSH.

Standard library only, deliberately *not* imported by the research project.

 Besides the job flow it provides the research driver a stable stdlib-only
per-key checkpoint CLI::

    python /content/mlbridge_worker.py checkpoint --request REQUEST.json

The interface is only valid inside the environment that worker main()
explicitly exports (MLBRIDGE_CHECKPOINT_PROTOCOL / MLBRIDGE_CHECKPOINT_WORKER).
For one checkpoint key it performs exactly one restic backup and writes an
atomic response file afterwards: no retries, no automatic restart of the
backup, no VM stop, and it never reports a broken session as successful.
MLBRIDGE_WORKER_BASE exists only for offline tests; on a real VM it is unset
and the default /content applies.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path

# MLBRIDGE_WORKER_BASE 仅用于离线测试；真实 VM 不设置，缺省 /content。
BASE = Path(os.environ.get('MLBRIDGE_WORKER_BASE') or '/content')
STATE = BASE / 'mlbridge_state.json'
PLAN = BASE / 'mlbridge_job.json'

# -------------------- per-key checkpoint protocol (v1) --------------------
CHECKPOINT_PROTOCOL = 'v1'
CHECKPOINT_TIMEOUT_SECONDS = 1800        # 单次 backup 默认有界时限
CHECKPOINT_MAX_TIMEOUT_SECONDS = 43200   # 显式 --timeout 上限（12 小时）
CHECKPOINT_MAX_ROOTS = 16                # roots 数上限（研究流程每人为 3）
CHECKPOINT_MAX_KEY_CHARS = 256
CHECKPOINT_MAX_METADATA_BYTES = 4 * 1024 * 1024
# VM 本地有界日志：超过 256KB 时裁剪保留最近 64KB。
CHECKPOINT_LOG = BASE / 'mlbridge_checkpoint.log'
CHECKPOINT_LOG_MAX_BYTES = 256 * 1024
CHECKPOINT_LOG_KEEP_BYTES = 64 * 1024
# 请求 JSON 只接受这些字段；未知字段 fail closed。provenance 显式携带
# backend/host/GPU 等来源属性，由 caller 提供，MLBridge 原样透传（不写死
# GPU 型号、不虚构）。
CHECKPOINT_REQUEST_FIELDS = frozenset(
    {'protocol', 'project', 'session', 'commit', 'key', 'roots', 'metadata',
     'response', 'provenance'})
_HEX64 = re.compile(r'^[a-f0-9]{64}$')
_COMMIT = re.compile(r'^[a-f0-9]{40}$')
# 快照 tag 方案：host 侧 discovery 按 project/session/protocol key 精确 tag
# AND 过滤（见 mlbridge.checkpoints）。
_SESSION_TAG_PREFIX = 'mlbridge-session-checkpoint-'
_PROTOCOL_TAG = 'mlbridge-checkpoint-' + CHECKPOINT_PROTOCOL
_KEY_TAG_PREFIX = 'mlbridge-checkpoint-key-'
_META_TAG_PREFIX = 'mlbridge-checkpoint-meta-'


class CheckpointError(Exception):
    """checkpoint 请求处理失败；exit_code 决定 CLI 返回码。"""
    exit_code = 2


class CheckpointRefused(CheckpointError):
    """协议环境/请求校验拒绝：发生在任何 backup 调用之前。"""
    exit_code = 2


class CheckpointAlreadyRecorded(CheckpointError):
    """同 key 已有 marker/response：显式拒绝，绝不重复 backup。"""
    exit_code = 3


class CheckpointFailed(CheckpointError):
    """backup 未启动或硬错误且无已建快照证据：已持久记录 FAILED。"""
    exit_code = 4


class CheckpointAmbiguous(CheckpointError):
    """超时 / 摘要缺失或 snapshot_id 非法 / response 落盘失败：已持久记录
    AMBIGUOUS；由 host 侧按 tag+intent 重发现，绝不做第二次 upload。"""
    exit_code = 5

# Optional private-repository credentials staged by colab_runner.launch().
GIT_AUTH = 'mlbridge_git_auth'
GIT_ASKPASS = 'mlbridge_git_askpass.sh'
# RUNNING/PUBLISHING 阶段定期写入心跳。
HEARTBEAT_INTERVAL = 300  # 秒；有界
HEARTBEAT_PHASES = frozenset({'RUNNING', 'PUBLISHING'})

# BASE 执行锁：唯一所有权原语（legacy 与批模式共用一把锁）。必须在任何
# STATE 写入、心跳启动、凭据 chmod、cleanup、clone 之前获得；非 owner
# 绝不写共享 STATE、绝不删除任何凭据。优先非阻塞拒绝竞争 worker，而不是
# 阻塞等待造成旧 cleanup 与新任务的竞态；锁在 finally 中最后释放。
EXECUTION_LOCK_NAME = 'mlbridge_execution.lock'

# 仅 owner 可清理的 ephemeral 凭据副本。
_CREDENTIAL_NAMES = ('mlbridge_rclone.conf', 'mlbridge_restic_password',
                     GIT_AUTH, GIT_ASKPASS)

_IDENT = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,62}$")

# 进程内串行化（主线程终态 vs 心跳线程）；跨进程互斥由 BASE 执行锁提供，
# 不靠进程内 thread lock 声称跨进程安全。
_WRITE_LOCK = threading.Lock()


class OwnershipBusy(RuntimeError):
    """BASE 执行锁被另一个 worker 持有；本进程未获得所有权。"""


def _atomic_write(info: dict) -> None:
    # 唯一临时名：跨线程/跨多次写入不复用同一个 .tmp，原子替换。
    temp = STATE.with_name(f"{STATE.name}.tmp-{os.getpid()}-{time.monotonic_ns()}")
    try:
        temp.write_text(json.dumps(info, ensure_ascii=False, indent=2))
        temp.replace(STATE)
    finally:
        with suppress(OSError):
            temp.unlink()


def status(phase, *, fresh: bool = False, **fields):
    """写入共享状态视图。fresh=True 时丢弃已有字段重写（用于把其他 job
    的残留终态覆盖为本 job 的失败，避免旧 snapshot_id 等字段混入）。"""
    with _WRITE_LOCK:
        info = {}
        if not fresh and STATE.exists():
            try:
                info = json.loads(STATE.read_text())
            except (ValueError, OSError):
                pass
        # status() 只在阶段变更时被调用：终态保留最后一次 heartbeat
        # 时间戳是有价值的诊断信息（worker 活到何时），不在这里丢弃。
        info.update(phase=phase, updated=datetime.now(timezone.utc).isoformat(), **fields)
        _atomic_write(info)


def _heartbeat_once() -> bool:
    """单次心跳步进：仅当当前阶段在 HEARTBEAT_PHASES 时刷新时间戳。

    返回是否写了状态文件。绝不修改 phase 及其他字段，因此不会与主线程
    写入的终态（SUCCEEDED/FAILED）竞态覆盖。
    """
    with _WRITE_LOCK:
        try:
            if not STATE.exists():
                return False
            info = json.loads(STATE.read_text())
            if info.get('phase') not in HEARTBEAT_PHASES:
                return False
            info['heartbeat'] = datetime.now(timezone.utc).isoformat()
            _atomic_write(info)
            return True
        except (ValueError, OSError):
            return False


def _start_heartbeat() -> tuple[threading.Thread, threading.Event]:
    """启动心跳线程（仅由持有 BASE 执行锁的 owner 调用）。
    返回 (thread, stop_event)；stop_event 置位后线程有界退出。"""
    stop = threading.Event()

    def beat():
        while not stop.wait(HEARTBEAT_INTERVAL):
            try:
                _heartbeat_once()
            except Exception:  # 心跳绝不能杀死 worker
                continue

    thread = threading.Thread(target=beat, daemon=True, name='mlbridge-heartbeat')
    thread.start()
    return thread, stop


def _stop_heartbeat(handle: tuple[threading.Thread, threading.Event]) -> None:
    """正常与异常路径都停止心跳并 join（有意界），避免 main 多次调用
    残留 daemon 心跳在锁外继续写共享 STATE。"""
    thread, stop = handle
    stop.set()
    thread.join(timeout=max(HEARTBEAT_INTERVAL, 10.0))


def run(args, *, cwd=None, env=None):
    print('+ ' + ' '.join(args), flush=True)
    subprocess.run(args, cwd=cwd, env=env, check=True)


def _remove_git_auth():
    """删除临时 GitHub 凭据与 askpass 脚本（仅在私有仓库克隆时存在）。"""
    for name in (GIT_AUTH, GIT_ASKPASS):
        (BASE / name).unlink(missing_ok=True)


def clone_repository(repository: str, commit: str, root: Path):
    """克隆并 checkout。token 绝不进入 argv、环境变量、remote URL 或日志。

    私有仓库:凭据在 0600 文件里,由 GIT_ASKPASS 可执行脚本按提示读取;
    GIT_TERMINAL_PROMPT=0 禁止任何交互式提示。克隆结束(无论成败)后
    立即删除凭据文件与 askpass 脚本。仅由持有执行锁的 owner 调用。
    """
    auth, askpass = BASE / GIT_AUTH, BASE / GIT_ASKPASS
    private = auth.is_file() and askpass.is_file()
    try:
        # GitHub must contain the pinned commit.
        if root.exists():
            shutil.rmtree(root, ignore_errors=True)
        if private:
            os.chmod(auth, 0o600)
            env = dict(os.environ)
            env['GIT_ASKPASS'] = str(askpass)
            env['GIT_TERMINAL_PROMPT'] = '0'
            run(['git', 'clone', '--no-checkout', repository, str(root)], env=env)
        else:
            run(['git', 'clone', '--no-checkout', repository, str(root)])
        run(['git', 'checkout', commit], cwd=root)
    finally:
        if private:
            _remove_git_auth()


# restic < 0.17.0 没有 restore --overwrite；它恢复时缺省覆盖已有文件。
# 老版本如果被传入该参数反而会报 unknown flag 而整 job 失败。
_OVERWRITE_MIN = (0, 17, 0)


def restore_args(mode: str = 'always') -> list[str]:
    """返回快照恢复是否符合快照优先语义所需参数（按 restic 实际版本自适应）。"""
    try:
        p = subprocess.run(['restic', 'version'], text=True, capture_output=True,
                           timeout=20)
        m = re.search(r'restic (\d+)\.(\d+)\.(\d+)', p.stdout or '')
        if m and tuple(int(g) for g in m.groups()) >= _OVERWRITE_MIN:
            return ['--overwrite', mode]
    except (subprocess.SubprocessError, ValueError):
        pass
    return []


# plan 级 execution_mode（v1）：缺省/ 'detached' = starter 在
# cell 内 Popen 分离启动 worker；'foreground' = cell 内同步运行。
_EXECUTION_MODES = ('detached', 'foreground')


def validate_execution_mode(plan: object) -> str:
    """校验 plan 级 execution_mode；在任何副作用（含心跳启动与执行锁内
    的任何写入）之前 fail closed。

    - 键缺失或 'detached'：legacy 与批皆可，
      job 载荷中不写入该键。
    - 'foreground'：仅批模式（plan 含 unit_batch 对象）允许；cell 内
      同步运行 worker，本地 exec 连接断开/超时可能随 cell 终止执行。
    - 其他任何值/类型、以及 legacy plan 带 foreground 一律拒绝。
    """
    if not isinstance(plan, dict) or 'execution_mode' not in plan:
        return 'detached'
    mode = plan['execution_mode']
    if mode not in _EXECUTION_MODES:
        raise RuntimeError(
            f'plan.execution_mode 必须是 {"/".join(_EXECUTION_MODES)} 的字符串，'
            f'拒绝其他值: {mode!r}')
    if mode == 'foreground' and not isinstance(plan.get('unit_batch'), dict):
        raise RuntimeError(
            'execution_mode=foreground 仅在批模式（plan 含 unit_batch 对象）'
            '允许；legacy 任务拒绝该值')
    return mode


def _acquire_execution_lock() -> int:
    """非阻塞独占获取 BASE 执行锁。失败立刻拒绝（OwnershipBusy），
    绝不阻塞等待、绝不触碰共享文件。"""
    path = BASE / EXECUTION_LOCK_NAME
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        os.close(fd)
        raise OwnershipBusy(
            '另一个 worker 正持有 BASE 执行锁；本进程未获得所有权，拒绝执行，'
            '未写任何 STATE、未删除任何凭据') from exc
    return fd


def _release_execution_lock(fd: int) -> None:
    with suppress(OSError):
        fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)


def _job_identity(plan: object) -> str | None:
    """已确认的 job 身份：plan 中合法形态的 job_id；否则 None（未确认）。"""
    if isinstance(plan, dict):
        value = plan.get('job_id')
        if isinstance(value, str) and _IDENT.fullmatch(value):
            return value
    return None


def main(argv: list[str] | None = None) -> int:
    """唯一入口。

    无参数 = 执行 job 流程（先拿 BASE 执行锁再触碰任何共享文件）。
    ``checkpoint --request ...`` = 逐 key 检查点接口：由 research driver
    在**同一个** job 会话里调用，此时 worker 已持有执行锁，因此该子命令
    必须在取锁之前分流。未知子命令 fail-closed。
    """
    if argv is None:
        argv = []
    if argv and argv[0] == 'checkpoint':
        parser = argparse.ArgumentParser(
            prog='mlbridge_worker.py checkpoint',
            description='按唯一 checkpoint key 一次 backup + 原子 response；'
                        '同 key 重调拒绝、非零/超时/缺 id 持久记录，无自动重试。')
        parser.add_argument('--request', required=True,
                            help='checkpoint 请求 JSON 文件路径（在同一 job '
                                 '会话里由 driver 生成）')
        parser.add_argument('--timeout', type=int,
                            default=CHECKPOINT_TIMEOUT_SECONDS,
                            help=f'单次 backup 有界时限 1..'
                                 f'{CHECKPOINT_MAX_TIMEOUT_SECONDS} 秒（默认 '
                                 f'{CHECKPOINT_TIMEOUT_SECONDS}）')
        parsed = parser.parse_args(argv[1:])
        return checkpoint_flow(parsed.request, parsed.timeout)
    if argv:
        print(f'[mlbridge worker] 未知子命令 {argv[0]!r}；无参数才执行 job 流程',
              file=sys.stderr, flush=True)
        return 2

    lock_fd = None
    identity = None
    try:
        try:
            lock_fd = _acquire_execution_lock()
        except OwnershipBusy:
            raise
        plan = json.loads(PLAN.read_text())
        # execution_mode 校验在任何心跳/锁内写入之前（fail closed）。
        validate_execution_mode(plan)
        identity = _job_identity(plan)
        heartbeat = _start_heartbeat()
        try:
            # plan['unit_batch'] 键存在（含显式 null）即走批模式校验：
            # 显式 null 属于非法批计划，必须在任何 clone/restore/setup
            # 副作用之前被拒绝；legacy plan（无该键）照旧。
            if isinstance(plan, dict) and 'unit_batch' in plan:
                _run_batch(plan)
                return 0
            _run_legacy(plan)
        finally:
            _stop_heartbeat(heartbeat)
    except OwnershipBusy:
        raise
    except BaseException as exc:
        _write_failed_if_open(f'{type(exc).__name__}: {exc}', identity)
        raise
    finally:
        if lock_fd is not None:
            try:
                # Only remote ephemeral credential copies. The parent Colab VM
                # and local originals are untouched.
                for name in _CREDENTIAL_NAMES:
                    (BASE / name).unlink(missing_ok=True)
            finally:
                _release_execution_lock(lock_fd)
    return 0


def _write_failed_if_open(error: str, job_id: str | None = None) -> None:
    """兜底补写终态 FAILED；只动属于本 job / 本 batch 的终态：

    - 身份已确认且已有终态属于同一 job => 保留（例如批模式已写成的
      SUCCEEDED/FAILED 逐单元 outcome，绝不替补旧成功掩盖新失败）；
    - 身份无法确认（计划不可读 / job_id 非法形态）=> 绝不改写已有 STATE：
      那份终态属于别的 job，不可冒充本 job 的结果；失败由异常与日志报告。
      仅当 STATE 不存在时写入不含身份的最小启动失败记录；
    - 有身份且已有终态来自其他 job（残留）=> 用 fresh 覆盖为本次失败，
      不继承残留的 snapshot_id 等字段；写真实 job_id。
    """
    try:
        current = json.loads(STATE.read_text()) if STATE.exists() else {}
    except (ValueError, OSError):
        current = {}
    stale_other = False
    if isinstance(current, dict) and current.get('phase') in ('SUCCEEDED', 'FAILED'):
        if job_id is None:
            return  # 身份未确认：不把其他 job 的终态当本 job 的结果
        if current.get('job_id') == job_id:
            return  # 本 job 已有终态（如批逐单元 outcome），保留
        stale_other = True  # 残留终态属于其他 job：覆盖，不继承旧字段
    elif (job_id is None and isinstance(current, dict) and current.get('job_id')):
        return  # STATE 归属其他 job 且身份未确认：拒绝写他人 STATE
    status('FAILED', fresh=stale_other, error=error,
           finished=datetime.now(timezone.utc).isoformat(),
           **({'job_id': job_id} if job_id is not None else {}))


def _batch_module():
    """opt-in 分支按需加载 sibling batch_worker（standalone，不 import mlbridge）。

    launch 必须显式 stage batch_worker.py 才能在远端使用批模式（现已由
    `mlbridge run --unit-batch` 接入）；缺失时会得到明确的 fail-closed
    错误（见 docs/UNIT_EXECUTION.md 和 docs/RECOVERY.md）。
    """
    import importlib.util
    path = Path(__file__).with_name('batch_worker.py')
    if not path.is_file():
        raise RuntimeError(
            'unit_batch 计划需要 sibling batch_worker.py，但该文件未随任务 '
            'stage（launch 仅在 --unit-batch 批模式显式 stage 该文件）。'
            '拒绝执行任何副作用；'
            '见 docs/UNIT_EXECUTION.md。')
    spec = importlib.util.spec_from_file_location('mlbridge_remote_batch_worker', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _base_env():
    os.chmod(BASE / 'mlbridge_rclone.conf', 0o600)
    os.chmod(BASE / 'mlbridge_restic_password', 0o600)
    env = dict(os.environ)
    env['RCLONE_CONFIG'] = str(BASE / 'mlbridge_rclone.conf')
    env['RESTIC_PASSWORD_FILE'] = str(BASE / 'mlbridge_restic_password')
    # 协议版本与 worker 绝对路径经环境传递（不落 argv/凭据）。不用 checkpoint
    # 的 driver 只是不读这些变量，缺省行为不变。
    env['MLBRIDGE_CHECKPOINT_PROTOCOL'] = CHECKPOINT_PROTOCOL
    env['MLBRIDGE_CHECKPOINT_WORKER'] = _worker_path()
    if (BASE / GIT_AUTH).exists():
        os.chmod(BASE / GIT_AUTH, 0o600)
    return env


def _prepare(plan, root, env):
    """一次 clone/restore/setup（legacy 与批模式共用的唯一环境准备）。"""
    # Clone first so the short-lived GitHub token is removed before any
    # potentially long restic/rclone installation or research setup.
    if not shutil.which('git'):
        run(['apt-get', 'update', '-qq'])
        run(['apt-get', 'install', '-y', '-qq', 'git'])
    # 私有仓库克隆使用临时 GIT_ASKPASS 凭据（见 clone_repository）。
    clone_repository(plan['repository'], plan['commit'], root)
    missing = [x for x in ('restic', 'rclone') if not shutil.which(x)]
    if missing:
        run(['apt-get', 'update', '-qq'])
        run(['apt-get', 'install', '-y', '-qq', *missing])
    status('RESTORING')
    # 快照是输入数据的唯一契约：快照中与 git tracked 文件同路径的内容
    # 必须"快照优先"，而不是被 --overwrite never 静默跳过。
    # restic ≥ 0.17.0 显式传 --overwrite always，老版本缺省即覆盖，语义一致。
    for snapshot in plan['inputs']:
        run(['restic', '-r', plan['restic_repository'], 'restore', snapshot,
             '--target', str(root), *restore_args()], env=env)
    for setup in plan['setup']:
        run(setup, cwd=root, env=env)


def _run_legacy(plan):
    """无 unit_batch 键的 plan：与第一阶段行为逐字节一致。"""
    env = _base_env()
    repo = plan['restic_repository']
    root = BASE / 'mlbridge-workspace'
    status('PREPARING', job_id=plan['job_id'], session=plan['session'])
    _prepare(plan, root, env)
    status('RUNNING', started=datetime.now(timezone.utc).isoformat())
    started = time.monotonic()
    run(plan['command'], cwd=root, env=env)
    elapsed = round(time.monotonic() - started, 1)
    result = ''
    if plan['outputs']:
        status('PUBLISHING', elapsed_seconds=elapsed)
        for relative in plan['outputs']:
            if not (root / relative).exists():
                raise RuntimeError('expected output missing: ' + relative)
        cmd = ['restic', '-r', repo, 'backup', '--json', '--tag',
               'mlbridge-project-' + plan['project'], '--tag',
               'mlbridge-artifact-' + plan['job_id'], *plan['outputs']]
        done = subprocess.run(cmd, cwd=root, env=env, text=True,
                              stdout=subprocess.PIPE, check=True)
        for line in done.stdout.splitlines():
            try:
                item = json.loads(line)
            except ValueError:
                continue
            if item.get('message_type') == 'summary':
                result = item.get('snapshot_id', '')
        if not result:
            raise RuntimeError('restic produced no final snapshot ID')
    # 先把最终状态留在 VM 上供本地 watch 读取；只有 watch 确认读取后
    # 才可按 auto_stop 选择释放实例，不能在这里主动 unassign。
    status('SUCCEEDED', elapsed_seconds=elapsed, snapshot_id=result,
           finished=datetime.now(timezone.utc).isoformat())


def _run_batch(plan):
    """unit_batch 顺序执行：validate（纯校验）→ claim（同批重复
    终态/运行记录拒绝）→ 一次 clone/restore/setup → 逐 unit 执行/发布/回执。

    进程间互斥由 main 持有的 BASE 执行锁提供（全程）；本函数使用该锁。
    validate_batch 是纯校验：任何 schema/身份/绑定非法（含显式 null 的
    unit_batch）都在任何副作用（包括凭据 chmod 与 clone）之前被拒绝。
    终态由 executor 写入；main 的兜底只补写身份可确认、executor 未成功
    报告终态时的失败，且绝不覆盖本 batch 已写成的逐单元终态。
    """
    batch_mod = _batch_module()
    batch = batch_mod.validate_batch(plan)
    batch_mod.claim_batch(batch, BASE)
    env = _base_env()
    root = BASE / 'mlbridge-workspace'
    def emit(phase, **fields):
        status(phase, **fields)
    status('PREPARING', job_id=batch['job_id'], session=batch['session'])
    _prepare(plan, root, env)
    batch_mod.execute_batch(batch, root=root, env=env, base=BASE, emit=emit)


# ---------------------------------------------------------------------------
# 每个 checkpoint key 的接口，供 research driver 使用（仅用 stdlib）。
#
# `python /content/mlbridge_worker.py checkpoint --request REQUEST.json`
#
# 返回码：0=SUCCEEDED；2=拒绝（协议环境/请求校验，发生在任何 backup 之前）；
# 3=同 key 已有记录（拒绝，绝不重复 backup）；4=FAILED（启动失败或无快照证据的硬错误）；
# 5=AMBIGUOUS（部分读取失败/非零但已有快照/超时/摘要缺失或id非法/response 落盘失败，须由 host 按
# tag+intent 重发现）。
#
# 身份契约（供中央跨主机原子 claim 对照；中央 claim 表本身属主会话后续
# 阶段）：checkpoint key 本体原样保留（无 NUL/换行/首尾空白），确定性加
# SHA-256 成 key tag；同 key 本 VM 以 marker 先写者胜，跨主机以 host 侧
# `mlbridge checkpoints` 发现记录对照，MLBridge 不猜测也不自动选快照。
# ---------------------------------------------------------------------------


def _worker_path() -> str:
    return str(Path(__file__).resolve())


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _checkpoint_log(event: dict) -> None:
    """追加一条 JSON 行到 VM 本地有界检查点日志，仅存密钥与错误的指纹/摘要，
    绝不输出 repo URL、凭据或完整 restic stderr。"""
    try:
        if CHECKPOINT_LOG.exists() and CHECKPOINT_LOG.stat().st_size > CHECKPOINT_LOG_MAX_BYTES:
            keep = CHECKPOINT_LOG.read_text(errors='replace')[-CHECKPOINT_LOG_KEEP_BYTES:]
            CHECKPOINT_LOG.write_text(keep, encoding='utf-8')
        with CHECKPOINT_LOG.open('a', encoding='utf-8') as fh:
            fh.write(json.dumps({'t': _now_iso(), **event},
                                ensure_ascii=False, default=str) + '\n')
    except OSError:
        pass  # 日志失败不影响主流程


def _excerpt(text: object, limit: int = 2000) -> str:
    """有界错误摘要；bytes 尽力替代解码。"""
    if text is None:
        return ''
    if isinstance(text, bytes):
        text = text.decode('utf-8', 'replace')
    return str(text).strip()[-limit:]


def _redact(text: str, *secrets: str) -> str:
    for s in secrets:
        if s:
            text = text.replace(s, '<redacted>')
    return text


def _rel_parts(value: object, label: str) -> tuple[str, list[str]]:
    """归一化并拒绝相对路径别名；返回 (posix 字符串, parts)。

    拒绝绝对路径、``..``、空、NUL/换行、首尾空白与 ``.git`` 目录；
    ``./`` 与 ``//`` 归一化（inplicit 别名统一后平比较），所有被请求路径
    都以归一化形态使用与备份。
    """
    if not isinstance(value, str) or not value:
        raise CheckpointRefused(f'{label} 必须是非空字符串: {value!r}')
    if '\x00' in value or '\n' in value or '\r' in value:
        raise CheckpointRefused(f'{label} 含 NUL/换行: {value!r}')
    if value != value.strip():
        raise CheckpointRefused(f'{label} 有首尾空白: {value!r}')
    if value.startswith('/'):
        raise CheckpointRefused(f'{label} 必须是相对路径: {value!r}')
    parts = [p for p in value.split('/') if p not in ('', '.')]
    if not parts or '..' in parts:
        raise CheckpointRefused(f'{label} 越界或为空: {value!r}')
    if any(p == '.git' for p in parts):
        raise CheckpointRefused(f'{label} 不允许 .git 目录: {value!r}')
    return '/'.join(parts), parts


def _assert_no_symlink(workspace: Path, parts: list[str], label: str) -> None:
    """逐级检查既存组件不得是软链接（拒绝入库路径别名）。"""
    current = workspace
    for part in parts:
        current = current / part
        if current.is_symlink():
            raise CheckpointRefused(
                f'{label} 含软链接组件，拒绝路径别名: {current}')


def _identity_provenance(req: dict) -> dict[str, str]:
    """校验 caller 提供的 backend/host/GPU provenance（原样透传，不写死任何
    GPU 型号、不虚构、不合并环境）。"""
    provenance = req['provenance']
    if not isinstance(provenance, dict) or not provenance:
        raise CheckpointRefused('provenance 必须是非空 dict: {backend, host, gpu 类字段由 caller 显式提供}')
    if len(provenance) > 32:
        raise CheckpointRefused('provenance 条目数超过上限(32)')
    out: dict[str, str] = {}
    for key, value in provenance.items():
        if not isinstance(key, str) or not key or len(key) > 64:
            raise CheckpointRefused(f'provenance 键必须是 1..64 字符字符串: {key!r}')
        if not isinstance(value, str) or not value or len(value) > 256:
            raise CheckpointRefused(f'provenance[{key!r}] 必须是 1..256 字符字符串')
        if '\x00' in key or '\x00' in value or '\n' in value or '\r' in value:
            raise CheckpointRefused(f'provenance[{key!r}] 含 NUL/换行')
        if key in out:
            raise CheckpointRefused(f'provenance 键重复（JSON 解析后不会重复）: {key!r}')
        out[key] = value
    return out


def _checkpoint_env_guard() -> None:
    """接口只允许在 worker main 显式导出的 command 环境中调用，fail-closed。"""
    protocol = os.environ.get('MLBRIDGE_CHECKPOINT_PROTOCOL')
    if protocol != CHECKPOINT_PROTOCOL:
        raise CheckpointRefused(
            f'MLBRIDGE_CHECKPOINT_PROTOCOL 是 {protocol!r}，应为 {CHECKPOINT_PROTOCOL!r}；'
            'checkpoint 接口必须由 research driver 在 worker 导出的 command 环境内调用')
    worker = os.environ.get('MLBRIDGE_CHECKPOINT_WORKER')
    if not worker or os.path.realpath(worker) != _worker_path():
        raise CheckpointRefused(
            f'MLBRIDGE_CHECKPOINT_WORKER 与当前 worker 不符: {worker!r}')


def _load_plan() -> dict:
    try:
        plan = json.loads(PLAN.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise CheckpointRefused(f'job plan 不可读: {exc}') from exc
    if not isinstance(plan, dict):
        raise CheckpointRefused('job plan 必须是 JSON object')
    for field in ('project', 'session', 'commit', 'restic_repository'):
        value = plan.get(field)
        if not isinstance(value, str) or not value:
            raise CheckpointRefused(f'job plan 缺少 {field}')
    if plan.get('job_id') != plan['session']:
        raise CheckpointRefused('job plan 的 job_id 与 session 不一致，拒绝')
    outputs = plan.get('outputs')
    if not isinstance(outputs, list) or not outputs or \
            any(not isinstance(x, str) or not x for x in outputs):
        raise CheckpointRefused('job plan 无可用 outputs；无授权目录，不可 checkpoint')
    return plan


def _load_request(request_path: str) -> dict:
    """读取并展开请求 JSON；类型/字段/协议不符一律 CheckpointRefused。"""
    path = Path(request_path)
    if not path.is_file():
        raise CheckpointRefused(f'request 文件不存在或不是普通文件: {request_path}')
    try:
        req = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise CheckpointRefused(f'request JSON 不可读或不是合法 JSON: {exc}') from exc
    if not isinstance(req, dict):
        raise CheckpointRefused('request 必须是 JSON object')
    unknown = set(req) - CHECKPOINT_REQUEST_FIELDS
    if unknown:
        raise CheckpointRefused(f'request 含未知字段: {", ".join(sorted(unknown))}')
    missing = CHECKPOINT_REQUEST_FIELDS - set(req)
    if missing:
        raise CheckpointRefused(f'request 缺少字段: {", ".join(sorted(missing))}')
    return req


def _validate_request(req: dict) -> dict[str, str]:
    """基础类型/协议校验（无关 plan）；失败即 CheckpointRefused。返回 provenance。"""
    if req['protocol'] != CHECKPOINT_PROTOCOL:
        raise CheckpointRefused(
            f'请求 protocol 是 {req["protocol"]!r}，应为 {CHECKPOINT_PROTOCOL!r}')
    for field in ('project', 'session'):
        value = req.get(field)
        if not isinstance(value, str) or not value or len(value) > 63:
            raise CheckpointRefused(
                f'请求 {field} 必须是 1..63 字符字符串: {value!r}')
        if '\x00' in value or '\n' in value or '\r' in value:
            raise CheckpointRefused(f'请求 {field} 含 NUL/换行: {value!r}')
    if not isinstance(req['commit'], str) or not _COMMIT.fullmatch(req['commit']):
        raise CheckpointRefused(
            f'请求 commit 必须是完整 40 位十六进制: {req["commit"]!r}')
    key = req['key']
    if not isinstance(key, str) or not key or len(key) > CHECKPOINT_MAX_KEY_CHARS:
        raise CheckpointRefused(
            f'key 必须是 1..{CHECKPOINT_MAX_KEY_CHARS} 字符字符串: {key!r}')
    if '\x00' in key or '\n' in key or '\r' in key or key != key.strip():
        raise CheckpointRefused('key 不得含 NUL/换行/首尾空白')
    for field in ('roots', 'metadata', 'response'):
        if field == 'roots' and not isinstance(req[field], list):
            raise CheckpointRefused('roots 必须是数组')
        if field != 'roots' and not isinstance(req[field], str):
            raise CheckpointRefused(f'{field} 必须是字符串: {req[field]!r}')
    return _identity_provenance(req)


def _build_context(req: dict, plan: dict) -> dict:
    """校验成功后的完整上下文（workspace 相对防越界 + 授权范围检查）。"""
    workspace = BASE / 'mlbridge-workspace'
    if workspace.is_symlink():
        raise CheckpointRefused('workspace 本身是软链接，拒绝')
    if req['project'] != plan['project']:
        raise CheckpointRefused(
            f'请求 project {req["project"]!r} 与 plan {plan["project"]!r} 不符')
    if req['session'] != plan['session']:
        raise CheckpointRefused(
            f'请求 session {req["session"]!r} 与 plan {plan["session"]!r} 不符')
    if req['commit'] != plan['commit']:
        raise CheckpointRefused(
            f'请求 commit {req["commit"]!r} 与 plan {plan["commit"]!r} 不符')
    authorized: list[list[str]] = []
    for output in plan['outputs']:
        _, parts = _rel_parts(output, 'plan.outputs')
        if parts not in authorized:
            authorized.append(parts)
    roots: list[str] = []
    root_list = req['roots']
    if not isinstance(root_list, list) or not root_list:
        raise CheckpointRefused('roots 必须是非空数组')
    if len(root_list) > CHECKPOINT_MAX_ROOTS:
        raise CheckpointRefused(f'roots 数量超过上限({CHECKPOINT_MAX_ROOTS})')
    for entry in root_list:
        rel, parts = _rel_parts(entry, 'roots')
        if rel in roots:
            raise CheckpointRefused(f'roots 重复: {rel}')
        _assert_under_authorized(parts, authorized, f'roots[{rel}]')
        _assert_no_symlink(workspace, parts, f'roots[{rel}]')
        dest = workspace / rel
        if not dest.is_dir() and not dest.is_file():
            raise CheckpointRefused(f'roots[{rel}] 不存在或不是普通文件/目录: {dest}')
        roots.append(rel)
    meta_rel, meta_parts = _rel_parts(req['metadata'], 'metadata')
    _assert_under_authorized(meta_parts, authorized, f'metadata[{meta_rel}]')
    _assert_no_symlink(workspace, meta_parts, f'metadata[{meta_rel}]')
    meta_abs = workspace / meta_rel
    if not meta_abs.is_file():
        raise CheckpointRefused(f'metadata 必须是当前已存在的普通文件: {meta_abs}')
    meta_bytes = meta_abs.read_bytes()
    if len(meta_bytes) > CHECKPOINT_MAX_METADATA_BYTES:
        raise CheckpointRefused(
            f'metadata 文件超过上限({CHECKPOINT_MAX_METADATA_BYTES} 字节)')
    if not meta_bytes.strip():
        raise CheckpointRefused('metadata intent 文件不得为空')
    meta_sha = hashlib.sha256(meta_bytes).hexdigest()
    resp_rel, resp_parts = _rel_parts(req['response'], 'response')
    _assert_under_authorized(resp_parts, authorized, f'response[{resp_rel}]')
    _assert_no_symlink(workspace, resp_parts, f'response[{resp_rel}]')
    root_parts = [root.split('/') for root in roots]
    if any(a == b or a[:len(b)] == b or b[:len(a)] == a
           for i, a in enumerate(root_parts) for b in root_parts[i + 1:]):
        raise CheckpointRefused('roots 不得互相包含或重叠')
    if resp_rel == meta_rel or any(resp_parts[:len(parts)] == parts
                                   or parts[:len(resp_parts)] == resp_parts
                                   for parts in root_parts):
        raise CheckpointRefused('response 不得与 metadata 相同，也不得与任一 root 包含或重叠')
    resp_abs = workspace / resp_rel
    if resp_abs.is_symlink() or (resp_abs.exists() and not resp_abs.is_file()):
        raise CheckpointRefused(
            f'response 路径不能是软链接/目录/设备等: {resp_abs}')
    key_hash = hashlib.sha256(req['key'].encode('utf-8')).hexdigest()
    return {
        'workspace': workspace, 'roots': roots,
        'project': plan['project'], 'session': plan['session'],
        'commit': req['commit'],
        'metadata': meta_rel, 'metadata_abs': meta_abs,
        'metadata_sha256': meta_sha, 'metadata_bytes': len(meta_bytes),
        'response': resp_rel, 'response_abs': resp_abs,
        'key_hash': key_hash,
        'key_tag': _KEY_TAG_PREFIX + key_hash,
        'meta_tag': _META_TAG_PREFIX + meta_sha,
        'restic_repository': plan['restic_repository'],
    }


def _assert_under_authorized(parts: list[str], authorized: list[list[str]],
                             label: str) -> None:
    for out in authorized:
        if parts == out or parts[:len(out)] == out:
            return
    raise CheckpointRefused(
        f'{label} 不在授权 plan.outputs 范围内（完整根或其子路径之外）')


def _marker_path(key_hash: str) -> Path:
    return BASE / 'mlbridge_checkpoint' / f'{key_hash}.json'


def _assert_no_prior_checkpoint(key_hash: str, response_abs: Path) -> None:
    """同 key 已有 marker 或已有 response 都视为已有结果，绝不重复 backup。
    """
    marker = _marker_path(key_hash)
    if marker.exists():
        try:
            prior = json.loads(marker.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            prior = {}
        raise CheckpointAlreadyRecorded(
            f'同 key 已有 checkpoint 记录（marker status={prior.get("status")!r}, '
            f'session={prior.get("session")!r}）；恢复/查证改为 host 侧 '
            '`mlbridge checkpoints` 按 tag 发现，MLBridge 不做第二次 backup')
    if response_abs.exists() or response_abs.is_symlink():
        raise CheckpointAlreadyRecorded(
            f'response 文件已存在，视为已有结果: {response_abs}')


def _claim_marker(key_hash: str, body: dict) -> None:
    """The atomic admission point. Existence checks alone cannot elect a winner.

    A crash during this first write leaves a blocking record, even if its JSON
    is incomplete. Never delete/reclaim it or start another backup implicitly.
    """
    path = _marker_path(key_hash)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    except FileExistsError as exc:
        raise CheckpointAlreadyRecorded('同 key 已有 checkpoint 记录；原子 marker 已被占用') from exc
    with os.fdopen(fd, 'w', encoding='utf-8') as fh:
        json.dump(body, fh, ensure_ascii=False, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _write_marker(key_hash: str, body: dict) -> None:
    _atomic_json(_marker_path(key_hash), body)


def _atomic_json(path: Path, body: dict, *, mode: int = 0o600) -> None:
    """原子写 JSON：先同目录临时文件 + fsync，再 os.replace。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.tmp-{os.getpid()}-{int(time.time() * 1000)}')
    try:
        with open(tmp, 'w', encoding='utf-8') as fh:
            fh.write(json.dumps(body, ensure_ascii=False, indent=2))
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _summary_snapshot_id(stdout: str) -> str | None:
    """严格从 JSON 摘要取 full 64 hex snapshot_id；summary 存在但 id 类型/格式
    非法一律返回 None（AMBIGUOUS），绝不猜测或截断。"""
    found: str | None = None
    for line in (stdout or '').splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        if event.get('message_type') != 'summary':
            continue
        sid = event.get('snapshot_id')
        if isinstance(sid, str) and _HEX64.fullmatch(sid):
            if found is not None and found != sid:
                return None
            found = sid
        else:
            return None
    return found


def _run_backup(ctx: dict, timeout: int) -> dict:
    """恰好一次 restic backup；分类返回，不再重试。"""
    cmd = ['restic', '-r', ctx['restic_repository'], 'backup', '--json',
           '--tag', 'mlbridge-project-' + ctx['project'],
           '--tag', _SESSION_TAG_PREFIX + ctx['session'],
           '--tag', _PROTOCOL_TAG,
           '--tag', ctx['key_tag'],
           '--tag', ctx['meta_tag'],
           *ctx['roots'], ctx['metadata']]
    env = dict(os.environ)  # RESTIC_PASSWORD_FILE/RCLONE_CONFIG 由 command 环境继承
    try:
        done = subprocess.run(cmd, cwd=ctx['workspace'], env=env, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        return {'phase': 'AMBIGUOUS', 'snapshot_id': '', 'rc': None,
                'reason': f'backup 超时(>{timeout}s)；是否已建快照未知，须按 tag 重发现',
                'excerpt': _excerpt(exc.stderr or exc.stdout)}
    except OSError as exc:
        return {'phase': 'FAILED', 'snapshot_id': '', 'rc': None,
                'reason': f'backup 调用未成功启动: {type(exc).__name__}: {exc}',
                'excerpt': ''}
    snapshot_id = _summary_snapshot_id(done.stdout or '')
    if done.returncode != 0:
        # restic exit 3 means an incomplete source read with a snapshot made.
        # Other nonzero returns may also contain a committed snapshot summary.
        # Neither is complete publication; preserve candidate identity and use
        # tag+intent recovery, never FAILED/no-snapshot or automatic re-backup.
        ambiguous = done.returncode == 3 or snapshot_id is not None
        return {'phase': 'AMBIGUOUS' if ambiguous else 'FAILED',
                'snapshot_id': snapshot_id or '', 'rc': done.returncode,
                'reason': f'backup 命令退出码 {done.returncode}' +
                          ('；快照可能已建立但内容不完整，须按 tag 重发现并验收'
                           if ambiguous else ''),
                'excerpt': _excerpt(done.stderr or done.stdout)}
    if snapshot_id is None:
        return {'phase': 'AMBIGUOUS', 'snapshot_id': '', 'rc': 0,
                'reason': '未从 restic JSON 摘要严格取得 full64 snapshot_id；'
                          '快照可能已建立，须按 tag 重发现，不得二次 backup',
                'excerpt': _excerpt(done.stdout)}
    return {'phase': 'SUCCEEDED', 'snapshot_id': snapshot_id, 'rc': 0,
            'reason': '', 'excerpt': ''}


_EXIT_CODES = {'SUCCEEDED': 0, 'FAILED': 4, 'AMBIGUOUS': 5}


def _base_response(req: dict, ctx: dict, provenance: dict[str, str],
                   phase: str) -> dict:
    return {
        'schema': 'mlbridge-checkpoint-response',
        'protocol': CHECKPOINT_PROTOCOL,
        'phase': phase,
        'exit_code': _EXIT_CODES[phase],
        'project': ctx['project'],
        'session': ctx['session'],
        'commit': req['commit'],
        'checkpoint_key': req['key'],
        'checkpoint_key_sha256': ctx['key_hash'],
        'requested_roots': ctx['roots'],
        'metadata': {'path': ctx['metadata'], 'sha256': ctx['metadata_sha256'],
                     'bytes': ctx['metadata_bytes']},
        'provenance': provenance,
        'written': _now_iso(),
    }


def checkpoint_flow(request_path: str, timeout: int) -> int:
    """执行完整 checkpoint 流程：marker 守卫 —> 一次 backup —> 原子 response。

    返回码含义见模块头部。任何失败都在 marker/日志里持久化；本函数不打
    mlbridge_state.json、不重试、不停止 VM，也不会把失败伪造成 SUCCEEDED。"""
    if not isinstance(timeout, int) or not 1 <= timeout <= CHECKPOINT_MAX_TIMEOUT_SECONDS:
        _print_checkpoint_error(CheckpointRefused(
            f'--timeout 须在 1..{CHECKPOINT_MAX_TIMEOUT_SECONDS} 秒: {timeout!r}'))
        return 2
    try:
        _checkpoint_env_guard()
        req = _load_request(request_path)
        provenance = _validate_request(req)
        plan = _load_plan()
        ctx = _build_context(req, plan)
        key_hash = ctx['key_hash']
        try:
            _assert_no_prior_checkpoint(key_hash, ctx['response_abs'])
        except CheckpointAlreadyRecorded as exc:
            _checkpoint_log({'event': 'rejected-existing', 'key_sha256': key_hash})
            raise
        _claim_marker(key_hash, {
            'status': 'PENDING', 'protocol': CHECKPOINT_PROTOCOL,
            'session': ctx['session'], 'commit': req['commit'],
            'checkpoint_key': req['key'], 'checkpoint_key_sha256': key_hash,
            'requested_roots': ctx['roots'], 'metadata_sha256': ctx['metadata_sha256'],
            'response': ctx['response'], 'started': _now_iso(),
            'provenance': provenance,
            'note': '同 marker 表示本压缩点请求已开始；backup 后 response 前丢档时按需清单重发现，绝不重 backup',
        })
        _checkpoint_log({'event': 'backup-start', 'key_sha256': key_hash,
                         'roots': len(ctx['roots'])})
        outcome = _run_backup(ctx, timeout=timeout)
        phase = outcome['phase']
        body = _base_response(req, ctx, provenance, phase)
        body['restic_exit_code'] = outcome['rc']
        if phase == 'SUCCEEDED':
            body['snapshot_id'] = outcome['snapshot_id']
        else:
            if outcome.get('snapshot_id'):
                body['candidate_snapshot_id'] = outcome['snapshot_id']
            body['error'] = outcome['reason']
            body['restic_log_excerpt'] = _redact(outcome['excerpt'],
                                                 ctx['restic_repository'],
                                                 os.environ.get('RESTIC_PASSWORD_FILE', ''),
                                                 os.environ.get('RESTIC_PASSWORD', ''),
                                                 os.environ.get('RCLONE_CONFIG', ''))
        try:
            _atomic_json(ctx['response_abs'], body, mode=0o644)
        except OSError as exc:
            _checkpoint_log({'event': 'response-write-failed',
                             'key_sha256': key_hash, 'phase': phase,
                             'error': _excerpt(str(exc), 300)})
            # A response failure does not erase a committed snapshot. Update
            # the marker if possible, retaining a candidate ID, but never let
            # marker failure trigger another backup or a successful response.
            uncertain = _base_response(req, ctx, provenance, 'AMBIGUOUS')
            uncertain.update(status='AMBIGUOUS', error='response write failed; recover by tags',
                             restic_exit_code=outcome['rc'])
            if outcome.get('snapshot_id'):
                uncertain['candidate_snapshot_id'] = outcome['snapshot_id']
            try:
                _write_marker(key_hash, uncertain)
            except OSError:
                pass  # marker may remain PENDING if its filesystem also failed
            return CheckpointAmbiguous.exit_code
        marker_body = {**body, 'status': phase}
        _write_marker(key_hash, marker_body)
        _checkpoint_log({'event': 'checkpoint-' + phase.lower(),
                         'key_sha256': key_hash,
                         'snapshot_id': body.get('snapshot_id', '')})
        return _EXIT_CODES[phase]
    except CheckpointRefused as exc:
        _print_checkpoint_error(exc)
        _checkpoint_log({'event': 'refused', 'error': _excerpt(str(exc), 400)})
        return exc.exit_code
    except CheckpointAlreadyRecorded as exc:
        _print_checkpoint_error(exc)
        return exc.exit_code
    except CheckpointFailed as exc:
        _print_checkpoint_error(exc)
        return exc.exit_code
    except CheckpointAmbiguous as exc:
        _print_checkpoint_error(exc)
        return exc.exit_code


def _print_checkpoint_error(exc: CheckpointError) -> None:
    print(f'[mlbridge checkpoint] {type(exc).__name__}: {exc}', file=sys.stderr,
          flush=True)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
