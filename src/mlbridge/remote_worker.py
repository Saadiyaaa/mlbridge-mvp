"""Uploaded unmodified to Colab; executes a job independent of SSH.

Standard library only, deliberately *not* imported by the research project.
"""
from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import subprocess
import threading
import time
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path

BASE = Path('/content')
STATE = BASE / 'mlbridge_state.json'
PLAN = BASE / 'mlbridge_job.json'
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


# plan 级 execution_mode（v1）：缺省/ 'detached' = 旧行为（starter 在
# cell 内 Popen 分离启动 worker）；'foreground' = cell 内同步运行。
_EXECUTION_MODES = ('detached', 'foreground')


def validate_execution_mode(plan: object) -> str:
    """校验 plan 级 execution_mode；在任何副作用（含心跳启动与执行锁内
    的任何写入）之前 fail closed。

    - 键缺失或 'detached'：与旧行为逐字节一致（legacy 与批皆可），
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


def main():
    """唯一入口：先拿 BASE 执行锁再触碰任何共享文件。

    锁范围：心跳启动/停止、读计划、批校验与 claim、clone/restore/setup、
    执行/状态写入、异常记录、凭据清理；执行锁最后释放。竞争 worker 在锁
    处被拒绝，不写 STATE、不删凭据。legacy 与批模式共用同一把锁并保持
    各自既有顺序执行语义。
    """
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
                return
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
    """unit_batch v1 顺序执行切片：validate（纯校验）→ claim（同批重复
    终态/运行记录拒绝）→ 一次 clone/restore/setup → 逐 unit 执行/发布/回执。

    进程间互斥由 main 持有的 BASE 执行锁提供（全程）；本函数不再自建锁。
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


if __name__ == '__main__':
    main()
