"""Lightweight Colab adapter, using only official CLI and local Python stdlib.

The ML process is detached *inside Colab* and does not depend on SSH lifetime.
"""
from __future__ import annotations

import json
import os
import re
import secrets
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .core import (BridgeError, default_gpu, default_high_mem, expand, git_commit,
                   GPU_CHOICES, job_spec, proc, project, repo_url, settings, secret_env,
                   SNAPSHOT, state_dir)

FINAL = {'SUCCEEDED', 'FAILED'}
SESSION = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_-]{0,62}$')

# 有界 colab 调用：kernel 通道 (exec/upload) 可能因 Colab 端故障无限阻塞
# （2026-10-19 smoke-aspl-sd15 staging 实测 exec >8 分钟无返回，最终
# "Connection was lost"）。因此所有 colab 子进程必须有超时；重试只针对
# 幂等操作（重传同名文件 / 重复 exec 同一脚本 / 重复 stop），绝不重复
# `colab new`——分配失败时不重试，避免复制计费实例。
EXEC_TIMEOUT = 300      # 单次 exec：staging 脚本均微小，正常 <10s
UPLOAD_TIMEOUT = 300    # 单次 upload：staging 文件均为 KB 级
ALLOC_TIMEOUT = 900     # new：A100 分配可能排队，但仍须有界
STOP_TIMEOUT = 120
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
    path = state_path(session)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    path.chmod(0o600)


def get_local(session: str) -> dict:
    try:
        return json.loads(state_path(session).read_text())
    except FileNotFoundError as exc:
        raise BridgeError(f'找不到本地任务记录 {session}') from exc


def cli(*args, capture=False, timeout=None):
    return proc(['colab', *args], capture=capture, timeout=timeout)


def cli_retry(*args, timeout, attempts=STAGE_ATTEMPTS, backoff=STAGE_BACKOFF):
    """有界重试的幂等 colab 调用；仅在重试预算内重试，不重新分配实例。"""
    last: Exception | None = None
    for attempt in range(attempts):
        if attempt:
            print(f'[mlbridge] colab {args[0]} 失败，{backoff}s 后重试 '
                  f'({attempt + 1}/{attempts})', file=sys.stderr, flush=True)
            time.sleep(backoff)
        try:
            return cli(*args, timeout=timeout)
        except BridgeError as exc:
            last = exc
    raise BridgeError(
        f'colab {args[0]} 连续 {attempts} 次失败（有界放弃，不会重复分配实例）: '
        f'{last}') from last


def get_status(session: str) -> dict:
    with tempfile.TemporaryDirectory(prefix='mlbridge-status-') as t:
        dest = Path(t) / 'state.json'
        try:
            cli('download', '-s', session, '/content/mlbridge_state.json', str(dest),
                capture=True, timeout=75)
            return json.loads(dest.read_text())
        except (BridgeError, ValueError, OSError) as exc:
            return {'phase': 'UNREACHABLE', 'reason': str(exc)}


def shutdown(session: str) -> None:
    # stop 幂等（对已停止 session colab 退出码为 0）；有界重试以穿透瞬时
    # 控制面抖动。只有 cli 成功返回后才标记本地 stopped:true。
    cli_retry('stop', '-s', session, timeout=STOP_TIMEOUT)
    try:
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


def starter_script() -> str:
    # Explicitly close inherited fds, redirect output and detach process.
    # The launched remote_worker survives closing this colab exec / SSH.
    return '''import os,subprocess,sys\nfrom pathlib import Path\np=Path("/content/mlbridge_run.log")\nwith p.open("ab",buffering=0) as f:\n    proc=subprocess.Popen([sys.executable,"/content/mlbridge_worker.py"],\n        cwd="/content",stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,\n        start_new_session=True,close_fds=True)\nprint("MLBridge remote PID:",proc.pid)\n'''


def put_script(session: str, contents: str):
    with tempfile.TemporaryDirectory(prefix='mlbridge-colab-') as t:
        p = Path(t) / 'script.py'
        p.write_text(contents)
        # exec 是幂等的（脚本可安全重复执行），因此允许有界重试。
        cli_retry('exec', '-s', session, '-f', str(p), timeout=EXEC_TIMEOUT)


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
    # 覆盖同一远端路径，重试是幂等的。
    cli_retry('upload', '-s', session, str(src), f'/content/{remote_basename}',
              timeout=UPLOAD_TIMEOUT)


def _staging_failed(session: str, starter_sent: bool) -> None:
    """staging 失败后的安全回收：绝不重复分配，也尽力不留计费实例。

    starter 脚本之前失败 ⇒ worker 一定未启动（此前只上传文件），直接停止。
    starter 阶段失败 ⇒ exec 回复丢失但内核可能已执行，worker 或已启动；
    先有界探测状态文件，仅确认未运行时才停止。探测不可达时也按未启动处理
    （此时任务仅启动数秒，重跑代价远低于闲置计费实例），并明确打印该决定。
    """
    try:
        record = get_local(session)
        record['staging_failed'] = True
        store_local(session, record)
    except BridgeError:
        pass
    if starter_sent:
        status = get_status(session)
        phase = status.get('phase', 'UNKNOWN')
        if phase in WORKER_PHASES:
            print(f'[mlbridge] 远端 worker 已启动 (phase={phase})；不停止实例。'
                  f'请用 mlbridge watch {session} 接管。', file=sys.stderr, flush=True)
            return
        print(f'[mlbridge] 状态探测不可达 ({phase})；按 worker 未启动处理并停止实例。',
              file=sys.stderr, flush=True)
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
           session: str | None = None,
           allow_empty_input: bool = False) -> str:
    cfg = project(name)
    store = settings()
    # Preflight: do not start a GPU for missing credentials or a malformed
    # private-repository URL. Fail before any Colab allocation.
    gh_auth = github_auth()
    if gh_auth:
        require_github_https_url(cfg['repository'])
    secret_env(store)
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
    request = ['new', '-s', session]
    if gpu_norm in ('CPU', 'NONE'):
        print(f'[mlbridge] Colab 分配中: {session} (CPU)', flush=True)
    else:
        print(f'[mlbridge] Colab 分配中: {session} ({gpu_norm})', flush=True)
        request.extend(('--gpu', gpu_norm))
    if use_high_mem:
        request.append('--high-mem')
    # 先写本地记录再分配：即使 new 本身挂死/失败，也存在可 stop/审计的记录。
    store_local(session, {'project': name, 'job': job, 'session': session,
                          'git_commit': commit, 'inputs': inputs, 'stopped': False})
    try:
        cli(*request, timeout=ALLOC_TIMEOUT)
    except BridgeError:
        # 分配失败绝不重试 new（避免复制计费实例）；若实例实际已创建，
        # stop 是幂等的，最坏情况只是对不存在的 session 空跑一次。
        try:
            shutdown(session)
        except BridgeError:
            pass
        print(f'[mlbridge] 分配失败或超时；若实例实际已创建请用 colab sessions '
              f'检查并 mlbridge stop {session}。', file=sys.stderr, flush=True)
        raise
    try:
        # These are tiny files, not the large dataset.
        starter_sent = False
        with tempfile.TemporaryDirectory(prefix='mlbridge-job-') as t:
            job_json = Path(t) / 'job.json'
            job_json.write_text(json.dumps(plan))
            job_json.chmod(0o600)
            worker = Path(__file__).with_name('remote_worker.py')
            put_script(session, prepare_script())
            stage(session, worker, 'mlbridge_worker.py')
            stage(session, job_json, 'mlbridge_job.json')
            stage(session, expand(store['rclone_config']), 'mlbridge_rclone.conf')
            stage(session, expand(store['password_file']), 'mlbridge_restic_password')
            if gh_auth:
                # 私有仓库凭据最后上传;上传后立刻在 VM 上收紧权限并生成 askpass。
                # 窗口期内凭据仅对同一 OS 用户可读(见 README/安全说明)。
                _stage_github_auth(session, *gh_auth)
            starter_sent = True
            put_script(session, starter_script())
    except Exception:
        # 安全回收：确认 worker 未运行时停止实例，避免留下计费的空闲 GPU；
        # 绝不重复 colab new。错误原样上抛。
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


def watch(session: str, *, interval: int = 20, auto_stop: bool = True) -> dict:
    if not 5 <= interval <= 3600:
        raise BridgeError('轮询间隔需在 5–3600 秒之间')
    try:
        record = get_local(session)
        stored = record.get('result')
        if stored and stored.get('phase') in FINAL:
            if stored['phase'] == 'FAILED' or record.get('stopped'):
                return {**stored, 'instance_stopped': bool(record.get('stopped'))}
            # A successful run may have been observed before stop failed (or
            # watch --no-stop was used). Retry stop without requiring the VM
            # to remain reachable for a second status download.
            return _finish_success(session, stored, auto_stop)
    except BridgeError:
        pass
    count = 0
    while True:
        status = get_status(session)
        phase = status.get('phase', 'UNKNOWN')
        print(f'[mlbridge] {session}: {phase}', flush=True)
        if phase in FINAL:
            try:
                record = get_local(session)
                record['result'] = status
                store_local(session, record)
            except BridgeError:
                pass
            if phase == 'SUCCEEDED':
                return _finish_success(session, status, auto_stop)
            print('[mlbridge] 运行失败；保留 Colab 供排查。请手动停止以免持续计费。', flush=True)
            return {**status, 'instance_stopped': False}
        # Never equate transport errors with death of the remote worker.
        count = count + 1 if phase == 'UNREACHABLE' else 0
        if count >= 6:
            print('[mlbridge] 连续 6 次无法访问 Colab；任务状态未知，未执行 stop。', flush=True)
            print(f'[mlbridge] 网络恢复后执行: mlbridge watch {session}', flush=True)
            return {'phase': 'UNKNOWN', 'reason': 'consecutive transport failures'}
        time.sleep(interval)
