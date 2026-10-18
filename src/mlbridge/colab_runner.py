"""Lightweight Colab adapter, using only official CLI and local Python stdlib.

The ML process is detached *inside Colab* and does not depend on SSH lifetime.
"""
from __future__ import annotations

import json
import re
import secrets
import sys
import tempfile
import time
from pathlib import Path

from .core import (BridgeError, expand, git_commit, job_spec, proc,
                   project, repo_url, settings, secret_env, SNAPSHOT, state_dir)

FINAL = {'SUCCEEDED', 'FAILED'}
SESSION = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_-]{0,62}$')


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
    cli('stop', '-s', session)
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
        cli('exec', '-s', session, '-f', str(p))


def cleanup_script() -> str:
    """删除 stage() 已上传的临时凭据；在 launch 失败路径上尽力执行。"""
    return ("import os\n"
            "for name in ('mlbridge_rclone.conf', 'mlbridge_restic_password'):\n"
            "    try:\n"
            "        os.remove('/content/' + name)\n"
            "    except OSError:\n"
            "        pass\n"
            "print('MLBridge cleanup: temporary credential files removed\\n')\n")


def stage(session: str, src: Path, remote_basename: str):
    cli('upload', '-s', session, str(src), f'/content/{remote_basename}')


def launch(name: str, job: str, inputs: list[str], gpu: str,
           *, allow_dirty: bool = False, session: str | None = None,
           allow_empty_input: bool = False) -> str:
    cfg = project(name)
    store = settings()
    secret_env(store)  # Preflight: do not start a GPU for a missing credential.
    spec = job_spec(cfg, job)
    for snapshot in inputs:
        if not SNAPSHOT.fullmatch(snapshot):
            raise BridgeError('输入必须是明确的 snapshot ID，不能是 latest')
    root = expand(cfg['workspace'])
    commit = git_commit(root, allow_dirty)
    if not inputs and not allow_empty_input:
        raise BridgeError('需要至少一个 --input 快照；无输入任务请显式使用 --allow-empty-input')
    if gpu not in ('T4', 'L4', 'A100', 'H100', 'G4'):
        raise BridgeError('GPU 参数无效')
    session = session or f'mlb-{name[:30]}-{secrets.token_hex(4)}'
    if not SESSION.fullmatch(session):
        raise BridgeError('session ID 不合法或太长')
    plan = dict(project=name, job_id=session, session=session,
                repository=cfg['repository'], commit=commit,
                restic_repository=repo_url(name, store),
                inputs=inputs, **spec)
    print(f'[mlbridge] Colab 分配中: {session} ({gpu})', flush=True)
    cli('new', '-s', session, '--gpu', gpu)
    store_local(session, {'project': name, 'job': job, 'session': session,
                          'git_commit': commit, 'inputs': inputs, 'stopped': False})
    try:
        # These are tiny files, not the large dataset.
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
            put_script(session, starter_script())
    except Exception:
        print('[mlbridge] 初始化失败；为避免误关用户的任务，session 保留供排查。',
              file=sys.stderr)
        # 凭据可能已经到达 VM：让 kernel 尽力删除。仅尽力而为——
        # exec 通道本身可能就是失败的那一环。
        try:
            put_script(session, cleanup_script())
            print('[mlbridge] 已尝试删除 VM 上的临时凭据副本。', file=sys.stderr)
        except Exception:
            print('[mlbridge] 无法删除 VM 上的临时凭据；请手动检查/停止该 session。',
                  file=sys.stderr)
        print(f'[mlbridge] 请检查 colab status -s {session} 并决定是否手动停止。',
              file=sys.stderr)
        raise
    return session


def watch(session: str, *, interval: int = 20, auto_stop: bool = True) -> dict:
    if not 5 <= interval <= 3600:
        raise BridgeError('轮询间隔需在 5–3600 秒之间')
    try:
        record = get_local(session)
        stored = record.get('result')
        if stored and stored.get('phase') in FINAL and record.get('stopped'):
            return stored
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
            if phase == 'SUCCEEDED' and auto_stop:
                shutdown(session)
                print('[mlbridge] 输出已发布，Colab runtime 已释放。', flush=True)
            elif phase == 'FAILED':
                print('[mlbridge] 运行失败；保留 Colab 供排查。请手动停止以免持续计费。', flush=True)
            return status
        # Never equate transport errors with death of the remote worker.
        count = count + 1 if phase == 'UNREACHABLE' else 0
        if count >= 6:
            print('[mlbridge] 连续 6 次无法访问 Colab；任务状态未知，未执行 stop。', flush=True)
            print(f'[mlbridge] 网络恢复后执行: mlbridge watch {session}', flush=True)
            return {'phase': 'UNKNOWN', 'reason': 'consecutive transport failures'}
        time.sleep(interval)
