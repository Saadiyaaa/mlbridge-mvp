"""Uploaded unmodified to Colab; executes a job independent of SSH.

Standard library only, deliberately *not* imported by the research project.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

BASE = Path('/content')
STATE = BASE / 'mlbridge_state.json'
PLAN = BASE / 'mlbridge_job.json'


def status(phase, **fields):
    info = {}
    if STATE.exists():
        try:
            info = json.loads(STATE.read_text())
        except (ValueError, OSError):
            pass
    info.update(phase=phase, updated=datetime.now(timezone.utc).isoformat(), **fields)
    temp = STATE.with_suffix('.tmp')
    temp.write_text(json.dumps(info, ensure_ascii=False, indent=2))
    temp.replace(STATE)


def run(args, *, cwd=None, env=None):
    print('+ ' + ' '.join(args), flush=True)
    subprocess.run(args, cwd=cwd, env=env, check=True)


# restic < 0.16.2 没有 restore --overwrite；它恢复时本来就无条件覆盖。
# 老版本如果被传入该参数反而会报 unknown flag 而整 job 失败。
_OVERWRITE_MIN = (0, 16, 2)


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


def main():
    plan = json.loads(PLAN.read_text())
    os.chmod(BASE / 'mlbridge_rclone.conf', 0o600)
    os.chmod(BASE / 'mlbridge_restic_password', 0o600)
    env = dict(os.environ)
    env['RCLONE_CONFIG'] = str(BASE / 'mlbridge_rclone.conf')
    env['RESTIC_PASSWORD_FILE'] = str(BASE / 'mlbridge_restic_password')
    repo = plan['restic_repository']
    root = BASE / 'mlbridge-workspace'
    try:
        status('PREPARING', job_id=plan['job_id'], session=plan['session'])
        missing = [x for x in ['restic', 'rclone', 'git'] if not shutil.which(x)]
        if missing:
            run(['apt-get', 'update', '-qq'])
            run(['apt-get', 'install', '-y', '-qq', *missing])
        # GitHub must contain the pinned commit. Private repository auth is not
        # automatically copied from the local machine in this first version.
        run(['git', 'clone', '--no-checkout', plan['repository'], str(root)])
        run(['git', 'checkout', plan['commit']], cwd=root)
        status('RESTORING')
        # 快照是输入数据的唯一契约：快照中与 git tracked 文件同路径的内容
        # 必须"快照优先"，而不是被 --overwrite never 静默跳过。
        # restic ≥ 0.16.2 显式传 --overwrite always，老版本缺省即覆盖，语义一致。
        for snapshot in plan['inputs']:
            run(['restic', '-r', repo, 'restore', snapshot,
                 '--target', str(root), *restore_args()], env=env)
        for setup in plan['setup']:
            run(setup, cwd=root, env=env)
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
        status('SUCCEEDED', elapsed_seconds=elapsed, snapshot_id=result,
               finished=datetime.now(timezone.utc).isoformat())
    except BaseException as exc:
        status('FAILED', error=f'{type(exc).__name__}: {exc}',
               finished=datetime.now(timezone.utc).isoformat())
        raise
    finally:
        # Only remote ephemeral credential copies. The parent Colab VM and local
        # originals are untouched.
        for name in ('mlbridge_rclone.conf', 'mlbridge_restic_password'):
            (BASE / name).unlink(missing_ok=True)


if __name__ == '__main__':
    main()
