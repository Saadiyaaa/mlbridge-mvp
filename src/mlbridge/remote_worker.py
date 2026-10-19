"""Uploaded unmodified to Colab; executes a job independent of SSH.

Standard library only, deliberately *not* imported by the research project.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

BASE = Path('/content')
STATE = BASE / 'mlbridge_state.json'
PLAN = BASE / 'mlbridge_job.json'
# Optional private-repository credentials staged by colab_runner.launch().
GIT_AUTH = 'mlbridge_git_auth'
GIT_ASKPASS = 'mlbridge_git_askpass.sh'
# 心跳：RUNNING/PUBLISHING 期间周期性刷新本地状态文件（有界间隔）。
# 心跳不能阻止 VM 消失（/content 仍可能整体丢失）；它的价值是本地把
# “最后看到”与“当前”区分开（配合本地 last-known 记录）。
HEARTBEAT_INTERVAL = 300  # 秒；有界
HEARTBEAT_PHASES = frozenset({'RUNNING', 'PUBLISHING'})

# 状态文件只有一个写入契约：串行化读-改-写 + 原子替换，避免主线程终态
# 与心跳线程互相覆盖字段（2026-10-19 双 A100 实例丢失事故的教训之一）。
_WRITE_LOCK = threading.Lock()


def _atomic_write(info: dict) -> None:
    temp = STATE.with_suffix('.tmp')
    temp.write_text(json.dumps(info, ensure_ascii=False, indent=2))
    temp.replace(STATE)


def status(phase, **fields):
    with _WRITE_LOCK:
        info = {}
        if STATE.exists():
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


def _start_heartbeat() -> threading.Thread:
    def beat():
        while True:
            time.sleep(HEARTBEAT_INTERVAL)
            try:
                _heartbeat_once()
            except Exception:  # 心跳绝不能杀死 worker
                continue

    thread = threading.Thread(target=beat, daemon=True, name='mlbridge-heartbeat')
    thread.start()
    return thread


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
    立即删除凭据文件与 askpass 脚本。公开仓库行为与之前完全一致。
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


def main():
    _start_heartbeat()
    try:
        plan = json.loads(PLAN.read_text())
        os.chmod(BASE / 'mlbridge_rclone.conf', 0o600)
        os.chmod(BASE / 'mlbridge_restic_password', 0o600)
        env = dict(os.environ)
        env['RCLONE_CONFIG'] = str(BASE / 'mlbridge_rclone.conf')
        env['RESTIC_PASSWORD_FILE'] = str(BASE / 'mlbridge_restic_password')
        if (BASE / GIT_AUTH).exists():
            os.chmod(BASE / GIT_AUTH, 0o600)
        repo = plan['restic_repository']
        root = BASE / 'mlbridge-workspace'
        status('PREPARING', job_id=plan['job_id'], session=plan['session'])
        # Clone first so the short-lived GitHub token is removed before any
        # potentially long restic/rclone installation or research setup.
        if not shutil.which('git'):
            run(['apt-get', 'update', '-qq'])
            run(['apt-get', 'install', '-y', '-qq', 'git'])
        # GitHub must contain the pinned commit. Private repositories are
        # cloned with a short-lived GIT_ASKPASS credential file (see
        # clone_repository); public repositories behave exactly as before.
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
        # 先把最终状态留在 VM 上供本地 watch 读取；只有 watch 确认读取后
        # 才可按 auto_stop 选择释放实例，不能在这里主动 unassign。
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
        _remove_git_auth()


if __name__ == '__main__':
    main()
