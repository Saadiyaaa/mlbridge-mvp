"""Run the *actual remote worker* against fake restic and a real tiny local Git repo.

Does not spend Google API quota or require a Colab account. The fake restic
emulates two builds: modern (0.17.x, supports --overwrite) and legacy
(0.12.x, no --overwrite flag) so the version-adaptive restore logic is
actually exercised.
"""
import json
import os
import subprocess
import sys

from mlbridge import remote_worker

FAKE_RESTIC = r'''#!/bin/sh
# restic 调用形如: restic -r REPO <backup|restore> ...；子命令需在全参数中识别
sub=""
for a in "$@"; do
  case "$a" in backup|restore|version) sub="$a"; break;; esac
done
case "$sub" in
  version)
    echo "restic ${FAKE_RESTIC_VERSION:-0.17.0} compiled with go1.22.5 on linux/amd64"
    ;;
  restore)
    printf '%s\n' "$*" >> "${FAKE_RESTIC_LOG:?FAKE_RESTIC_LOG required}"
    prev=""; target=""
    for a in "$@"; do
      if [ "$prev" = "--target" ]; then target="$a"; fi
      prev="$a"
    done
    printf 'from pathlib import Path\np=Path("outputs")\np.mkdir(exist_ok=True)\n(p/"img.png").write_bytes(b"SNAPSHOT")\n' > "$target/run.py"
    ;;
  backup)
    printf '%s\n' "$*" >> "${FAKE_RESTIC_LOG:?FAKE_RESTIC_LOG required}"
    printf '{"message_type":"summary","snapshot_id":"%064d"}\n' 0
    ;;
esac
exit 0
'''


def _run_worker(tmp_path, monkeypatch, fake_version):
    repo = tmp_path / 'existing-git-project'
    repo.mkdir()
    subprocess.run(['git', 'init', '-q', str(repo)], check=True)
    subprocess.run(['git', 'config', 'user.name', 'Test'], cwd=repo, check=True)
    subprocess.run(['git', 'config', 'user.email', 'test@example.com'], cwd=repo, check=True)
    # tracked 的 run.py 在 git 版本里写 b'GIT'；fake restore 会用"快照版"覆盖它
    (repo / 'run.py').write_text(
        "from pathlib import Path\np=Path('outputs')\np.mkdir(exist_ok=True)\n"
        "(p/'img.png').write_bytes(b'GIT')\n")
    subprocess.run(['git', 'add', '.'], cwd=repo, check=True)
    subprocess.run(['git', 'commit', '-qm', 'initial'], cwd=repo, check=True)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                     cwd=repo, text=True).strip()
    base = tmp_path / 'remote-root'
    base.mkdir()
    (base / 'mlbridge_rclone.conf').write_text('[gdrive]\ntype=drive\n')
    (base / 'mlbridge_restic_password').write_text('test')
    plan = {'job_id': 'test', 'session': 'test', 'project': 'demo',
            'repository': str(repo), 'commit': commit, 'restic_repository': 'rclone:gdrive:store/demo',
            'inputs': ['0'*64], 'setup': [], 'command': [sys.executable, 'run.py'],
            'outputs': ['outputs']}
    (base / 'mlbridge_job.json').write_text(json.dumps(plan))
    fakebin = tmp_path / 'fakebin'
    fakebin.mkdir()
    restic_log = tmp_path / 'restic.log'
    restic_log.write_text('')
    (fakebin / 'restic').write_text(FAKE_RESTIC)
    (fakebin / 'restic').chmod(0o755)
    (fakebin / 'rclone').write_text('#!/bin/sh\nexit 0\n')
    (fakebin / 'rclone').chmod(0o755)
    monkeypatch.setenv('PATH', str(fakebin) + os.pathsep + os.environ['PATH'])
    monkeypatch.setenv('FAKE_RESTIC_VERSION', fake_version)
    monkeypatch.setenv('FAKE_RESTIC_LOG', str(restic_log))
    monkeypatch.setattr(remote_worker, 'BASE', base)
    monkeypatch.setattr(remote_worker, 'STATE', base / 'mlbridge_state.json')
    monkeypatch.setattr(remote_worker, 'PLAN', base / 'mlbridge_job.json')
    remote_worker.main()
    state = json.loads((base / 'mlbridge_state.json').read_text())
    return {'state': state, 'base': base, 'ws': base / 'mlbridge-workspace',
            'restore_log': [line for line in restic_log.read_text().splitlines()
                            if ' restore ' in line]}


def test_worker_snapshot_wins_over_tracked_files(tmp_path, monkeypatch):
    """restic ≥0.16.2:恢复输入必须显式 --overwrite always,快照覆盖 tracked 文件。"""
    result = _run_worker(tmp_path, monkeypatch, '0.17.0')
    assert result['state']['phase'] == 'SUCCEEDED'
    assert len(result['state']['snapshot_id']) == 64
    # 快照版的 run.py 已执行:输出是 SNAPSHOT,而不是 git checkout 的 GIT
    assert (result['ws'] / 'outputs' / 'img.png').read_bytes() == b'SNAPSHOT'
    assert result['restore_log']
    assert all('--overwrite always' in line for line in result['restore_log'])
    assert not any('never' in line for line in result['restore_log'])
    # 凭据清理不变
    assert not (result['base'] / 'mlbridge_rclone.conf').exists()
    assert not (result['base'] / 'mlbridge_restic_password').exists()


def test_worker_legacy_restic_without_overwrite_flag(tmp_path, monkeypatch):
    """restic 0.12.x(如 Ubuntu 22.04/Colab apt):不传 --overwrite 也能完成任务。

    旧实现无条件传 --overwrite never,在旧版 restic 上是 unknown-flag 直接失败;
    现在按版本自适应,不再传参,语义(无条件覆盖)与旧版缺省行为一致。
    """
    result = _run_worker(tmp_path, monkeypatch, '0.12.1')
    assert result['state']['phase'] == 'SUCCEEDED'
    assert (result['ws'] / 'outputs' / 'img.png').read_bytes() == b'SNAPSHOT'
    assert result['restore_log']
    assert all('--overwrite' not in line for line in result['restore_log'])


def test_worker_cleans_dirty_workspace_before_clone(tmp_path, monkeypatch):
    """如果之前任务留下了未清空的 workspace 目录，重新运行时自动安全重建，不报 git clone 冲突。"""
    repo = tmp_path / 'existing-git-project'
    repo.mkdir()
    subprocess.run(['git', 'init', '-q', str(repo)], check=True)
    subprocess.run(['git', 'config', 'user.name', 'Test'], cwd=repo, check=True)
    subprocess.run(['git', 'config', 'user.email', 'test@example.com'], cwd=repo, check=True)
    (repo / 'run.py').write_text(
        "from pathlib import Path\np=Path('outputs')\np.mkdir(exist_ok=True)\n"
        "(p/'img.png').write_bytes(b'OK')\n")
    subprocess.run(['git', 'add', '.'], cwd=repo, check=True)
    subprocess.run(['git', 'commit', '-qm', 'initial'], cwd=repo, check=True)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    base = tmp_path / 'remote-root'
    base.mkdir()
    # 模拟遗留的脏目录
    dirty_ws = base / 'mlbridge-workspace'
    dirty_ws.mkdir()
    (dirty_ws / 'dirty_file.txt').write_text('stale residue')
    (base / 'mlbridge_rclone.conf').write_text('[gdrive]\ntype=drive\n')
    (base / 'mlbridge_restic_password').write_text('test')
    plan = {'job_id': 'test2', 'session': 'test2', 'project': 'demo',
            'repository': str(repo), 'commit': commit, 'restic_repository': 'rclone:gdrive:store/demo',
            'inputs': ['0'*64], 'setup': [], 'command': [sys.executable, 'run.py'],
            'outputs': ['outputs']}
    (base / 'mlbridge_job.json').write_text(json.dumps(plan))
    fakebin = tmp_path / 'fakebin'
    fakebin.mkdir()
    restic_log = tmp_path / 'restic.log'
    restic_log.write_text('')
    (fakebin / 'restic').write_text(FAKE_RESTIC)
    (fakebin / 'restic').chmod(0o755)
    (fakebin / 'rclone').write_text('#!/bin/sh\nexit 0\n')
    (fakebin / 'rclone').chmod(0o755)
    monkeypatch.setenv('PATH', str(fakebin) + os.pathsep + os.environ['PATH'])
    monkeypatch.setenv('FAKE_RESTIC_VERSION', '0.17.0')
    monkeypatch.setenv('FAKE_RESTIC_LOG', str(restic_log))
    monkeypatch.setattr(remote_worker, 'BASE', base)
    monkeypatch.setattr(remote_worker, 'STATE', base / 'mlbridge_state.json')
    monkeypatch.setattr(remote_worker, 'PLAN', base / 'mlbridge_job.json')
    remote_worker.main()
    state = json.loads((base / 'mlbridge_state.json').read_text())
    assert state['phase'] == 'SUCCEEDED'
    assert not (dirty_ws / 'dirty_file.txt').exists()

