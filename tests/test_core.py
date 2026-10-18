import json
import subprocess

import pytest

from mlbridge import core


@pytest.fixture(autouse=True)
def _fresh_version_cache():
    core._OVERWRITE_OK.clear()
    yield
    core._OVERWRITE_OK.clear()


def prepared(tmp_path, monkeypatch):
    root = tmp_path / 'existing-project'
    root.mkdir()
    (root / 'keep.txt').write_text('NEVER MODIFY')
    (root / 'data').mkdir()
    (root / 'data' / 'a.png').write_bytes(b'fake image bytes')
    config = tmp_path / 'cfg'
    (config / 'mlbridge').mkdir(parents=True)
    monkeypatch.setenv('XDG_CONFIG_HOME', str(config))
    rclone = tmp_path / 'rclone.conf'
    passwd = tmp_path / 'password'
    rclone.write_text('[drive]\ntype=drive\n')
    passwd.write_text('fake test credential')
    (config / 'mlbridge' / 'settings.toml').write_text('[storage]\nremote="drive"\nprefix="data"\n'
       f'password_file="{passwd}"\nrclone_config="{rclone}"\n')
    return root


def test_register_is_external_and_safe(tmp_path, monkeypatch):
    root = prepared(tmp_path, monkeypatch)
    before = sorted(str(p.relative_to(root)) for p in root.rglob('*'))
    path = core.register('demo', str(root), 'https://github.com/demo/demo.git')
    assert path.exists() and not path.is_relative_to(root)
    assert core.project('demo')['workspace'] == str(root)
    assert before == sorted(str(p.relative_to(root)) for p in root.rglob('*'))


def test_path_validation_blocks_traversal_and_symlinks(tmp_path):
    (tmp_path / 'safe').mkdir()
    assert core.validate_paths(tmp_path, ['safe']) == ['safe']
    for path in ['.', '../escape', '/etc/passwd', 'safe/../../bad', '.git']:
        with pytest.raises(core.BridgeError):
            core.validate_paths(tmp_path, [path], exist=False)
    (tmp_path / 'escape').symlink_to('/etc')
    with pytest.raises(core.BridgeError):
        core.validate_paths(tmp_path, ['escape'], exist=False)


def test_publish_is_relative_and_parses_summary(tmp_path, monkeypatch):
    root = prepared(tmp_path, monkeypatch)
    core.register('demo', str(root), 'https://github.com/demo/demo.git')
    calls = []
    def fake(cmd, *, cwd=None, env=None, capture=False, timeout=None):
        calls.append((cmd, cwd, env))
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps({'message_type': 'summary', 'snapshot_id': 'a' * 64}))
    monkeypatch.setattr(core, 'proc', fake)
    snapshot = core.publish('demo', ['data'], 'upstream_001')
    assert snapshot == 'a' * 64
    cmd, cwd, env = calls[0]
    assert cmd[:3] == ['restic', '-r', 'rclone:drive:data/demo']
    assert cmd[-1] == 'data'
    assert str(cwd) == str(root)
    assert 'RESTIC_PASSWORD_FILE' in env
    assert (root / 'keep.txt').read_text() == 'NEVER MODIFY'


def test_fetch_refuses_latest_and_defaults_never_overwrite(tmp_path, monkeypatch):
    root = prepared(tmp_path, monkeypatch)
    core.register('demo', str(root), 'https://github.com/demo/demo.git')
    with pytest.raises(core.BridgeError):
        core.fetch('demo', 'latest')
    calls = []
    def fake(cmd, **kwargs):
        calls.append(cmd)
        if cmd[:2] == ['restic', 'version']:
            return subprocess.CompletedProcess(cmd, 0,
                stdout='restic 0.17.3 compiled with go1.22.5 on linux/amd64')
        return subprocess.CompletedProcess(cmd, 0)
    monkeypatch.setattr(core, 'proc', fake)
    core.fetch('demo', 'a' * 64)
    assert calls[-1][-2:] == ['--overwrite', 'never']


def test_fetch_error_without_explicit_overwrite_on_old_restic(tmp_path, monkeypatch):
    """发行版 restic 常见为 0.12.x：不带 --overwrite 时拒绝可能覆盖的恢复。"""
    root = prepared(tmp_path, monkeypatch)
    core.register('demo', str(root), 'https://github.com/demo/demo.git')
    def fake(cmd, **kwargs):
        if cmd[:2] == ['restic', 'version']:
            return subprocess.CompletedProcess(cmd, 0,
                stdout='restic 0.12.1 compiled with go1.13.8 on linux/amd64')
        return subprocess.CompletedProcess(cmd, 0)
    monkeypatch.setattr(core, 'proc', fake)
    with pytest.raises(core.BridgeError, match=r'0\.16\.2'):
        core.fetch('demo', 'a' * 64)


def test_fetch_old_restic_explicit_overwrite_omits_flag(tmp_path, monkeypatch):
    """老版 restic 显式请求覆盖时成功，且不再传不存在的 --overwrite 参数。"""
    root = prepared(tmp_path, monkeypatch)
    core.register('demo', str(root), 'https://github.com/demo/demo.git')
    calls = []
    def fake(cmd, **kwargs):
        calls.append(cmd)
        if cmd[:2] == ['restic', 'version']:
            return subprocess.CompletedProcess(cmd, 0,
                stdout='restic 0.12.1 compiled with go1.13.8 on linux/amd64')
        return subprocess.CompletedProcess(cmd, 0)
    monkeypatch.setattr(core, 'proc', fake)
    core.fetch('demo', 'a' * 64, overwrite=True)
    assert '--overwrite' not in calls[-1]
    assert calls[-1][1:4] == ['-r', 'rclone:drive:data/demo', 'restore']


def test_job_spec_allows_jobs_without_outputs():
    """纯 setup/外部副作用任务合法；remote_worker 支持空 outputs。"""
    cfg = {'jobs': {'onlysetup': {'command': ['python', 'train.py'],
                                  'setup': [], 'outputs': []}}}
    spec = core.job_spec(cfg, 'onlysetup')
    assert spec['outputs'] == []
    bad = {'jobs': {'j': {'command': ['python', 'x.py'], 'outputs': 'outputs'}}}
    with pytest.raises(core.BridgeError):
        core.job_spec(bad, 'j')


def test_json_summary_required():
    with pytest.raises(core.BridgeError):
        core.parse_backup_output('{"message_type":"status"}')
