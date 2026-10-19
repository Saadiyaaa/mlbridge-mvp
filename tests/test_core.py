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


def test_publish_materializes_explicit_symlink_by_default(tmp_path, monkeypatch):
    root = prepared(tmp_path, monkeypatch)
    external = tmp_path / 'external-weights'
    external.mkdir()
    (external / 'weights.bin').write_bytes(b'weights')
    (root / 'models').mkdir()
    (root / 'models' / 'model').symlink_to(external, target_is_directory=True)
    core.register('demo', str(root), 'https://github.com/demo/demo.git')
    stage_paths = []
    def fake_restic(name, store, args, *, cwd=None, capture=False):
        stage_paths.append(cwd)
        assert args[-1] == 'models/model'
        assert (cwd / 'models/model/weights.bin').read_bytes() == b'weights'
        assert not (cwd / 'models/model').is_symlink()
        return subprocess.CompletedProcess(args, 0, stdout=json.dumps({
            'message_type': 'summary', 'snapshot_id': 'c' * 64}))
    monkeypatch.setattr(core, 'restic', fake_restic)
    assert core.publish('demo', ['models/model'], 'model') == 'c' * 64
    assert len(stage_paths) == 1 and not stage_paths[0].exists()
    assert (root / 'models/model').is_symlink()
    assert (external / 'weights.bin').read_bytes() == b'weights'
    with pytest.raises(core.BridgeError, match='软链接越界'):
        core.publish('demo', ['models/model'], 'model', materialize_links=False)


def test_publish_rejects_symlink_ancestor_even_with_in_root_target(tmp_path, monkeypatch):
    root = prepared(tmp_path, monkeypatch)
    (root / 'alias').symlink_to('data', target_is_directory=True)
    core.register('demo', str(root), 'https://github.com/demo/demo.git')
    with pytest.raises(core.BridgeError, match='软链接祖先'):
        core.publish('demo', ['alias/a.png'], 'ancestor')
    assert (root / 'alias').is_symlink()


def test_publish_parent_with_nested_link_rejected_instead_of_dangling_snapshot(
        tmp_path, monkeypatch):
    root = prepared(tmp_path, monkeypatch)
    external = tmp_path / 'external-model'
    external.mkdir()
    (external / 'weights').write_text('model')
    (root / 'models').mkdir()
    (root / 'models/model').symlink_to(external, target_is_directory=True)
    core.register('demo', str(root), 'https://github.com/demo/demo.git')
    with pytest.raises(core.BridgeError, match='请单独 --path'):
        core.publish('demo', ['models'], 'models')
    # The explicit symlink path is materialized without touching the parent.
    calls = []
    def fake_restic(name, store, args, *, cwd=None, capture=False):
        calls.append((cwd / 'models/model/weights').read_text())
        return subprocess.CompletedProcess(args, 0, stdout=json.dumps({
            'message_type': 'summary', 'snapshot_id': 'e' * 64}))
    monkeypatch.setattr(core, 'restic', fake_restic)
    assert core.publish('demo', ['models/model'], 'model') == 'e' * 64
    assert calls == ['model']


def test_publish_mixed_paths_one_staged_snapshot(tmp_path, monkeypatch):
    root = prepared(tmp_path, monkeypatch)
    external = tmp_path / 'outside.bin'
    external.write_bytes(b'actual')
    (root / 'weights.bin').symlink_to(external)
    core.register('demo', str(root), 'https://github.com/demo/demo.git')
    calls = []
    def fake_restic(name, store, args, *, cwd=None, capture=False):
        calls.append((cwd, args))
        assert (cwd / 'weights.bin').read_bytes() == b'actual'
        assert (cwd / 'data/a.png').read_bytes() == b'fake image bytes'
        return subprocess.CompletedProcess(args, 0, stdout=json.dumps({
            'message_type': 'summary', 'snapshot_id': 'd' * 64}))
    monkeypatch.setattr(core, 'restic', fake_restic)
    assert core.publish('demo', ['weights.bin', 'data'], 'mix') == 'd' * 64
    assert len(calls) == 1 and calls[0][1][-2:] == ['weights.bin', 'data']
    assert not calls[0][0].exists()


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
    with pytest.raises(core.BridgeError, match=r'0\.17\.0'):
        core.fetch('demo', 'a' * 64)


def test_restic_0164_does_not_support_overwrite(tmp_path, monkeypatch):
    root = prepared(tmp_path, monkeypatch)
    core.register('demo', str(root), 'https://github.com/demo/demo.git')
    calls = []
    def fake(cmd, **kwargs):
        calls.append(cmd)
        if cmd[:2] == ['restic', 'version']:
            return subprocess.CompletedProcess(cmd, 0,
                stdout='restic 0.16.4 compiled with go1.22.2 on linux/amd64')
        return subprocess.CompletedProcess(cmd, 0)
    monkeypatch.setattr(core, 'proc', fake)
    with pytest.raises(core.BridgeError, match=r'0\.17\.0'):
        core.fetch('demo', 'a' * 64)
    core.fetch('demo', 'a' * 64, overwrite=True)
    assert '--overwrite' not in calls[-1]


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


def test_default_gpu_from_global_settings(tmp_path, monkeypatch):
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path))
    path = core.config_dir() / 'settings.toml'
    path.parent.mkdir()
    path.write_text('[storage]\nremote="drive"\n')
    assert core.default_gpu() == 'L4'  # backward compatible without [colab]
    path.write_text('[colab]\ndefault_gpu="t4"\n')
    assert core.default_gpu() == 'T4'
    path.write_text('[colab]\ndefault_gpu="cpu"\n')
    assert core.default_gpu() == 'CPU'


@pytest.mark.parametrize('raw', [
    '[colab]\ndefault_gpu="TPU"\n',
    '[colab]\ndefault_gpu=42\n',
    'colab="CPU"\n',
])
def test_invalid_default_gpu_fails_closed(tmp_path, monkeypatch, raw):
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path))
    path = core.config_dir() / 'settings.toml'
    path.parent.mkdir()
    path.write_text(raw)
    with pytest.raises(core.BridgeError, match='colab'):
        core.default_gpu()


def test_default_high_mem_from_global_settings(tmp_path, monkeypatch):
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path))
    path = core.config_dir() / 'settings.toml'
    path.parent.mkdir()
    path.write_text('[storage]\nremote="drive"\n')
    assert core.default_high_mem() is False  # 未配置时使用普通内存机型
    path.write_text('[colab]\nhigh_mem=true\n')
    assert core.default_high_mem() is True
    path.write_text('[colab]\nhigh_mem="true"\n')
    with pytest.raises(core.BridgeError, match='colab.high_mem'):
        core.default_high_mem()


def test_new_settings_includes_colab_default(tmp_path, monkeypatch):
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path))
    path = core.new_settings()
    assert '[colab]' in path.read_text()
    assert core.default_gpu() == 'L4'
    assert core.default_high_mem() is False


# ---- XDG 空值回退 ----

def test_app_dir_empty_or_missing_xdg_falls_back_to_home(tmp_path, monkeypatch):
    home = tmp_path / 'home'
    home.mkdir()
    monkeypatch.setenv('HOME', str(home))
    monkeypatch.setenv('XDG_CONFIG_HOME', '')
    monkeypatch.delenv('XDG_STATE_HOME', raising=False)
    assert core.config_dir() == home / '.config' / 'mlbridge'
    assert core.state_dir() == home / '.local' / 'state' / 'mlbridge'
    # 正常设置时仍优先 XDG。
    monkeypatch.setenv('XDG_STATE_HOME', str(tmp_path / 'state'))
    assert core.state_dir() == tmp_path / 'state' / 'mlbridge'


# ---- git preflight fail-closed ----

def _git_root(tmp_path):
    root = tmp_path / 'proj'
    root.mkdir()
    return root


def _ok(cmd, stdout=''):
    return subprocess.CompletedProcess(cmd, 0, stdout=stdout)


def _happy_git(cmd, *, upstream='origin/main', remote='origin', url=None,
              unpushed=''):
    if cmd[1:3] == ['status', '--porcelain']:
        return _ok(cmd)
    if cmd[2:4] == ['--abbrev-ref', '@{u}']:
        return _ok(cmd, stdout=upstream + '\n')
    if cmd[1] == 'remote' and len(cmd) == 2:
        return _ok(cmd, stdout=remote + '\n')
    if cmd[1:3] == ['remote', 'get-url']:
        return _ok(cmd, stdout=url + '\n')
    if cmd[1] == 'rev-list':
        return _ok(cmd, stdout=unpushed)
    if cmd[1:3] == ['rev-parse', 'HEAD']:
        return _ok(cmd, stdout='a' * 40 + '\n')
    raise AssertionError(f'意外的（可能是联网的）git 命令: {cmd}')


def test_git_preflight_fails_closed_without_upstream(tmp_path, monkeypatch):
    root = _git_root(tmp_path)
    calls = []
    def fake(cmd, **kwargs):
        calls.append(cmd)
        if cmd[1:3] == ['status', '--porcelain']:
            return _ok(cmd)
        if cmd[2:4] == ['--abbrev-ref', '@{u}']:
            raise core.BridgeError('命令失败(128): git fatal: no upstream')
        raise AssertionError(f'意外的 git 命令: {cmd}')
    monkeypatch.setattr(core, 'proc', fake)
    with pytest.raises(core.BridgeError, match='上游跟踪分支'):
        core.git_commit(root, False)
    # 不允许用 fetch/ls-remote 等联网命令兜底。
    assert all(cmd[1:2] != ['fetch'] and cmd[1:2] != ['ls-remote'] for cmd in calls)


def test_git_preflight_fails_closed_on_rev_list_error(tmp_path, monkeypatch):
    root = _git_root(tmp_path)
    def fake(cmd, **kwargs):
        if cmd[1] == 'rev-list':
            raise core.BridgeError('命令失败(128): git fatal: bad revision')
        return _happy_git(cmd)
    monkeypatch.setattr(core, 'proc', fake)
    with pytest.raises(core.BridgeError, match='rev-list'):
        core.git_commit(root, False)


def test_git_preflight_reports_unpushed_commits(tmp_path, monkeypatch):
    root = _git_root(tmp_path)
    def fake(cmd, **kwargs):
        return _happy_git(cmd, unpushed='b' * 40 + '\n' + 'c' * 40 + '\n')
    monkeypatch.setattr(core, 'proc', fake)
    with pytest.raises(core.BridgeError, match='未推送'):
        core.check_git_pushed(root)


def test_git_commit_allow_dirty_bypasses_preflight(tmp_path, monkeypatch):
    root = _git_root(tmp_path)
    calls = []
    def fake(cmd, **kwargs):
        calls.append(cmd)
        if cmd[1:3] == ['status', '--porcelain']:
            return _ok(cmd, stdout=' M tracked.py\n')
        if cmd[1:3] == ['rev-parse', 'HEAD']:
            return _ok(cmd, stdout='a' * 40 + '\n')
        raise AssertionError(cmd)
    monkeypatch.setattr(core, 'proc', fake)
    assert core.git_commit(root, True) == 'a' * 40
    assert calls == [['git', 'rev-parse', 'HEAD']]


def test_git_preflight_rejects_repository_mismatch(tmp_path, monkeypatch):
    root = _git_root(tmp_path)
    def fake(cmd, **kwargs):
        return _happy_git(cmd, url='git@github.com:other/other.git')
    monkeypatch.setattr(core, 'proc', fake)
    with pytest.raises(core.BridgeError, match='repository'):
        core.git_commit(root, False, repository='https://github.com/demo/demo.git')


def test_git_preflight_accepts_matching_repository_ignoring_git_suffix(tmp_path,
                                                                       monkeypatch):
    root = _git_root(tmp_path)
    def fake(cmd, **kwargs):
        return _happy_git(cmd, url='https://github.com/demo/demo')
    monkeypatch.setattr(core, 'proc', fake)
    assert core.git_commit(
        root, False, repository='https://github.com/demo/demo.git') == 'a' * 40


def test_git_preflight_fails_when_repository_cannot_be_proven(tmp_path, monkeypatch):
    """上游指向未配置的 remote：无法证明配置仓库的推送状态，必须拒绝。"""
    root = _git_root(tmp_path)
    def fake(cmd, **kwargs):
        return _happy_git(cmd, upstream='fork/main', remote='origin')
    monkeypatch.setattr(core, 'proc', fake)
    with pytest.raises(core.BridgeError, match='无法证明'):
        core.git_commit(root, False, repository='https://github.com/demo/demo.git')


def test_git_preflight_without_repository_checks_upstream(tmp_path, monkeypatch):
    """不传 repository 时只检查当前分支的上游。"""
    root = _git_root(tmp_path)
    def fake(cmd, **kwargs):
        return _happy_git(cmd)
    monkeypatch.setattr(core, 'proc', fake)
    assert core.git_commit(root, False) == 'a' * 40
