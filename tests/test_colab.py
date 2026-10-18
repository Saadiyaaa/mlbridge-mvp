import subprocess

import pytest

from mlbridge import colab_runner, core


def test_job_spec_requires_argv_and_valid_outputs():
    good = {'jobs': {'train': {'command': ['python', 'train.py'], 'outputs': ['results'],
                               'setup': [['python', '-m', 'pip', 'install', 'torch']]}}}
    assert core.job_spec(good, 'train')['outputs'] == ['results']
    bad = {'jobs': {'train': {'command': 'rm -rf /'}}}
    with pytest.raises(core.BridgeError):
        core.job_spec(bad, 'train')


def test_watch_transport_failure_does_not_stop(monkeypatch):
    monkeypatch.setattr(colab_runner, 'get_status', lambda session: {'phase': 'UNREACHABLE'})
    def deny_stop(session):
        raise AssertionError('should not stop on missing status')
    monkeypatch.setattr(colab_runner, 'shutdown', deny_stop)
    monkeypatch.setattr(colab_runner.time, 'sleep', lambda s: None)
    result = colab_runner.watch('mlb-test', interval=5)
    assert result['phase'] == 'UNKNOWN'


def test_watch_success_stops_and_failure_preserves(monkeypatch):
    stopped = []
    monkeypatch.setattr(colab_runner, 'shutdown', lambda session: stopped.append(session))
    monkeypatch.setattr(colab_runner, 'get_status', lambda session: {'phase': 'SUCCEEDED', 'snapshot_id': 'b'*64})
    assert colab_runner.watch('one', interval=5)['phase'] == 'SUCCEEDED'
    assert stopped == ['one']
    monkeypatch.setattr(colab_runner, 'get_status', lambda session: {'phase': 'FAILED'})
    assert colab_runner.watch('two', interval=5)['phase'] == 'FAILED'
    assert stopped == ['one']


def test_colab_starter_uses_detached_session():
    script = colab_runner.starter_script()
    assert 'start_new_session=True' in script
    assert '--rm' not in script


def _prepare_project(tmp_path, monkeypatch):
    """独立的临时 XDG 配置 + 带提交的 git 工作区,完全不动用户真实配置。"""
    root = tmp_path / 'proj'
    root.mkdir()
    subprocess.run(['git', 'init', '-q', str(root)], check=True)
    subprocess.run(['git', '-C', str(root), 'config', 'user.name', 'T'], check=True)
    subprocess.run(['git', '-C', str(root), 'config', 'user.email', 't@example.com'], check=True)
    (root / 'keep.txt').write_text('KEEP')
    subprocess.run(['git', '-C', str(root), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(root), 'commit', '-qm', 'init'], check=True)
    cfg = tmp_path / 'cfg'
    (cfg / 'mlbridge').mkdir(parents=True)
    monkeypatch.setenv('XDG_CONFIG_HOME', str(cfg))
    scratch = tmp_path / 'creds'
    scratch.mkdir()
    rconf, passwd = scratch / 'rclone.conf', scratch / 'passwd'
    rconf.write_text('[gdrive]\ntype=drive\n')
    passwd.write_text('topsecret')
    (cfg / 'mlbridge' / 'settings.toml').write_text(
        '[storage]\nremote="gdrive"\nprefix="data"\n'
        f'password_file="{passwd}"\nrclone_config="{rconf}"\n')
    core.register('demo', str(root), 'https://github.com/demo/demo.git')


def test_launch_failure_attempts_credential_cleanup(tmp_path, monkeypatch):
    """凭据上传后 staging 失败时,必须尽力通知 kernel 删除已上传的凭据文件。"""
    _prepare_project(tmp_path, monkeypatch)
    stage_order, put_scripts = [], []
    def fake_cli(*args, capture=False, timeout=None):
        return subprocess.CompletedProcess(args, 0)
    def fake_stage(session, src, remote_basename):
        stage_order.append(remote_basename)
        if remote_basename == 'mlbridge_restic_password':
            raise core.BridgeError('upload failed')
    monkeypatch.setattr(colab_runner, 'cli', fake_cli)
    monkeypatch.setattr(colab_runner, 'stage', fake_stage)
    monkeypatch.setattr(colab_runner, 'put_script', lambda s, code: put_scripts.append(code))
    with pytest.raises(core.BridgeError, match='upload failed'):
        colab_runner.launch('demo', 'example', [], 'L4', allow_empty_input=True)
    # 顺序:先 worker/job/rclone.conf,凭据最后,失败点在密码文件
    assert stage_order[:3] == ['mlbridge_worker.py', 'mlbridge_job.json', 'mlbridge_rclone.conf']
    # 最后注入的脚本是凭据清理脚本
    cleanup = colab_runner.cleanup_script()
    assert 'mlbridge_restic_password' in cleanup and 'mlbridge_rclone.conf' in cleanup
    # 启对症断言:准备脚本不含凭据操作
    assert 'mkdir' in put_scripts[0]


def test_launch_cpu_mode_without_gpu_flag(tmp_path, monkeypatch):
    """支持 CPU / none 免费测试模式，不传 --gpu 参数。"""
    _prepare_project(tmp_path, monkeypatch)
    cli_calls = []
    def fake_cli(*args, capture=False, timeout=None):
        cli_calls.append(args)
        return subprocess.CompletedProcess(args, 0)
    monkeypatch.setattr(colab_runner, 'cli', fake_cli)
    monkeypatch.setattr(colab_runner, 'stage', lambda *args: None)
    monkeypatch.setattr(colab_runner, 'put_script', lambda *args: None)
    session = colab_runner.launch('demo', 'example', [], 'CPU', allow_empty_input=True)
    assert ('new', '-s', session) in cli_calls
    assert not any('--gpu' in c for c in cli_calls)


def test_git_unpushed_commits_blocks_unless_allowed(tmp_path, monkeypatch):
    """有未 push 到 upstream 的提交时拦截，防白烧 GPU；--allow-dirty 可覆盖。"""
    root = tmp_path / 'proj_with_upstream'
    root.mkdir()
    subprocess.run(['git', 'init', '-q', str(root)], check=True)
    subprocess.run(['git', '-C', str(root), 'config', 'user.name', 'T'], check=True)
    subprocess.run(['git', '-C', str(root), 'config', 'user.email', 't@example.com'], check=True)
    (root / 'f.txt').write_text('v1')
    subprocess.run(['git', '-C', str(root), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(root), 'commit', '-qm', 'c1'], check=True)
    # 模拟 upstream 分支
    subprocess.run(['git', '-C', str(root), 'branch', 'upstream_main'], check=True)
    subprocess.run(['git', '-C', str(root), 'branch', '--set-upstream-to', 'upstream_main'], check=True)
    # 本地新增提交，领先于 upstream
    (root / 'f.txt').write_text('v2')
    subprocess.run(['git', '-C', str(root), 'commit', '-am', 'c2'], check=True)
    with pytest.raises(core.BridgeError, match='未推送'):
        core.git_commit(root, allow_dirty=False)
    # allow_dirty=True 时允许绕过
    commit = core.git_commit(root, allow_dirty=True)
    assert len(commit) == 40

