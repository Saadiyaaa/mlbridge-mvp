import subprocess
from pathlib import Path

import pytest

from mlbridge import colab_runner, core


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    monkeypatch.setenv('XDG_STATE_HOME', str(tmp_path / 'state'))
    monkeypatch.delenv('MLBRIDGE_GITHUB_USERNAME', raising=False)
    monkeypatch.delenv('MLBRIDGE_GITHUB_TOKEN', raising=False)
    monkeypatch.setattr(colab_runner, '_github_secrets_path',
                        lambda: tmp_path / 'isolated-secrets.env')


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


def test_watch_stop_failure_keeps_success_and_retries(tmp_path, monkeypatch):
    colab_runner.store_local('job', {'stopped': False})
    monkeypatch.setattr(colab_runner, 'get_status',
                        lambda session: {'phase': 'SUCCEEDED', 'snapshot_id': 'a' * 64})
    attempts = []
    def fail_stop(session):
        attempts.append(session)
        raise core.BridgeError('network timeout')
    monkeypatch.setattr(colab_runner, 'shutdown', fail_stop)
    first = colab_runner.watch('job', interval=5)
    assert first['phase'] == 'SUCCEEDED' and first['instance_stopped'] is False
    assert first['stop_error'] == 'network timeout'
    assert colab_runner.get_local('job')['result']['phase'] == 'SUCCEEDED'
    monkeypatch.setattr(colab_runner, 'get_status',
                        lambda session: (_ for _ in ()).throw(AssertionError('must use cached result')))
    second = colab_runner.watch('job', interval=5)
    assert second['phase'] == 'SUCCEEDED' and len(attempts) == 2
    def success_stop(session):
        record = colab_runner.get_local(session)
        record['stopped'] = True
        colab_runner.store_local(session, record)
    monkeypatch.setattr(colab_runner, 'shutdown', success_stop)
    assert colab_runner.watch('job', interval=5)['instance_stopped'] is True
    assert colab_runner.get_local('job')['stopped'] is True


def test_watch_no_stop_and_cached_failed_result(monkeypatch):
    colab_runner.store_local('success', {'stopped': False,
                             'result': {'phase': 'SUCCEEDED', 'snapshot_id': 'a' * 64}})
    colab_runner.store_local('failed', {'stopped': False,
                             'result': {'phase': 'FAILED', 'error': 'job failed'}})
    monkeypatch.setattr(colab_runner, 'get_status',
                        lambda session: (_ for _ in ()).throw(AssertionError('must use cached result')))
    monkeypatch.setattr(colab_runner, 'shutdown',
                        lambda session: (_ for _ in ()).throw(AssertionError('must not stop')))
    assert colab_runner.watch('success', interval=5, auto_stop=False)['instance_stopped'] is False
    assert colab_runner.watch('failed', interval=5)['error'] == 'job failed'


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
    repository = 'https://github.com/demo/demo.git'
    branch = subprocess.check_output(['git', '-C', str(root), 'branch', '--show-current'],
                                     text=True).strip()
    commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                                     text=True).strip()
    subprocess.run(['git', '-C', str(root), 'remote', 'add', 'origin', repository], check=True)
    # Simulate a pushed tracking ref without making any network request.
    subprocess.run(['git', '-C', str(root), 'update-ref',
                    f'refs/remotes/origin/{branch}', commit], check=True)
    subprocess.run(['git', '-C', str(root), 'branch', '--set-upstream-to',
                    f'origin/{branch}'], check=True, capture_output=True)
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
    core.register('demo', str(root), repository)


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


@pytest.mark.parametrize('configured, explicit, expected', [
    ('t4', None, 'T4'),
    ('t4', 'A100', 'A100'),
    ('TPU', 'T4', 'T4'),  # explicit CLI value overrides even an invalid configured default
    ('cpu', None, None),
    ('none', None, None),
])
def test_launch_configured_gpu_and_explicit_override(tmp_path, monkeypatch,
                                                      configured, explicit, expected):
    _prepare_project(tmp_path, monkeypatch)
    with (core.config_dir() / 'settings.toml').open('a') as f:
        f.write(f'\n[colab]\ndefault_gpu="{configured}"\n')
    calls = []
    monkeypatch.setattr(colab_runner, 'cli',
                        lambda *args, **kwargs: calls.append(args))
    monkeypatch.setattr(colab_runner, 'stage', lambda *args: None)
    monkeypatch.setattr(colab_runner, 'put_script', lambda *args: None)
    session = colab_runner.launch('demo', 'example', [], explicit,
                                  allow_empty_input=True)
    new = next(args for args in calls if args[0] == 'new')
    assert new == (('new', '-s', session, '--gpu', expected) if expected
                   else ('new', '-s', session))


@pytest.mark.parametrize('configured, explicit, gpu, expected', [
    (None, True, 'A100', ('--gpu', 'A100', '--high-mem')),
    (True, None, 'A100', ('--gpu', 'A100', '--high-mem')),
    (True, False, 'A100', ('--gpu', 'A100')),
    (None, True, 'CPU', ('--high-mem',)),
])
def test_launch_high_mem_selection(tmp_path, monkeypatch, configured, explicit,
                                   gpu, expected):
    _prepare_project(tmp_path, monkeypatch)
    if configured is not None:
        with (core.config_dir() / 'settings.toml').open('a') as f:
            f.write(f'\n[colab]\nhigh_mem={str(configured).lower()}\n')
    calls = []
    monkeypatch.setattr(colab_runner, 'cli', lambda *args, **kwargs: calls.append(args))
    monkeypatch.setattr(colab_runner, 'stage', lambda *args: None)
    monkeypatch.setattr(colab_runner, 'put_script', lambda *args: None)
    session = colab_runner.launch('demo', 'example', [], gpu,
                                  high_mem=explicit, allow_empty_input=True)
    assert next(args for args in calls if args[0] == 'new') == (
        'new', '-s', session, *expected)


@pytest.mark.parametrize('colab_settings, gpu, error', [
    ('high_mem="true"', 'A100', 'colab.high_mem'),
    ('high_mem=true', 'L4', 'L4'),
])
def test_high_mem_preflight_rejects_invalid_request(tmp_path, monkeypatch,
                                                     colab_settings, gpu, error):
    _prepare_project(tmp_path, monkeypatch)
    with (core.config_dir() / 'settings.toml').open('a') as f:
        f.write(f'\n[colab]\n{colab_settings}\n')
    monkeypatch.setattr(colab_runner, 'cli',
                        lambda *args, **kwargs: (_ for _ in ()).throw(
                            AssertionError('must not allocate')))
    with pytest.raises(core.BridgeError, match=error):
        colab_runner.launch('demo', 'example', [], gpu, allow_empty_input=True)


def test_invalid_configured_gpu_rejected_before_allocation(tmp_path, monkeypatch):
    _prepare_project(tmp_path, monkeypatch)
    with (core.config_dir() / 'settings.toml').open('a') as f:
        f.write('\n[colab]\ndefault_gpu="TPU"\n')
    monkeypatch.setattr(colab_runner, 'cli',
                        lambda *args, **kwargs: (_ for _ in ()).throw(
                            AssertionError('must not allocate')))
    with pytest.raises(core.BridgeError, match='colab.default_gpu'):
        colab_runner.launch('demo', 'example', [], allow_empty_input=True)


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



def test_put_script_bounded_timeout_and_retry(monkeypatch):
    """exec 必须带超时；瞬时失败重试同一 session，绝不重新分配。"""
    calls, sleeps = [], []
    monkeypatch.setattr(colab_runner.time, 'sleep', sleeps.append)
    outcomes = iter([core.BridgeError('Connection was lost'), None])

    def fake_cli(*args, capture=False, timeout=None):
        calls.append((args, timeout))
        if next(outcomes) is not None:
            raise core.BridgeError('Connection was lost')
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(colab_runner, 'cli', fake_cli)
    colab_runner.put_script('sess', 'print(1)')
    assert len(calls) == 2
    assert all(args[0] == 'exec' for args, _ in calls)
    assert calls[0][1] == colab_runner.EXEC_TIMEOUT
    assert sleeps == [colab_runner.STAGE_BACKOFF]


def test_cli_retry_gives_up_within_budget_and_stop_stays_idempotent(monkeypatch):
    """重试有上限；stop 失败绝不把本地记录标成已停止。"""
    colab_runner.store_local('sess', {'stopped': False})
    calls = []
    monkeypatch.setattr(colab_runner.time, 'sleep', lambda s: None)

    def fake_cli(*args, capture=False, timeout=None):
        calls.append(args)
        raise core.BridgeError('read timeout')

    monkeypatch.setattr(colab_runner, 'cli', fake_cli)
    with pytest.raises(core.BridgeError, match='不会重复分配实例'):
        colab_runner.shutdown('sess')
    assert len(calls) == colab_runner.STAGE_ATTEMPTS
    assert all(args[0] == 'stop' for args in calls)
    assert colab_runner.get_local('sess')['stopped'] is False


def test_alloc_failure_never_retries_new(tmp_path, monkeypatch):
    """new 失败/超时不重试分配（防复制计费实例），但会尝试幂等 stop 兜底。"""
    _prepare_project(tmp_path, monkeypatch)
    calls = []

    def fake_cli(*args, capture=False, timeout=None):
        calls.append(args)
        if args[0] == 'new':
            raise core.BridgeError('alloc timeout')
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(colab_runner, 'cli', fake_cli)
    monkeypatch.setattr(colab_runner, 'stage', lambda *a: None)
    monkeypatch.setattr(colab_runner, 'put_script', lambda *a: None)
    monkeypatch.setattr(colab_runner.time, 'sleep', lambda s: None)
    with pytest.raises(core.BridgeError, match='alloc timeout'):
        colab_runner.launch('demo', 'example', [], 'L4', allow_empty_input=True)
    assert sum(1 for c in calls if c[0] == 'new') == 1
    assert any(c[0] == 'stop' for c in calls)


def test_staging_failure_before_starter_stops_instance(tmp_path, monkeypatch):
    """starter 之前失败 ⇒ worker 必未启动 ⇒ 自动停止，不留计费实例。"""
    _prepare_project(tmp_path, monkeypatch)
    stops = []

    def fake_stage(session, src, remote_basename):
        if remote_basename == 'mlbridge_restic_password':
            raise core.BridgeError('upload failed')

    monkeypatch.setattr(colab_runner, 'cli',
                        lambda *a, **k: subprocess.CompletedProcess(a, 0))
    monkeypatch.setattr(colab_runner, 'stage', fake_stage)
    monkeypatch.setattr(colab_runner, 'put_script', lambda s, code: None)
    monkeypatch.setattr(colab_runner, 'shutdown', lambda s: stops.append(s))
    monkeypatch.setattr(colab_runner, 'get_status',
                        lambda s: (_ for _ in ()).throw(AssertionError('starter 未发出，不应探测')))
    with pytest.raises(core.BridgeError, match='upload failed'):
        colab_runner.launch('demo', 'example', [], 'L4', allow_empty_input=True)
    assert len(stops) == 1
    record = colab_runner.get_local(stops[0])
    assert record['staging_failed'] is True


def test_staging_failure_after_starter_keeps_running_worker(tmp_path, monkeypatch):
    """starter 阶段失败且 worker 已运行 ⇒ 保留实例交给 watch，绝不误停。"""
    _prepare_project(tmp_path, monkeypatch)
    seen = []

    def fake_starter(session, contents):
        seen.append(session)
        raise core.BridgeError('exec lost')

    monkeypatch.setattr(colab_runner, 'cli',
                        lambda *a, **k: subprocess.CompletedProcess(a, 0))
    monkeypatch.setattr(colab_runner, 'stage', lambda *a: None)
    monkeypatch.setattr(colab_runner, 'exec_starter_once', fake_starter)
    monkeypatch.setattr(colab_runner, 'get_status',
                        lambda s: {'phase': 'RUNNING'})
    monkeypatch.setattr(colab_runner, 'shutdown',
                        lambda s: (_ for _ in ()).throw(
                            AssertionError('must not stop a running worker')))
    with pytest.raises(core.BridgeError, match='exec lost'):
        colab_runner.launch('demo', 'example', [], 'L4', allow_empty_input=True)
    assert len(seen) == 1
    record = colab_runner.get_local(seen[0])
    assert record['staging_failed'] is True and record['stopped'] is False


def _fake_launch_cli(monkeypatch, status):
    """launch 全流程打桩：starter exec 丢失一次回复后抛错，且不重试。

    返回 (exec_bodies, captured_session, stops)。exec_bodies 是每次 exec
    的脚本内容；cleanup 凭据删除脚本含 os.remove 可据此识别。
    """
    exec_bodies, captured, stops = [], [], []

    def fake_cli(*args, capture=False, timeout=None):
        if args[0] == 'exec':
            contents = Path(args[args.index('-f') + 1]).read_text()
            exec_bodies.append(contents)
            if 'start_new_session' in contents:
                raise core.BridgeError('exec lost')
        return subprocess.CompletedProcess(args, 0)

    def fake_get_status(session):
        captured.append(session)
        return status

    def fake_shutdown(session):
        stops.append(session)

    monkeypatch.setattr(colab_runner, 'cli', fake_cli)
    monkeypatch.setattr(colab_runner, 'stage', lambda *a: None)
    monkeypatch.setattr(colab_runner, 'get_status', fake_get_status)
    monkeypatch.setattr(colab_runner, 'shutdown', fake_shutdown)
    return exec_bodies, captured, stops


def test_starter_reply_lost_status_unreachable_no_cleanup_no_stop(tmp_path,
                                                                 monkeypatch):
    """starter 回复丢失 + 状态 UNREACHABLE ⇒ 传输未知 ≠ worker 未启动：
    不清理凭据、不停止实例，只留本地标记与可操作提示。"""
    _prepare_project(tmp_path, monkeypatch)
    exec_bodies, captured, stops = _fake_launch_cli(
        monkeypatch, {'phase': 'UNREACHABLE', 'reason': 'connection lost'})
    with pytest.raises(core.BridgeError, match='exec lost'):
        colab_runner.launch('demo', 'example', [], 'L4', allow_empty_input=True)
    assert len(captured) == 1
    record = colab_runner.get_local(captured[0])
    assert record['staging_failed'] is True
    assert record['staging_probe_phase'] == 'UNREACHABLE'
    assert record['stopped'] is False
    assert stops == []
    assert not any('os.remove' in body for body in exec_bodies)


def test_starter_reply_lost_status_unknown_no_stop(tmp_path, monkeypatch):
    """status 返回 {}（UNKNOWN）⇒ 同样绝不 stop、不清理。"""
    _prepare_project(tmp_path, monkeypatch)
    exec_bodies, captured, stops = _fake_launch_cli(monkeypatch, {})
    with pytest.raises(core.BridgeError, match='exec lost'):
        colab_runner.launch('demo', 'example', [], 'L4', allow_empty_input=True)
    record = colab_runner.get_local(captured[0])
    assert record['staging_failed'] is True
    assert record['staging_probe_phase'] == 'UNKNOWN'
    assert stops == []
    assert not any('os.remove' in body for body in exec_bodies)


def test_starter_exec_fails_exactly_once_no_duplicate_worker(tmp_path,
                                                             monkeypatch):
    """starter exec 失败后绝不重试：只调用一次 colab exec（Popen 不可
    重试，重试会复制 worker）。幂等 prepare 脚本仍走有界重试路径。"""
    _prepare_project(tmp_path, monkeypatch)
    exec_bodies, captured, stops = _fake_launch_cli(
        monkeypatch, {'phase': 'UNREACHABLE', 'reason': 'down'})
    with pytest.raises(core.BridgeError, match='exec lost'):
        colab_runner.launch('demo', 'example', [], 'L4', allow_empty_input=True)
    assert sum('start_new_session' in body for body in exec_bodies) == 1
    assert stops == []


def test_exec_starter_once_never_retries(monkeypatch):
    """exec_starter_once 直接调用 cli 一次（带界超时），不走 cli_retry。"""
    calls, sleeps = [], []
    monkeypatch.setattr(colab_runner.time, 'sleep', sleeps.append)

    def fake_cli(*args, capture=False, timeout=None):
        calls.append((args, timeout))
        raise core.BridgeError('read timeout')

    monkeypatch.setattr(colab_runner, 'cli', fake_cli)
    with pytest.raises(core.BridgeError, match='read timeout'):
        colab_runner.exec_starter_once('sess', colab_runner.starter_script())
    assert len(calls) == 1
    assert calls[0][0][:3] == ('exec', '-s', 'sess')
    assert calls[0][1] == colab_runner.EXEC_TIMEOUT
    assert sleeps == []  # 无退避 ⇒ 无重试
