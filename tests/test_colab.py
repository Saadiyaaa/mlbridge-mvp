import json
import subprocess
from pathlib import Path

import pytest

from mlbridge import colab_runner, core

# autouse fixture 会把 list_active_sessions 打桩为空；需要真实实现（打桩
# 底层 cli）的解析测试从这里取回原函数。
_REAL_LIST_ACTIVE = colab_runner.list_active_sessions


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    monkeypatch.setenv('XDG_STATE_HOME', str(tmp_path / 'state'))
    monkeypatch.delenv('MLBRIDGE_GITHUB_USERNAME', raising=False)
    monkeypatch.delenv('MLBRIDGE_GITHUB_TOKEN', raising=False)
    monkeypatch.setattr(colab_runner, '_github_secrets_path',
                        lambda: tmp_path / 'isolated-secrets.env')
    # 单实例守卫默认打桩为空，已有测试不触网；守卫本体由专属测试覆盖。
    monkeypatch.setattr(colab_runner, 'list_active_sessions', lambda: [])


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


# ---------------------------------------------------------------------------
# 单实例分配守卫（fail-closed）：默认 max_active_sessions=1
# ---------------------------------------------------------------------------

def test_list_active_sessions_parses_empty_and_active_lines(monkeypatch):
    monkeypatch.setattr(colab_runner, 'list_active_sessions', _REAL_LIST_ACTIVE)
    def fake_cli(*args, capture=False, timeout=None):
        assert args == ('sessions',) and capture is True
        assert timeout == colab_runner.SESSIONS_TIMEOUT
        return subprocess.CompletedProcess(args, 0, stdout='[colab] No active sessions found on server.\n')
    monkeypatch.setattr(colab_runner, 'cli', fake_cli)
    assert colab_runner.list_active_sessions() == []

    def fake_cli_active(*args, capture=False, timeout=None):
        return subprocess.CompletedProcess(args, 0, stdout=(
            '[mlb-demo-ab12cd34] endpoint-1 | Hardware: A100 | Shape: ... | Variant: GPU\n'
            '[other] endpoint-2 | Hardware: T4 | Shape: ... | Variant: DEFAULT\n'))
    monkeypatch.setattr(colab_runner, 'cli', fake_cli_active)
    assert colab_runner.list_active_sessions() == ['mlb-demo-ab12cd34', 'other']


def test_list_active_sessions_fails_closed_on_cli_error_and_garbage(monkeypatch):
    """CLI 失败/超时/格式未知 ⇒ 一律视为拒绝分配，绝不当作空。"""
    monkeypatch.setattr(colab_runner, 'list_active_sessions', _REAL_LIST_ACTIVE)
    def boom(*args, **kwargs):
        raise core.BridgeError('命令超时: colab')
    monkeypatch.setattr(colab_runner, 'cli', boom)
    with pytest.raises(core.BridgeError, match='超时'):
        colab_runner.list_active_sessions()

    def garbage(*args, capture=False, timeout=None):
        return subprocess.CompletedProcess(args, 0, stdout='??????\n')
    monkeypatch.setattr(colab_runner, 'cli', garbage)
    with pytest.raises(core.BridgeError, match='无法解析'):
        colab_runner.list_active_sessions()

    def empty_but_weird(*args, capture=False, timeout=None):
        return subprocess.CompletedProcess(args, 0, stdout='')
    monkeypatch.setattr(colab_runner, 'cli', empty_but_weird)
    with pytest.raises(core.BridgeError, match='按未知处理'):
        colab_runner.list_active_sessions()


def test_check_single_active_empty_other_and_own():
    assert colab_runner.check_single_active('s1', 1, []) == []
    # 已有 1 个他人实例：达到上限，拒绝
    with pytest.raises(core.BridgeError, match='单实例策略拒绝分配'):
        colab_runner.check_single_active('new', 1, ['existing'])
    # 活跃 session 不可复用；limit=0 只关闭数量上限。
    with pytest.raises(core.BridgeError, match='已在 colab 服务器上活跃'):
        colab_runner.check_single_active('existing', 1, ['existing'])
    with pytest.raises(core.BridgeError, match='已在 colab 服务器上活跃'):
        colab_runner.check_single_active('existing', 0, ['existing'])
    # limit=2 时 1 个他人实例放行；limit=0 只关闭上限计数（own 未活跃时）
    assert colab_runner.check_single_active('new', 2, ['existing']) == ['existing']
    assert colab_runner.check_single_active('new', 0, ['busy']) == []


def test_check_single_active_never_stops_or_kills(monkeypatch):
    """守卫拒绝时只抛错：绝不调用 stop/shutdown。"""
    def deny_shutdown(session):
        raise AssertionError('guard must not stop any instance')
    monkeypatch.setattr(colab_runner, 'shutdown', deny_shutdown)
    with pytest.raises(core.BridgeError):
        colab_runner.check_single_active('new', 1, ['busy'])
    with pytest.raises(core.BridgeError):
        colab_runner.check_single_active('own', 1, ['own'])


def test_launch_guard_refuses_before_new_and_without_local_record(tmp_path, monkeypatch):
    """已有他人活跃实例 ⇒ 在 `colab new` 之前拒绝，且不留本地任务记录。"""
    _prepare_project(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(colab_runner, 'cli',
                        lambda *a, **k: calls.append(a) or subprocess.CompletedProcess(a, 0))
    monkeypatch.setattr(colab_runner, 'list_active_sessions', lambda: ['mlb-other-0000'])
    with pytest.raises(core.BridgeError, match='单实例策略拒绝分配'):
        colab_runner.launch('demo', 'example', [], 'L4', allow_empty_input=True)
    assert not any(c[0] == 'new' for c in calls)
    assert not list((core.state_dir() / 'jobs').glob('*.json'))


def test_launch_guard_cli_failure_is_refusal_not_retry(tmp_path, monkeypatch):
    """`colab sessions` 本身失败 ⇒ 拒绝分配（不重试、不分配、不清理）。"""
    _prepare_project(tmp_path, monkeypatch)
    monkeypatch.setattr(colab_runner, 'list_active_sessions', _REAL_LIST_ACTIVE)
    calls = []
    sleeps = []
    monkeypatch.setattr(colab_runner.time, 'sleep', sleeps.append)

    def fake_cli(*args, capture=False, timeout=None):
        calls.append(args)
        if args[0] == 'sessions':
            raise core.BridgeError('read timeout')
        return subprocess.CompletedProcess(args, 0)
    monkeypatch.setattr(colab_runner, 'cli', fake_cli)
    with pytest.raises(core.BridgeError, match='read timeout'):
        colab_runner.launch('demo', 'example', [], 'L4', allow_empty_input=True)
    assert [c[0] for c in calls] == ['sessions']  # 绝不走到 new
    assert sleeps == []  # 无重试
    assert not list((core.state_dir() / 'jobs').glob('*.json'))


def test_launch_queries_sessions_before_new(tmp_path, monkeypatch):
    """正常路径：launch 先查 sessions（守卫）再分配 new。"""
    _prepare_project(tmp_path, monkeypatch)
    monkeypatch.setattr(colab_runner, 'list_active_sessions', _REAL_LIST_ACTIVE)
    order = []

    def fake_cli(*args, capture=False, timeout=None):
        order.append(args[0])
        if args[0] == 'sessions':
            return subprocess.CompletedProcess(
                args, 0, stdout='[colab] No active sessions found on server.\n')
        return subprocess.CompletedProcess(args, 0)
    monkeypatch.setattr(colab_runner, 'cli', fake_cli)
    monkeypatch.setattr(colab_runner, 'stage', lambda *a: None)
    monkeypatch.setattr(colab_runner, 'put_script', lambda *a: None)
    session = colab_runner.launch('demo', 'example', [], 'L4', allow_empty_input=True)
    assert order.index('sessions') < order.index('new')
    assert order.count('new') == 1
    assert colab_runner.get_local(session)['stopped'] is False


def test_launch_max_active_sessions_zero_skips_limit_count(tmp_path, monkeypatch):
    """--max-active-sessions 0 ⇒ 关闭并发上限计数：多个他人实例也放行。"""
    _prepare_project(tmp_path, monkeypatch)
    monkeypatch.setattr(colab_runner, 'list_active_sessions',
                        lambda: ['other-1', 'other-2'])
    monkeypatch.setattr(colab_runner, 'cli',
                        lambda *a, **k: subprocess.CompletedProcess(a, 0))
    monkeypatch.setattr(colab_runner, 'stage', lambda *a: None)
    monkeypatch.setattr(colab_runner, 'put_script', lambda *a: None)
    session = colab_runner.launch('demo', 'example', [], 'L4',
                                  allow_empty_input=True, max_active_sessions=0)
    assert colab_runner.get_local(session)['stopped'] is False


def test_launch_disabled_limit_still_refuses_own_active_session(tmp_path, monkeypatch):
    """limit=0 时复用活跃 session 仍拒绝分配、停止和写本地记录。"""
    _prepare_project(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(colab_runner, 'list_active_sessions', lambda: ['mlb-own-0001'])
    monkeypatch.setattr(colab_runner, 'cli',
                        lambda *a, **k: calls.append(a) or subprocess.CompletedProcess(a, 0))
    monkeypatch.setattr(colab_runner, 'shutdown',
                        lambda s: (_ for _ in ()).throw(
                            AssertionError('must not stop on guard refusal')))
    with pytest.raises(core.BridgeError, match='已在 colab 服务器上活跃'):
        colab_runner.launch('demo', 'example', [], 'L4', session='mlb-own-0001',
                            allow_empty_input=True, max_active_sessions=0)
    assert not any(c[0] == 'new' for c in calls)
    assert not any(c[0] == 'stop' for c in calls)
    assert not list((core.state_dir() / 'jobs').glob('*.json'))


def test_launch_negative_limit_rejected(tmp_path, monkeypatch):
    _prepare_project(tmp_path, monkeypatch)
    monkeypatch.setattr(colab_runner, 'cli',
                        lambda *a, **k: subprocess.CompletedProcess(a, 0))
    with pytest.raises(core.BridgeError, match='max-active-sessions'):
        colab_runner.launch('demo', 'example', [], 'L4', allow_empty_input=True,
                            max_active_sessions=-1)


def test_launch_configured_limit_from_settings(tmp_path, monkeypatch):
    """settings.toml [colab].max_active_sessions=2 ⇒ 允许第 2 个实例。"""
    _prepare_project(tmp_path, monkeypatch)
    with (core.config_dir() / 'settings.toml').open('a') as f:
        f.write('\n[colab]\ndefault_gpu="L4"\nmax_active_sessions=2\n')
    monkeypatch.setattr(colab_runner, 'list_active_sessions', lambda: ['one'])
    monkeypatch.setattr(colab_runner, 'cli',
                        lambda *a, **k: subprocess.CompletedProcess(a, 0))
    monkeypatch.setattr(colab_runner, 'stage', lambda *a: None)
    monkeypatch.setattr(colab_runner, 'put_script', lambda *a: None)
    # 1 个他人实例 < limit 2 ⇒ 放行
    session = colab_runner.launch('demo', 'example', [], allow_empty_input=True)
    assert colab_runner.get_local(session)['stopped'] is False


def test_launch_configured_limit_one_refuses(tmp_path, monkeypatch):
    """settings.toml [colab].max_active_sessions=1（缺省）+ 1 个他人实例 ⇒ 拒绝。"""
    _prepare_project(tmp_path, monkeypatch)
    with (core.config_dir() / 'settings.toml').open('a') as f:
        f.write('\n[colab]\ndefault_gpu="L4"\nmax_active_sessions=1\n')
    monkeypatch.setattr(colab_runner, 'list_active_sessions', lambda: ['one'])
    monkeypatch.setattr(colab_runner, 'cli',
                        lambda *a, **k: subprocess.CompletedProcess(a, 0))
    with pytest.raises(core.BridgeError, match='单实例策略拒绝分配'):
        colab_runner.launch('demo', 'example', [], allow_empty_input=True)


def test_launch_lock_file_created_and_released(tmp_path, monkeypatch):
    """跨进程 flock：锁文件在 state 目录下创建，且可重复获取（无死锁）。"""
    _prepare_project(tmp_path, monkeypatch)
    monkeypatch.setattr(colab_runner, 'cli',
                        lambda *a, **k: subprocess.CompletedProcess(a, 0))
    monkeypatch.setattr(colab_runner, 'stage', lambda *a: None)
    monkeypatch.setattr(colab_runner, 'put_script', lambda *a: None)
    colab_runner.launch('demo', 'example', [], 'CPU', allow_empty_input=True)
    colab_runner.launch('demo', 'example', [], 'CPU', allow_empty_input=True,
                        max_active_sessions=0)
    assert (core.state_dir() / 'launch.lock').exists()


def test_watch_records_last_known_and_terminal_diagnostic(tmp_path, monkeypatch):
    """watch 观测写入本地 last-known；终态另存诊断文件（无凭据）。"""
    colab_runner.store_local('job', {'stopped': False})
    monkeypatch.setattr(colab_runner, 'get_status',
                        lambda s: {'phase': 'FAILED', 'error': 'RuntimeError: boom'})
    monkeypatch.setattr(colab_runner, 'shutdown',
                        lambda s: (_ for _ in ()).throw(AssertionError('must not stop')))
    result = colab_runner.watch('job', interval=5)
    assert result['phase'] == 'FAILED'
    record = colab_runner.get_local('job')
    assert record['last_known']['phase'] == 'FAILED' and 'observed' in record['last_known']
    diag_path = colab_runner.diagnostic_path('job')
    assert diag_path.is_file()
    payload = json.loads(diag_path.read_text())
    assert payload['terminal']['error'] == 'RuntimeError: boom'
    assert '不构成' in payload['note']
    blob = diag_path.read_text()
    assert 'password' not in blob.lower() and 'token' not in blob.lower()


def test_record_observation_ignores_missing_record():
    """没有本地记录时（如手工观察未知 session）静默跳过，绝不上抛。"""
    colab_runner.record_observation('ghost', {'phase': 'RUNNING'})  # 不应抛错


def test_record_observation_keeps_last_confirmed_phase_over_unreachable(tmp_path):
    """UNREACHABLE 绝不覆盖最后一次确认的真实远端阶段。

    watch 反复遇网/超时会把 last_known RUNNING 刷成 UNREACHABLE，丢失
    “消失前在干什么”的全部信息；正确行为是跳过而不是写入。
    """
    colab_runner.store_local('job', {'stopped': False})
    # 首次是真观测：RUNNING 被记录
    colab_runner.record_observation('job', {'phase': 'RUNNING'})
    first = colab_runner.get_local('job')['last_known']
    assert first['phase'] == 'RUNNING'
    # 连续多次 UNREACHABLE：last_known 保持 RUNNING，观测时间戳不变
    for _ in range(6):
        colab_runner.record_observation('job', {'phase': 'UNREACHABLE',
                                                'reason': 'connection lost'})
    last = colab_runner.get_local('job')['last_known']
    assert last['phase'] == 'RUNNING'
    assert last['observed'] == first['observed']
    # 再次确认到真实阶段后正常覆盖
    colab_runner.record_observation('job', {'phase': 'RESTORING'})
    assert colab_runner.get_local('job')['last_known']['phase'] == 'RESTORING'


def test_record_observation_writes_unreachable_without_prior_phase():
    """没有已确认阶段时，UNREACHABLE 可以落盘（此时没有可保留的东西）。"""
    colab_runner.store_local('fresh', {'stopped': False})
    colab_runner.record_observation('fresh', {'phase': 'UNREACHABLE'})
    assert colab_runner.get_local('fresh')['last_known']['phase'] == 'UNREACHABLE'


def test_watch_preserves_running_last_known_through_unreachable_spell(monkeypatch):
    """RUNNING 后连续 6 次 UNREACHABLE ⇒ 返回 UNKNOWN，但 last_known
    仍是最后确认的 RUNNING，而不是被刷成 UNREACHABLE。"""
    colab_runner.store_local('job', {'stopped': False})
    phases = iter(['RUNNING'] + ['UNREACHABLE'] * 6)
    monkeypatch.setattr(colab_runner, 'get_status',
                        lambda s: {'phase': next(phases)})
    monkeypatch.setattr(colab_runner.time, 'sleep', lambda s: None)
    monkeypatch.setattr(colab_runner, 'shutdown',
                        lambda s: (_ for _ in ()).throw(
                            AssertionError('must not stop on unknown status')))
    result = colab_runner.watch('job', interval=5)
    assert result['phase'] == 'UNKNOWN'
    record = colab_runner.get_local('job')
    assert record['last_known']['phase'] == 'RUNNING'


def test_concurrent_stop_and_watch_record_merge(monkeypatch):
    """并发 stop/watch 状态写入：per-job flock 使读改写合并，字段不丢。

    多线程反复并发执行 record_observation（watch 的写路径）与 shutdown
    （stop 的写路径）。关键是最终状态必须同时包含两边写入的字段；
    若写路径不加锁，record_observation 的整份覆写会丢掉并发写入的
    stopped（或反之）。锁文件落在 jobs/<session>.lock。
    """
    colab_runner.store_local('job', {'stopped': False})
    monkeypatch.setattr(colab_runner, 'cli',
                        lambda *a, **k: subprocess.CompletedProcess(a, 0))

    import threading
    errors: list[Exception] = []

    def watcher():
        try:
            for i in range(24):
                colab_runner.record_observation('job', {'phase': 'RUNNING'})
                if i % 4 == 0:
                    colab_runner.shutdown('job')  # 与其他线程的 stop 并发
        except Exception as exc:  # pragma: no cover - 仅在出现真正错误时触发
            errors.append(exc)

    threads = [threading.Thread(target=watcher) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert errors == []
    record = colab_runner.get_local('job')
    # 两路写入都被保留（merge 而非丢字段覆盖）
    assert record['stopped'] is True
    assert record['last_known']['phase'] == 'RUNNING'
    assert 'observed' in record['last_known']
    # per-job 锁文件存在
    assert colab_runner.state_path('job').with_suffix('.lock').is_file()
