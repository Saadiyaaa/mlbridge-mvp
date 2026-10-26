"""launch 边界本地诊断留证的离线测试（fake colab CLI，不触网）。"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from mlbridge import colab_runner, core

_REAL_LIST_ACTIVE = colab_runner.list_active_sessions

SESSION = 'diag-sess'


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    monkeypatch.setenv('XDG_STATE_HOME', str(tmp_path / 'state'))
    monkeypatch.setattr(colab_runner, 'list_active_sessions', lambda: [])
    monkeypatch.setattr(colab_runner.time, 'sleep', lambda s: None)


def _trace():
    path = colab_runner.launch_trace_path(SESSION)
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text().splitlines()]


def _fake_cli(monkeypatch, capsys, *, returncode=0, stdout='', stderr='',
              fail=None):
    calls = []
    kwargs_seen = {}

    def fake(*args, **kwargs):
        calls.append(args)
        kwargs_seen.update(kwargs)
        if fail is not None:
            raise core.BridgeError(fail)
        r = subprocess.CompletedProcess(args, returncode)
        r.stdout, r.stderr = stdout, stderr
        return r
    return fake, calls, kwargs_seen


def test_echo_accepts_legacy_no_result_and_byte_output(capsys):
    colab_runner._echo(None)
    colab_runner._echo(subprocess.CompletedProcess([], 0, b'bytes', b'bytes'))
    assert capsys.readouterr().out == ''


def test_legacy_empty_completion_does_not_change_starter_or_stage(monkeypatch, tmp_path):
    monkeypatch.setattr(colab_runner, 'cli', lambda *a, **kw: None)
    monkeypatch.setattr(colab_runner, 'cli_retry', lambda *a, **kw: None)
    assert colab_runner.exec_starter_once(SESSION, 'no-content-capture') is None
    colab_runner.put_script(SESSION, 'no-content-capture')
    src = tmp_path / 'small.json'
    src.write_text('{}')
    colab_runner.stage(SESSION, src, 'small.json')
    records = _trace()
    assert len(records) == 3
    assert all(r['rc'] is None for r in records)
    assert all(r['stdout'] == '' and r['stderr'] == '' for r in records)


def test_put_script_traces_exec_boundary_and_echoes(tmp_path, monkeypatch, capsys):
    fake, calls, kwargs = _fake_cli(
        monkeypatch, capsys, stdout='remote echo out\n', stderr='remote warn\n')
    monkeypatch.setattr(colab_runner, 'cli_retry', fake)
    colab_runner.put_script(SESSION, 'SECRET_SCRIPT_TEXT = 1')
    assert calls == [('exec', '-s', SESSION, '-f', calls[0][4])]
    assert capsys.readouterr().out.strip() == 'remote echo out'
    events = _trace()
    kinds = [e['kind'] for e in events if e['event'] == 'exec']
    assert kinds == ['put-script']
    last = [e for e in events if e['event'] == 'exec'][-1]
    assert last['rc'] == 0 and 'remote echo out' in last['stdout']
    # 脚本内容绝不入本地轨迹
    assert all('SECRET_SCRIPT_TEXT' not in json.dumps(e) for e in events)


def test_put_script_failure_traces_error(tmp_path, monkeypatch):
    fake, calls, _ = _fake_cli(monkeypatch, None, fail='channel lost')
    monkeypatch.setattr(colab_runner, 'cli_retry', fake)
    with pytest.raises(core.BridgeError, match='channel lost'):
        colab_runner.put_script(SESSION, 'whatever')
    events = [e for e in _trace() if e['event'] == 'exec-failed']
    assert len(events) == 1 and events[0]['kind'] == 'put-script'
    assert 'channel lost' in events[0]['error']


def test_starter_executed_once_and_pid_recorded(tmp_path, monkeypatch, capsys):
    fake, calls, _ = _fake_cli(
        monkeypatch, capsys, stdout='MLBridge remote PID: 4321\n')
    monkeypatch.setattr(colab_runner, 'cli', fake)
    pid = colab_runner.exec_starter_once(SESSION, 'STARTER_TEXT')
    assert pid == 4321
    assert len(calls) == 1  # starter 只执行一次，无重试
    events = [e for e in _trace() if e['event'] == 'starter']
    assert len(events) == 1 and events[0]['rc'] == 0
    assert 'MLBridge remote PID: 4321' in events[0]['stdout']


def test_starter_without_parseable_pid_returns_none(tmp_path, monkeypatch):
    fake, _, _ = _fake_cli(monkeypatch, None, stdout='worker spawned\n')
    monkeypatch.setattr(colab_runner, 'cli', fake)
    assert colab_runner.exec_starter_once(SESSION, 'x') is None
    events = [e for e in _trace() if e['event'] == 'starter']
    assert len(events) == 1


def test_starter_failure_traces_without_retry(tmp_path, monkeypatch):
    fake, calls, _ = _fake_cli(monkeypatch, None, fail='exec transport died')
    monkeypatch.setattr(colab_runner, 'cli', fake)
    with pytest.raises(core.BridgeError):
        colab_runner.exec_starter_once(SESSION, 'x')
    assert len(calls) == 1  # 绝不因回复丢失而重试 starter
    events = [e for e in _trace() if e['event'] == 'starter-failed']
    assert len(events) == 1
    assert '只执行一次' in events[0]['note']
    assert 'exec transport died' in events[0]['error']


def test_stage_traces_and_echoes(tmp_path, monkeypatch, capsys):
    fake, calls, kwargs = _fake_cli(
        monkeypatch, capsys, stdout='uploaded mlbridge_plan.json\n')
    monkeypatch.setattr(colab_runner, 'cli_retry', fake)
    src = tmp_path / 'plan.json'
    src.write_text('PLAN_CONTENT_TAIL')
    colab_runner.stage(SESSION, src, 'mlbridge_plan.json')
    assert calls == [('upload', '-s', SESSION, str(src), '/content/mlbridge_plan.json')]
    events = [e for e in _trace() if e['event'] == 'stage']
    assert len(events) == 1 and events[0]['remote'] == 'mlbridge_plan.json'
    assert events[0]['rc'] == 0
    assert all('PLAN_CONTENT_TAIL' not in json.dumps(e)
               for e in _trace())  # 源文件内容不入轨迹


def test_stage_failure_traces_remote_and_error(tmp_path, monkeypatch):
    fake, _, _ = _fake_cli(monkeypatch, None, fail='upload broken')
    monkeypatch.setattr(colab_runner, 'cli_retry', fake)
    src = tmp_path / 'plan.json'
    src.write_text('PLAN_CONTENT_TAIL')
    with pytest.raises(core.BridgeError, match='upload broken'):
        colab_runner.stage(SESSION, src, 'mlbridge_plan.json')
    events = [e for e in _trace() if e['event'] == 'stage-failed']
    assert len(events) == 1
    assert events[0]['remote'] == 'mlbridge_plan.json'
    assert 'upload broken' in events[0]['error']
    assert all('PLAN_CONTENT_TAIL' not in json.dumps(e)
               for e in _trace())


@pytest.mark.parametrize('phase', ['RUNNING', 'UNREACHABLE'])
def test_staging_probe_never_stops_running_or_unknown(tmp_path, monkeypatch, phase):
    monkeypatch.setattr(colab_runner, 'get_status', lambda s: {'phase': phase})
    def deny_shutdown(name):
        raise AssertionError('must not shutdown on any probe phase')
    monkeypatch.setattr(colab_runner, 'shutdown', deny_shutdown)
    colab_runner._staging_failed(SESSION, starter_sent=True)
    events = [e for e in _trace() if e['event'] == 'staging-probe']
    assert len(events) == 1 and events[0]['phase'] == phase
    assert events[0]['note']


def test_trace_is_bounded_and_jsonl_per_session(tmp_path, monkeypatch):
    for i in range(300):
        colab_runner._trace_launch(SESSION, {'event': 'ping', 'n': i,
                                             'pad': 'x' * 4096})
    path = colab_runner.launch_trace_path(SESSION)
    lines = path.read_text().splitlines()
    text = path.read_text()
    # 超过 512KB 后裁剪到最近 64KB：不丢 JSONL 结构，总量有限
    assert 64 * 1024 // 2 <= len(text.encode('utf-8')) <= colab_runner._TRACE_MAX_BYTES
    records = [json.loads(ln) for ln in lines]
    assert all(isinstance(item, dict) for item in records)
    assert records[0]['n'] > 0  # old rows really rotated, not just an upper bound
    assert records[-1]['n'] == 299
    other = colab_runner.launch_trace_path('diag-other')
    colab_runner._trace_launch('diag-other', {'event': 'ping'})
    assert path.name != other.name  # per-session 独立文件


def test_trace_rotation_preserves_unicode_rows_and_bounds_one_large_event():
    for i in range(250):
        colab_runner._trace_launch(SESSION, {'event': 'unicode', 'n': i,
                                           'pad': '界' * 1200})
    path = colab_runner.launch_trace_path(SESSION)
    records = [json.loads(ln) for ln in path.read_text().splitlines()]
    assert records[-1]['n'] == 249
    assert path.stat().st_size <= colab_runner._TRACE_MAX_BYTES
    colab_runner._trace_launch(SESSION, {'event': 'huge', 'pad': 'x' * 800000})
    last = json.loads(path.read_text().splitlines()[-1])
    assert last['oversize_event_omitted'] is True
    assert last['event'] == 'huge'
    assert path.stat().st_size <= colab_runner._TRACE_MAX_BYTES
