"""doctor rclone 探针：成功 / 运行时失败 / 编程错误不得被吞。"""
import subprocess

import pytest

from mlbridge import cli
from mlbridge.core import BridgeError


def test_publish_materializes_links_by_default_with_opt_out(monkeypatch):
    calls = []
    def fake_publish(*args, **kwargs):
        calls.append((args, kwargs))
        return 'a' * 64
    monkeypatch.setattr(cli, 'publish', fake_publish)
    cli.execute(cli.parser().parse_args([
        'publish', 'demo', '--label', 'model', '--path', 'models/model']))
    assert calls[-1][1] == {'materialize_links': True}
    cli.execute(cli.parser().parse_args([
        'publish', 'demo', '--label', 'model', '--path', 'models/model',
        '--no-materialize-links']))
    assert calls[-1][1] == {'materialize_links': False}


def test_run_gpu_and_high_mem_options_are_optional_and_overrideable():
    assert cli.parser().parse_args(['run', 'demo']).gpu is None
    assert cli.parser().parse_args(['run', 'demo']).high_mem is None
    assert cli.parser().parse_args(['run', 'demo', '--gpu', 'A100', '--high-mem']).high_mem is True
    assert cli.parser().parse_args(['run', 'demo', '--no-high-mem']).high_mem is False


def test_run_passes_high_mem_request_to_launch(monkeypatch):
    seen = {}
    def fake_launch(*args, **kwargs):
        seen.update(kwargs)
        return 'session'
    monkeypatch.setattr(cli, 'launch', fake_launch)
    assert cli.execute(cli.parser().parse_args([
        'run', 'demo', '--gpu', 'A100', '--high-mem', '--detach'])) == 0
    assert seen['high_mem'] is True


def test_ssh_default_snapshot_and_custom_command(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, 'ssh_observe',
                        lambda *args, **kwargs: calls.append((args, kwargs)))
    assert cli.execute(cli.parser().parse_args(['ssh', '--timeout', '18', 'session'])) == 0
    assert calls[-1] == (('session', None), {'timeout': 18, 'identity': None})
    assert cli.execute(cli.parser().parse_args([
        'ssh', '--identity', '/tmp/id_ed25519', 'session', '--',
        'sh', '-c', 'echo safe'])) == 0
    assert calls[-1] == (('session', ['sh', '-c', 'echo safe']),
                         {'timeout': 60, 'identity': '/tmp/id_ed25519'})
    with pytest.raises(BridgeError, match='写在 <session> 前'):
        cli.execute(cli.parser().parse_args(['ssh', 'session', '--timeout', '10']))


def doctor_args():
    return cli.parser().parse_args(['doctor'])


def doctor_env(tmp_path, monkeypatch):
    """隔离 XDG，注入 settings/依赖，避免触碰真实用户配置。"""
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'xdg-config'))
    (tmp_path / 'pw').write_text('fake password')
    (tmp_path / 'rc').write_text('[drive]\ntype=drive\n')
    store = {'remote': 'drive', 'prefix': 'data',
             'password_file': str(tmp_path / 'pw'),
             'rclone_config': str(tmp_path / 'rc')}
    monkeypatch.setattr(cli, 'settings', lambda: store)
    monkeypatch.setattr(cli.shutil, 'which', lambda b: f'/usr/bin/{b}')
    monkeypatch.setattr(cli, 'restic_overwrite_supported', lambda: True)
    monkeypatch.setattr(cli, 'secret_env', lambda s: dict(s))
    return store


def test_doctor_rclone_probe_success(tmp_path, monkeypatch, capsys):
    doctor_env(tmp_path, monkeypatch)
    calls = []

    def fake_proc(cmd, **kwargs):
        calls.append(cmd)
        assert kwargs.get('capture') is True
        return subprocess.CompletedProcess(cmd, 0, stdout='')

    monkeypatch.setattr(cli, 'proc', fake_proc)
    assert cli.execute(doctor_args()) == 0
    out = capsys.readouterr().out
    assert 'CONNECTED' in out
    assert calls[0][:3] == ['rclone', 'about', 'drive:']


def test_doctor_rclone_probe_runtime_failure_is_unverified(tmp_path, monkeypatch,
                                                           capsys):
    doctor_env(tmp_path, monkeypatch)

    def fake_proc(cmd, **kwargs):
        raise BridgeError('命令失败(3): rclone 目录不存在或网络未连通')

    monkeypatch.setattr(cli, 'proc', fake_proc)
    assert cli.execute(doctor_args()) == 0
    out = capsys.readouterr().out
    assert 'UNVERIFIED' in out
    assert '网络未连通' in out


def test_doctor_rclone_probe_does_not_hide_programming_errors(tmp_path, monkeypatch):
    doctor_env(tmp_path, monkeypatch)

    def fake_proc(cmd, **kwargs):
        raise KeyError('RCLONE_CONFIG')  # 编程错误，不是网络/运行时失败

    monkeypatch.setattr(cli, 'proc', fake_proc)
    with pytest.raises(KeyError):
        cli.execute(doctor_args())


def test_doctor_probe_skipped_when_rclone_missing(tmp_path, monkeypatch, capsys):
    doctor_env(tmp_path, monkeypatch)
    monkeypatch.setattr(cli.shutil, 'which',
                        lambda b: None if b == 'rclone' else f'/usr/bin/{b}')
    called = []
    monkeypatch.setattr(cli, 'proc', lambda cmd, **kw: called.append(cmd))
    assert cli.execute(doctor_args()) == 0
    assert called == []
    out = capsys.readouterr().out
    assert 'CONNECTED' not in out and 'UNVERIFIED' not in out


def test_watch_stop_failure_returns_nonzero_even_after_success(monkeypatch):
    monkeypatch.setattr(cli, 'watch',
                        lambda *args, **kwargs: {'phase': 'SUCCEEDED',
                                                 'stop_error': 'network timeout'})
    assert cli.execute(cli.parser().parse_args(['watch', 'job'])) == 1


def test_run_stop_failure_returns_nonzero_even_after_success(monkeypatch):
    monkeypatch.setattr(cli, 'launch', lambda *args, **kwargs: 'job')
    monkeypatch.setattr(cli, 'watch',
                        lambda *args, **kwargs: {'phase': 'SUCCEEDED',
                                                 'stop_error': 'network timeout'})
    assert cli.execute(cli.parser().parse_args(['run', 'demo'])) == 1
