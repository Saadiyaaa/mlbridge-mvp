"""ssh_observe 离线测试：隔离存储、绝不自动建机/绝不 --rm、连接前拒绝。

全程 monkeypatch，无真实 SSH、网络或 Colab VM；不调用任何 `colab new`。
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path

import pytest

from mlbridge import core, ssh_observe

SESSION_A, SESSION_B = 'mlb-a', 'mlb-b'
ENTRY_A = {'name': SESSION_A, 'token': 'TOKEN-A-SECRET',
           'url': 'https://colab.example/a', 'endpoint': 'ep-a'}
ENTRY_B = {'name': SESSION_B, 'token': 'TOKEN-B-SECRET',
           'url': 'https://colab.example/b', 'endpoint': 'ep-b'}


@pytest.fixture
def store(tmp_path, monkeypatch):
    """真实的 colab-cli 会话存储替身；通过受控绝对路径重定向。"""
    path = tmp_path / 'colab-cli' / 'sessions.json'

    def _make(entries: dict | None = None) -> Path:
        # None = 正常双会话；{} = 空存储
        entries = ({SESSION_A: ENTRY_A, SESSION_B: ENTRY_B}
                   if entries is None else entries)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(entries))
        path.chmod(0o600)
        monkeypatch.setenv('MLBRIDGE_COLAB_SESSIONS', str(path))
        return path

    return _make


@pytest.fixture
def fake_bins(tmp_path, monkeypatch):
    colab = tmp_path / 'bin' / 'colab'
    colab.parent.mkdir(parents=True, exist_ok=True)
    colab.write_text('', encoding='utf-8')
    colab.chmod(0o755)
    monkeypatch.setattr(ssh_observe.shutil, 'which',
                        lambda name: str(colab) if name == 'colab' else None)
    key = tmp_path / 'keys' / 'id_ed25519'
    key.parent.mkdir(parents=True, exist_ok=True)
    key.write_text('-----BEGIN FAKE KEY-----\n')
    key.chmod(0o600)
    return colab, key


@pytest.fixture
def ssh_traffic(monkeypatch, tmp_path):
    """替换 ssh_observe._run；从 ProxyCommand 解析调用时的隔离存储状态。"""
    seen: list[dict] = []

    def _parse(argv) -> dict:
        proxy = next(o for o in argv if o.startswith('ProxyCommand='))
        parts = shlex.split(proxy.removeprefix('ProxyCommand='))
        config = Path(parts[parts.index('--config') + 1])
        record = {'argv': argv, 'proxy_parts': parts, 'config': config,
                  'timeout': None}
        return record

    def record_state(record):
        record['config_entry'] = json.loads(record['config'].read_text())
        record['config_mode'] = os.stat(record['config']).st_mode & 0o777
        record['dir_mode'] = os.stat(record['config'].parent).st_mode & 0o777

    def ok(returncode=0, stderr=''):
        def fake_run(argv, *, timeout, **kw):
            record = _parse(argv)
            record['timeout'] = timeout
            record_state(record)
            seen.append(record)
            return subprocess.CompletedProcess(argv, returncode, None, stderr)
        return fake_run

    def timeout(seconds=99):
        def fake_run(argv, *, timeout, **kw):
            seen.append(_parse(argv))
            raise subprocess.TimeoutExpired(argv[0], seconds)
        return fake_run

    def reject():
        def fake_run(*args, **kw):
            raise AssertionError('must not reach the ssh subprocess')
        return fake_run

    return {'seen': seen, 'ok': ok, 'timeout': timeout, 'reject': reject,
            'install': lambda fake_run: monkeypatch.setattr(
                ssh_observe, '_run', fake_run)}


def key_path(fake_bins) -> str:
    return str(fake_bins[1])


# ---------- 连接前拒绝（绝不先碰 ssh） ----------

def test_unknown_session_refused_before_ssh(store, fake_bins, ssh_traffic):
    store()  # 有 A、B，但没有 mlb-z
    ssh_traffic['install'](ssh_traffic['reject']())
    with pytest.raises(core.BridgeError, match='mlb-z'):
        ssh_observe.ssh_observe('mlb-z', identity=key_path(fake_bins))


def test_empty_and_missing_store_refused_before_ssh(
        store, fake_bins, ssh_traffic, monkeypatch, tmp_path):
    ssh_traffic['install'](ssh_traffic['reject']())
    store({})  # 空存储
    with pytest.raises(core.BridgeError, match='没有任何活跃会话'):
        ssh_observe.ssh_observe(SESSION_A, identity=key_path(fake_bins))
    monkeypatch.setenv('MLBRIDGE_COLAB_SESSIONS',
                       str(tmp_path / 'gone.json'))  # 存储文件缺失
    with pytest.raises(core.BridgeError, match='找不到 colab-cli 会话存储'):
        ssh_observe.ssh_observe(SESSION_A, identity=key_path(fake_bins))


def test_corrupt_store_refused_without_leaking(tmp_path, fake_bins,
                                               ssh_traffic, monkeypatch):
    path = tmp_path / 'broken.json'
    path.write_text('{"%s": {"token": "TOKEN-A-SECRET", "broken":' % SESSION_A)
    monkeypatch.setenv('MLBRIDGE_COLAB_SESSIONS', str(path))
    ssh_traffic['install'](ssh_traffic['reject']())
    with pytest.raises(core.BridgeError, match='不可读') as info:
        ssh_observe.ssh_observe(SESSION_A, identity=key_path(fake_bins))
    assert 'TOKEN-A-SECRET' not in str(info.value)


def test_invalid_session_id_and_store_shape(store, fake_bins, ssh_traffic):
    ssh_traffic['install'](ssh_traffic['reject']())
    store()
    with pytest.raises(core.BridgeError, match='不合法'):
        ssh_observe.ssh_observe('../escape', identity=key_path(fake_bins))
    store({SESSION_A: 'not-a-dict'})  # 条目不是 dict
    with pytest.raises(core.BridgeError, match='不在本地'):
        ssh_observe.ssh_observe(SESSION_A, identity=key_path(fake_bins))


def test_relative_redirection_rejected(fake_bins, ssh_traffic, monkeypatch):
    monkeypatch.setenv('MLBRIDGE_COLAB_SESSIONS', 'relative/path.json')
    ssh_traffic['install'](ssh_traffic['reject']())
    with pytest.raises(core.BridgeError, match='绝对路径'):
        ssh_observe.ssh_observe(SESSION_A, identity=key_path(fake_bins))


# ---------- 身份与 colab 定位（连接前） ----------

def test_no_compatible_identity_refuses(store, fake_bins, ssh_traffic,
                                        monkeypatch, tmp_path):
    store()
    home = tmp_path / 'naked_home'
    (home / '.ssh').mkdir(parents=True)
    monkeypatch.setenv('HOME', str(home))
    monkeypatch.setenv('XDG_CONFIG_HOME', str(home / '.config'))
    ssh_traffic['install'](ssh_traffic['reject']())
    with pytest.raises(core.BridgeError, match='绝不生成密钥'):
        ssh_observe.ssh_observe(SESSION_A)


def test_nonexistent_identity_path_refuses(store, fake_bins, ssh_traffic):
    store()
    ssh_traffic['install'](ssh_traffic['reject']())
    with pytest.raises(core.BridgeError, match='不存在'):
        ssh_observe.ssh_observe(SESSION_A, identity='/no/such/key')


def test_missing_colab_binary_refuses(tmp_path, monkeypatch, store,
                                      fake_bins, ssh_traffic):
    store()
    monkeypatch.setattr(ssh_observe.shutil, 'which', lambda name: None)
    ssh_traffic['install'](ssh_traffic['reject']())
    with pytest.raises(core.BridgeError, match='colab'):
        ssh_observe.ssh_observe(SESSION_A, identity=key_path(fake_bins))


# ---------- 隔离存储：内容、权限、清理 ----------

def test_isolated_copy_content_modes_and_cleanup(store, fake_bins,
                                                 ssh_traffic):
    path = store()
    ssh_traffic['install'](ssh_traffic['ok']())
    assert ssh_observe.ssh_observe(SESSION_A, identity=key_path(fake_bins)) == 0
    record = ssh_traffic['seen'][0]
    assert record['config_entry'] == {SESSION_A: ENTRY_A}
    assert 'TOKEN-A-SECRET' in json.dumps(record['config_entry'])
    assert 'TOKEN-B-SECRET' not in json.dumps(record['config_entry'])
    assert 'TOKEN-A-SECRET' in path.read_text()      # 源存储未被改动
    assert record['config_mode'] == 0o600            # 隔离存储 0600
    assert record['dir_mode'] == 0o700               # 临时目录 0700
    # 含 token 副本的临时目录已清理
    assert not record['config'].exists()
    assert not record['config'].parent.exists()


# ---------- ssh / ProxyCommand 参数边界 ----------

def test_ssh_and_proxy_argv_have_no_side_effects(store, fake_bins,
                                                 ssh_traffic):
    store()
    ssh_traffic['install'](ssh_traffic['ok']())
    ssh_observe.ssh_observe(SESSION_A, identity=key_path(fake_bins))
    record = ssh_traffic['seen'][0]
    argv, parts = record['argv'], record['proxy_parts']
    assert argv[0] == 'ssh'
    assert argv[argv.index('-F') + 1] == '/dev/null'
    assert '-T' in argv
    flat = ' '.join(argv)
    for opt in ('BatchMode=yes', 'ConnectTimeout=15',
                'StrictHostKeyChecking=accept-new', 'UserKnownHostsFile='):
        assert opt in flat
    # 仅对隔离存储里已存在的目标传 -s；既不会误建也不自动销毁。
    assert parts[parts.index('-s') + 1] == SESSION_A
    assert '--rm' not in parts
    assert 'new' not in parts
    assert '--proxy-mode' in parts
    assert '--rm' not in flat
    assert '-s' not in argv
    assert argv[argv.index('--') + 1] == 'root@colab-runtime'
    assert record['timeout'] >= 5
    assert len(ssh_traffic['seen']) == 1  # 单次调用，无 stop/重试类后手


def test_selected_identity_used_by_both_layers(store, fake_bins, ssh_traffic):
    store()
    _, key = fake_bins
    ssh_traffic['install'](ssh_traffic['ok']())
    ssh_observe.ssh_observe(SESSION_A, identity=str(key))
    record = ssh_traffic['seen'][0]
    parts, argv = record['proxy_parts'], record['argv']
    assert '--identity' in parts
    assert parts[parts.index('--identity') + 1] == str(key)
    assert argv[argv.index('-i') + 1] == str(key)


def test_default_identity_prefers_mlbridge_specific_key(tmp_path, monkeypatch,
                                                       store, fake_bins, ssh_traffic):
    store()
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'config'))
    key = tmp_path / 'config/mlbridge/ssh/id_ed25519'
    key.parent.mkdir(parents=True)
    key.write_text('key')
    key.chmod(0o600)
    ssh_traffic['install'](ssh_traffic['ok']())
    ssh_observe.ssh_observe(SESSION_A)
    parts = ssh_traffic['seen'][0]['proxy_parts']
    assert parts[parts.index('--identity') + 1] == str(key)


def test_default_identity_discovery_via_home(tmp_path, monkeypatch, store,
                                             fake_bins, ssh_traffic):
    store()
    home = tmp_path / 'home'
    key = home / '.ssh' / 'id_ecdsa'
    key.parent.mkdir(parents=True)
    key.write_text('ECDSA')
    key.chmod(0o600)
    monkeypatch.setenv('HOME', str(home))
    monkeypatch.setenv('XDG_CONFIG_HOME', str(home / '.config'))
    ssh_traffic['install'](ssh_traffic['ok']())
    ssh_observe.ssh_observe(SESSION_A)
    parts = ssh_traffic['seen'][0]['proxy_parts']
    assert parts[parts.index('--identity') + 1] == str(key)


# ---------- 失败语义：BridgeError、不重试、不停 runtime ----------

def test_ssh_child_does_not_inherit_github_pat(monkeypatch):
    monkeypatch.setenv('MLBRIDGE_GITHUB_TOKEN', 'test-secret')
    seen = []
    class FakeChild:
        pid = 12345
        returncode = 0
        def communicate(self, *, timeout):
            return None, ''
    def fake_popen(argv, **kwargs):
        seen.append(kwargs['env'])
        return FakeChild()
    monkeypatch.setattr(ssh_observe.subprocess, 'Popen', fake_popen)
    assert ssh_observe._run(['ssh', '--help'], timeout=5).returncode == 0
    assert 'MLBRIDGE_GITHUB_TOKEN' not in seen[0]


def test_connectivity_failure_becomes_BridgeError_once(store, fake_bins,
                                                       ssh_traffic):
    store()
    ssh_traffic['install'](ssh_traffic['ok'](255, 'proxy unreachable'))
    with pytest.raises(core.BridgeError, match='观察失败'):
        ssh_observe.ssh_observe(SESSION_A, identity=key_path(fake_bins))
    assert len(ssh_traffic['seen']) == 1
    assert not ssh_traffic['seen'][0]['config'].exists()  # 失败路径同样清理


def test_timeout_bounded_and_cleaned(store, fake_bins, ssh_traffic):
    store()
    ssh_traffic['install'](ssh_traffic['timeout']())
    with pytest.raises(core.BridgeError, match='超时'):
        ssh_observe.ssh_observe(SESSION_A, identity=key_path(fake_bins),
                                timeout=7)
    assert not ssh_traffic['seen'][0]['config'].exists()


def test_timeout_budget_validation(store, fake_bins, ssh_traffic):
    store()
    ssh_traffic['install'](ssh_traffic['ok']())
    with pytest.raises(core.BridgeError, match='timeout'):
        ssh_observe.ssh_observe(SESSION_A, identity=key_path(fake_bins),
                                timeout=0)
    ssh_traffic['seen'].clear()
    ssh_observe.ssh_observe(SESSION_A, identity=key_path(fake_bins),
                            timeout=30)
    assert ssh_traffic['seen'][0]['timeout'] == 30


# ---------- 远端命令：引用与默认快照边界 ----------

def test_custom_command_quoted_as_single_remote_argv(store, fake_bins,
                                                     ssh_traffic):
    store()
    ssh_traffic['install'](ssh_traffic['ok']())
    command = ['sh', '-c', 'echo "hello ; rm -rf $HOME" </dev/null']
    ssh_observe.ssh_observe(SESSION_A, command, identity=key_path(fake_bins))
    argv = ssh_traffic['seen'][0]['argv']
    assert argv.count('--') == 1
    host, remote = argv[argv.index('--') + 1:]
    assert host == 'root@colab-runtime'
    # 整条命令以单个 argv 发出，并能在 POSIX shell 里安全还原。
    assert shlex.split(remote) == command


def test_default_snapshot_is_bounded_and_content_free(store, fake_bins,
                                                      ssh_traffic):
    store()
    ssh_traffic['install'](ssh_traffic['ok']())
    ssh_observe.ssh_observe(SESSION_A, identity=key_path(fake_bins))
    argv = ssh_traffic['seen'][0]['argv']
    remote = argv[argv.index('--') + 2]
    parts = shlex.split(remote)
    assert parts[:2] == ['sh', '-c']
    script = parts[2]
    assert 'nvidia-smi' in script
    assert 'head -n 12' in script            # 进程列表封顶
    assert 'files={count} bytes={size}' in script
    assert 'etime,comm' in script             # 不暴露 argv
    assert 'du -sh' in script                # 工作区/日志只报大小
    assert 'grep -o' in script and '"phase"' in script   # worker 阶段
    for token in ('TOKEN-A-SECRET', 'TOKEN-B-SECRET'):
        assert token not in script           # 无凭据
    # 不打印日志/环境正文
    for forbidden in ('cat /content/mlbridge_run.log', 'printenv',
                      'python -c', 'MLBRIDGE_COLAB_SESSIONS'):
        assert forbidden not in script
