"""`mlbridge checkpoints` 只读发现的离线测试。

fake restic 仓库内容是**预先写死的全量混合仓库**（跨 session / 跨 project /
跨 key / artifact 快照 / 各种畸形），fake 不按"本测试预期"挑挑拣拣：

- `FAKE_RESTIC_FILTER=restic`：fake 只按 restic 0.18.0 help + 实证留证的
  通用过滤语义（多 `--tag` flag 之间 OR、单 flag 内逗号 AND）作用于完整
  混合仓库，不是按测例答案过滤；
- `FAKE_RESTIC_FILTER=none`：fake 完全不过滤，直接吐全量——用于验证
  代码对 backend 返回越界结果是 fail-closed，而不是静默丢弃或当作命中。

不访问网络、不依赖真实 storage；配置走隔离的 XDG_CONFIG_HOME/XDG_STATE_HOME。
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from mlbridge import cli
from mlbridge.checkpoints import discover_checkpoints, key_sha256
from mlbridge.core import BridgeError

PROJECT, SESSION = 'demo', 'sess-x'
SCOPE = ['mlbridge-project-' + PROJECT,
         'mlbridge-session-checkpoint-' + SESSION,
         'mlbridge-checkpoint-v1']
KEY_A = 'row-a'
KEY_B = 'row-b'


def _hex(ch, n=64):
    return ch * n


def _mixed_snapshots():
    """全量混合仓库快照（fixture 一次性写好，与单一测例预期无关）。"""
    def ck(sid, session, key, *, meta=_hex('a'),
           project_tag='mlbridge-project-' + PROJECT, add=()):
        return {'id': _hex(sid), 'time': '2026-10-26T09:00:00Z',
                'tags': [project_tag, 'mlbridge-session-checkpoint-' + session,
                         'mlbridge-checkpoint-v1',
                         'mlbridge-checkpoint-key-' + key_sha256(key),
                         'mlbridge-checkpoint-meta-' + meta, *add]}
    return [
        # 本 session 的正常快照：row-a 一个、row-b 两个（冲突）
        ck('1', SESSION, KEY_A, meta=_hex('a')),
        ck('2', SESSION, KEY_B, meta=_hex('b')),
        ck('3', SESSION, KEY_B, meta=_hex('c')),
        # 其他 session、其他 project 的 checkpoint 快照（污染物）
        ck('4', 'sess-other', 'row-c', meta=_hex('d')),
        ck('5', SESSION, 'row-d', meta=_hex('e'),
           project_tag='mlbridge-project-demo2'),
        # 与 checkpoint 协议无关的 artifact 快照
        {'id': _hex('6'), 'time': '2026-10-26T09:00:00Z',
         'tags': ['mlbridge-project-' + PROJECT, 'mlbridge-artifact-art1']},
    ]


def _malformed_snapshots():
    base = dict(time='2026-10-26T09:00:00Z',
                tags=['mlbridge-project-' + PROJECT,
                      'mlbridge-session-checkpoint-' + SESSION,
                      'mlbridge-checkpoint-v1'])
    return [
        # metadata 指纹缺失
        {**base, 'id': _hex('1', 64),
         'tags': [*(base['tags'] + ['mlbridge-checkpoint-key-' + key_sha256(KEY_A)])]},
        # metadata 指纹重复
        {**base, 'id': _hex('2', 64),
         'tags': [*(base['tags'] + ['mlbridge-checkpoint-key-' + key_sha256(KEY_A),
                                    'mlbridge-checkpoint-meta-' + _hex('a'),
                                    'mlbridge-checkpoint-meta-' + _hex('b')])]},
        # metadata 指纹非 full64
        {**base, 'id': _hex('3', 64),
         'tags': [*(base['tags'] + ['mlbridge-checkpoint-key-' + key_sha256(KEY_A),
                                    'mlbridge-checkpoint-meta-zz'])]},
        # key tag 缺失
        {**base, 'id': _hex('4', 64),
         'tags': [*(base['tags'] + ['mlbridge-checkpoint-meta-' + _hex('a')])]},
        # key tag 重复
        {**base, 'id': _hex('5', 64),
         'tags': [*(base['tags'] + ['mlbridge-checkpoint-key-' + key_sha256(KEY_A),
                                    'mlbridge-checkpoint-key-' + key_sha256(KEY_B),
                                    'mlbridge-checkpoint-meta-' + _hex('a')])]},
        # snapshot id 非 full64
        {**base, 'id': 'abc',
         'tags': [*(base['tags'] + ['mlbridge-checkpoint-key-' + key_sha256(KEY_A),
                                    'mlbridge-checkpoint-meta-' + _hex('a')])]},
    ]


@pytest.fixture
def fake_restic_repo(tmp_path, monkeypatch, request):
    """隔离 XDG 配置 + fake restic（过滤模式由 import 时 fixture 参数决定）。"""
    mode = getattr(request, 'param', 'restic') or 'restic'
    config_base, state = tmp_path / 'cfg', tmp_path / 'state'
    cfg = config_base / 'mlbridge'
    (cfg / 'projects').mkdir(parents=True)
    (cfg / 'projects' / f'{PROJECT}.toml').write_text(
        f'name = "{PROJECT}"\nworkspace = "{tmp_path}/ws"\n'
        'repository = "https://example.com/o/repo"\n')
    (cfg / 'settings.toml').write_text(
        '[storage]\nremote = "gdrive"\nprefix = "store/vm"\n'
        f'password_file = "{tmp_path}/pw"\nrclone_config = "{tmp_path}/rc"\n')
    (tmp_path / 'pw').write_text('not-a-secret')
    (tmp_path / 'rc').write_text('[gdrive]\ntype=drive\n')
    fakebin = tmp_path / 'fakebin'
    fakebin.mkdir()
    (fakebin / 'restic').write_text(_FAKE_RESTIC)
    (fakebin / 'restic').chmod(0o755)
    snapshots = tmp_path / 'snapshots.json'
    snapshots.write_text(json.dumps(
        _malformed_snapshots() if mode == 'malformed' else _mixed_snapshots()))
    log = tmp_path / 'restic.log'
    log.write_text('')
    for name, value in (
        ('XDG_CONFIG_HOME', str(config_base)), ('XDG_STATE_HOME', str(state)),
        ('PATH', str(fakebin) + os.pathsep + os.environ['PATH']),
        ('FAKE_RESTIC_LOG', str(log)),
        ('FAKE_SNAPSHOTS_FILE', str(snapshots)),
        ('FAKE_RESTIC_FILTER', 'restic' if mode != 'unfiltered' else 'none'),
    ):
        monkeypatch.setenv(name, value)
    return {'log': log, 'state': state, 'snapshots': snapshots}


_FAKE_RESTIC = '''#!/usr/bin/env python3
import json, os, sys
args = sys.argv[1:]
with open(os.environ['FAKE_RESTIC_LOG'], 'a', encoding='utf-8') as fh:
    fh.write('restic ' + ' '.join(args) + '\\n')
if 'snapshots' in args:
    snaps = json.load(open(os.environ['FAKE_SNAPSHOTS_FILE'], encoding='utf-8'))
    if os.environ.get('FAKE_RESTIC_FILTER') == 'restic':
        # restic 实证语义：多个 --tag flag 之间 OR，单 flag 内逗号是 AND。
        groups = []
        i = 0
        while i < len(args):
            if args[i] == '--tag':
                groups.append(args[i + 1].split(','))
                i += 2
            else:
                i += 1
        snaps = [s for s in snaps
                 if any(all(t in s['tags'] for t in g) for g in groups)]
    print(json.dumps(snaps))
'''


def _argv_lines(fake_restic_repo):
    return [ln for ln in fake_restic_repo['log'].read_text().splitlines()
            if ' --tag ' in ln]


def test_scope_query_uses_single_comma_joined_and_tag(fake_restic_repo):
    discover_checkpoints(SESSION, PROJECT)
    calls = _argv_lines(fake_restic_repo)
    assert len(calls) == 1
    assert calls[0].split() == [
        'restic', '-r', 'rclone:gdrive:store/vm/demo', 'snapshots', '--json',
        '--tag', ','.join(SCOPE)]


def test_discovery_reports_each_key_and_conflict(fake_restic_repo):
    result = discover_checkpoints(SESSION, PROJECT)
    sha_a, sha_b = key_sha256(KEY_A), key_sha256(KEY_B)
    by_key = {k['key_sha256']: k for k in result['keys']}
    assert set(by_key) == {sha_a, sha_b}
    assert by_key[sha_a]['state'] == 'discovered-unverified'
    assert by_key[sha_b]['state'] == 'conflict'
    assert result['conflicts'] == [sha_b]
    assert [s['snapshot_id'] for s in by_key[sha_a]['snapshots']] == [_hex('1')]
    # metadata 指纹逐快照严格提取
    assert by_key[sha_a]['snapshots'][0]['metadata_sha256'] == _hex('a')
    assert by_key[sha_b]['snapshots'][0]['metadata_sha256'] == _hex('b')
    assert by_key[sha_b]['snapshots'][1]['metadata_sha256'] == _hex('c')
    # 发现不等于验收
    assert all(k['pins_verification'].startswith('NOT_ACCEPTED')
               for k in result['keys'])
    assert '验收' in result['note'] or 'accepted' in result['note'].lower()


@pytest.mark.parametrize('fake_restic_repo', ['restic'], indirect=True)
def test_key_query_exact_match_and_not_found(fake_restic_repo):
    got = discover_checkpoints(SESSION, PROJECT, key=KEY_A)
    assert got['queried'] == {'checkpoint_key': KEY_A,
                              'key_sha256': key_sha256(KEY_A),
                              'found': True, 'state': 'discovered-unverified'}
    assert [s['snapshot_id'] for s in got['keys'][0]['snapshots']] == [_hex('1')]
    # --key 限定后 keys 只含该 key
    assert all(k['key_sha256'] == key_sha256(KEY_A) for k in got['keys'])
    # 查询的 key 不存在时绝不猜测
    missing = discover_checkpoints(SESSION, PROJECT, key='row-zz')
    assert missing['queried']['found'] is False
    assert missing['queried']['state'] == 'not-found'
    assert missing['keys'] == []
    assert missing['queried']['key_sha256'] == key_sha256('row-zz')


@pytest.mark.parametrize('fake_restic_repo', ['unfiltered'], indirect=True)
def test_unfiltered_backend_never_claims_found_for_other_key(fake_restic_repo):
    """backend 漏回无关 key 的快照：fail-closed，而不是误报 found。"""
    with pytest.raises(BridgeError, match='key 归属与查询不符'):
        discover_checkpoints(SESSION, PROJECT, key=KEY_A)


@pytest.mark.parametrize('fake_restic_repo', ['unfiltered'], indirect=True)
def test_unfiltered_backend_pollution_is_fail_closed(fake_restic_repo):
    """backend 返回 scope 外快照（其他 session）：严格报错，不静默丢弃。"""
    with pytest.raises(BridgeError, match='ownership tag'):
        discover_checkpoints(SESSION, PROJECT)


@pytest.mark.parametrize('fake_restic_repo', ['malformed'], indirect=True)
@pytest.mark.parametrize('which, needle', [
    (0, 'metadata 指纹'), (1, 'metadata 指纹'), (2, 'metadata 指纹'),
    (3, 'key tag'), (4, 'key tag'), (5, 'snapshot id'),
])
def test_malformed_in_scope_snapshot_fails_closed(
        fake_restic_repo, monkeypatch, which, needle):
    """fixture 仅保留第 which 条畸形快照，逐一验证 fail-closed 拒绝。"""
    keep = [json.loads(fake_restic_repo['snapshots'].read_text())[which]]
    fake_restic_repo['snapshots'].write_text(json.dumps(keep))
    with pytest.raises(BridgeError, match=needle):
        discover_checkpoints(SESSION, PROJECT)
    assert len(_argv_lines(fake_restic_repo)) == 1  # 恰好一次查询，无重试


@pytest.mark.parametrize('fake_restic_repo', ['restic'], indirect=True)
def test_cli_checkpoints_conflict_exit_and_json(fake_restic_repo, capsys):
    # 混合仓库内 row-b 冲突：退出码 3
    args = cli.parser().parse_args(['checkpoints', SESSION, '--project', PROJECT])
    assert cli.execute(args) == 3
    out = capsys.readouterr().out
    assert 'conflict' in out
    # --json 输出机器可直接消费的结构
    assert cli.execute(cli.parser().parse_args(
        ['checkpoints', SESSION, '--project', PROJECT, '--key', KEY_A,
         '--json'])) == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed['queried']['found'] is True
    assert parsed['queried']['key_sha256'] == key_sha256(KEY_A)
    assert parsed['conflicts'] == []


def test_invalid_session_or_project_fail_fast(fake_restic_repo):
    with pytest.raises(BridgeError):
        discover_checkpoints('../escape', PROJECT)
    with pytest.raises(BridgeError):
        discover_checkpoints(SESSION, 'no-such-project')
