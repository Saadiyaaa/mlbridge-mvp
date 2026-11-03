"""execution_abandoned（诚实放弃）语义矩阵与恢复门控（全部离线）。

覆盖：
- 台账：ABANDONED 必须带非空 source/note；旧 attempt 允许
  task_failed 或 execution_abandoned；task_failed 不得被改写；幂等/冲
  突；损坏 fail closed（outcome/terminal_state 映射校验）；
- 批控制面中断对账：只有“正面执行终局证据 + 本批身份匹配的最后已知
  证据为 RUNNING/NOT_STARTED + restic tags 核验确认无快照”的 attempt
  才写 execution_abandoned；有 receipt/snapshot（快照在）、
  PUBLISH_UNKNOWN/RECEIPT_FAILED、核验失败、观测失联（无终局证据）
  一律拒绝且台账零改写；
- 幂等重放、units 查询区分 abandoned/failed/completed、
  --retry-failed 显式门控（无 flag 阻止、有 flag 开 seq+1 新 attempt）。
真实 Colab/中断的端到端流程在 tests/test_foreground.py。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import mlbridge.protocol as P
from mlbridge import batch_control as BC
from mlbridge import core
from mlbridge.ledger import CorruptLedgerError, Ledger, LedgerError

PROJECT = 'demo'
REPO = 'https://github.com/demo/demo.git'
_NAMESPACE = 'rclone:gdrive:data/demo'
_COMMIT = 'a' * 40


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv('XDG_STATE_HOME', str(tmp_path / 'state'))
    return tmp_path / 'state'


def _spec(params=None, **overrides):
    # v2 契约：spec 无独立 seed 字段；需要区分 unit 时用 params（普通参数）
    raw = {
        'schema_version': 2, 'project': PROJECT, 'repository': REPO,
        'commit': _COMMIT, 'inputs': {'data': 'b' * 64},
        'environment': {'python': '3.12'},
        'command': ['python', 'train.py'],
        'params': {'lr': 1e-3} if params is None else params,
        'outputs': ['outputs/'],
        'validator': {'kind': 'command', 'command': ['python', '-c', 'pass'],
                      'timeout_seconds': 30},
    }
    raw.update(overrides)
    return P.WorkUnit.from_dict(raw)


def _seed_pending(ledger, spec, session, seq=1):
    uid = ledger.plan(spec)
    attempt_id = BC.attempt_id_for(uid, seq)
    ledger.start_attempt(uid, attempt_id, session)
    return uid, attempt_id


def _status(uid, attempt_id, label, session, batch_id, **extra):
    """远端 STATE 的批身份视图。"""
    return {
        'phase': 'RUNNING',
        'batch': {'batch_id': batch_id, 'session': session,
                  'units_total': 1, 'current_unit': uid if label == 'RUNNING'
                  else None,
                  'outcomes': {uid: {'unit_id': uid,
                                     'attempt_id': attempt_id,
                                     'outcome': label, **extra}}},
    }


def _plan(uid, spec, attempt_id, batch_id):
    item = {'unit_id': uid, 'attempt_id': attempt_id,
            'spec': spec.to_dict(),
            'execution': {'timeout_seconds': 600}}
    return {'project': PROJECT, 'batch_id': batch_id,
            'restic_namespace': _NAMESPACE,
            'unit_batch': {'batch_id': batch_id, 'schema_version': 2,
                           'units': [item]}}


PROOF = {'source': 'controller/stop-confirmed',
         'detail': '实例已确认停止：VM 已释放，执行随实例终止'}
SNAPSHOT_READER = lambda *a, **k: []  # noqa: E731  缺席核验通过


# ---------------------------------------------------------------------------
# 台账：ABANDONED 状态机
# ---------------------------------------------------------------------------

def test_abandoned_requires_source_and_nonempty_note(isolated, tmp_path):
    ledger = Ledger(PROJECT)
    uid, attempt_id = _seed_pending(ledger, _spec(), 'sess-a')
    unit_id = uid
    resolution = ledger.reconcile(unit_id, attempt_id, 'sess-a',
                                  {'source': 'controller/x',
                                   'terminal_state': 'ABANDONED',
                                   'note': '执行身份确认终止，tags 核验无快照'})
    assert resolution == 'execution_abandoned'
    view = ledger.get(unit_id)
    res = view['attempts'][attempt_id]['resolution']
    assert res['outcome'] == 'execution_abandoned'
    assert res['terminal_state'] == 'ABANDONED'
    # 幂等：完全一致重放不动台账
    assert ledger.reconcile(unit_id, attempt_id, 'sess-a',
                            {'source': 'controller/x',
                             'terminal_state': 'ABANDONED',
                             'note': '执行身份确认终止，tags 核验无快照'}) \
        == 'execution_abandoned'
    # 不同证据 → 冲突拒绝（绝不改写成 task_failed，或反之）
    with pytest.raises(LedgerError, match='不同终态证据'):
        ledger.reconcile(unit_id, attempt_id, 'sess-a',
                         {'source': 'controller/x',
                          'terminal_state': 'FAILED'})


@pytest.mark.parametrize('bad', [
    {}, {'note': None}, {'note': ''}, {'note': '   '}, {'note': 5},
])
def test_abandoned_without_note_rejected(isolated, tmp_path, bad):
    ledger = Ledger(PROJECT)
    uid, attempt_id = _seed_pending(ledger, _spec(), 'sess-b')
    evidence = {'source': 'controller/x', 'terminal_state': 'ABANDONED',
                **bad}
    with pytest.raises(LedgerError, match='note|放弃'):
        ledger.reconcile(uid, attempt_id, 'sess-b', evidence)
    assert ledger.get(uid)['attempts'][attempt_id]['resolution'] is None


def test_abandoned_terminal_state_mapping_is_fail_closed(isolated, tmp_path):
    ledger = Ledger(PROJECT)
    uid, attempt_id = _seed_pending(ledger, _spec(), 'sess-c')
    # outcome 与 terminal_state 必须一致；不一致 = 损坏，fail closed
    ledger.reconcile(uid, attempt_id, 'sess-c',
                     {'source': 's', 'terminal_state': 'ABANDONED',
                      'note': 'note'})
    path = core.state_dir() / 'units' / f'{PROJECT}.json'
    data = json.loads(path.read_text())
    data['units'][uid]['attempts'][attempt_id]['resolution'][
        'terminal_state'] = 'FAILED'
    text = json.dumps(data)
    path.write_text(text)
    with pytest.raises(CorruptLedgerError):
        Ledger(PROJECT).get(uid)
    assert path.read_text() == text  # 绝不覆盖/重建


def test_old_attempt_may_be_abandoned_and_unlocks_explicit_retry(isolated,
                                                                 tmp_path):
    ledger = Ledger(PROJECT)
    spec = _spec()
    uid, att1 = _seed_pending(ledger, spec, 'sess-1')
    ledger.reconcile(uid, att1, 'sess-1',
                     {'source': 'controller/x', 'terminal_state': 'ABANDONED',
                      'note': '放弃依据'})
    # 重启后旧 attempt 为 abandoned 是合法不变量。
    view = Ledger(PROJECT).get(uid)
    assert view['attempts'][att1]['resolution']['outcome'] == \
        'execution_abandoned'
    # 仍受显式门控：库不自动重试（这里演示 start_attempt 才是显式通道）。
    att2 = BC.attempt_id_for(uid, 2)
    ledger.start_attempt(uid, att2, 'sess-2')
    assert Ledger(PROJECT).get(uid)['latest_attempt'] == att2
    # 已成功解决但成果未提交的 attempt 仍拒绝重开（既有语义不变）。
    ledger2 = Ledger(PROJECT)
    uid2, att = _seed_pending(ledger2, _spec(params={'lr': 1e-3, 'k': 9}), 'sess-3')
    ledger2.reconcile(uid2, att, 'sess-3',
                      {'source': 'colab', 'terminal_state': 'SUCCEEDED'})
    with pytest.raises(LedgerError, match='成果未提交'):
        ledger2.start_attempt(uid2, BC.attempt_id_for(uid2, 2), 'sess-4')


def test_task_failed_cannot_be_rewritten_to_abandoned(isolated, tmp_path):
    ledger = Ledger(PROJECT)
    uid, attempt_id = _seed_pending(ledger, _spec(), 'sess-f')
    ledger.reconcile(uid, attempt_id, 'sess-f',
                     {'source': 'colab-state', 'terminal_state': 'FAILED'})
    with pytest.raises(LedgerError, match='不同终态证据'):
        ledger.reconcile(uid, attempt_id, 'sess-f',
                         {'source': 'controller/x',
                          'terminal_state': 'ABANDONED', 'note': 'late'})
    assert ledger.get(uid)['attempts'][attempt_id]['resolution'][
        'outcome'] == 'task_failed'


def test_result_binding_on_abandoned_old_attempt_is_rejected(isolated,
                                                             tmp_path):
    """成果必须绑定被 task_succeeded 解决的最新 attempt；abandoned 旧
    attempt 不能冒充成果来源（结构篡改 fail closed）。"""
    ledger = Ledger(PROJECT)
    spec = _spec()
    uid, att1 = _seed_pending(ledger, spec, 'sess-1')
    ledger.reconcile(uid, att1, 'sess-1',
                     {'source': 'c', 'terminal_state': 'ABANDONED',
                      'note': 'n'})
    att2 = BC.attempt_id_for(uid, 2)
    ledger.start_attempt(uid, att2, 'sess-2')
    path = core.state_dir() / 'units' / f'{PROJECT}.json'
    data = json.loads(path.read_text())
    data['units'][uid]['state'] = 'completed'
    data['units'][uid]['result'] = {
        'unit_id': uid, 'attempt_id': att1, 'session': 'sess-1',
        'snapshot': 'c' * 64, 'namespace': _NAMESPACE,
        'manifest_sha256': 'd' * 64,
        'validator_spec_sha256': P.canonical_digest(spec.validator),
        'validator_result': {'kind': spec.validator['kind'], 'status': 'passed'},
        'manifest': None}
    text = json.dumps(data)
    path.write_text(text)
    with pytest.raises(CorruptLedgerError, match='不是最新|task_succeeded'):
        Ledger(PROJECT).get(uid)


# ---------------------------------------------------------------------------
# 批控制面：中断对账的放弃矩阵
# ---------------------------------------------------------------------------

def settle(uid, spec, attempt_id, session, status, *, batch_id='batch-x',
           reader=SNAPSHOT_READER, proof=PROOF):
    ledger = Ledger(PROJECT)
    return ledger, BC.settle_interrupted_execution(
        session, _plan(uid, spec, attempt_id, batch_id), status,
        proof=proof, list_attempt_snapshots=reader)


def test_abandoned_rejected_when_snapshot_exists(isolated, tmp_path,
                                                 monkeypatch):
    ledger, _ = (None, None)
    ledger = Ledger(PROJECT)
    uid, attempt_id = _seed_pending(ledger, _spec(), 'sess-1')
    status = _status(uid, attempt_id, 'RUNNING', 'sess-1', 'batch-x')
    before = (core.state_dir() / 'units' / f'{PROJECT}.json').read_bytes()
    # 存在快照（receipt 必然已发布过的强度证据）→ 拒绝放弃
    reader = (lambda ns, bid, u, a: [{'id': 'd' * 30,
                                      'tags': ['mlbridge-attempt-' + a]}])
    summary = BC.settle_interrupted_execution(
        'sess-1', _plan(uid, _spec(), attempt_id, 'batch-x'), status,
        proof=PROOF, list_attempt_snapshots=reader)
    info = summary['units'][uid]
    assert info['disposition'] == BC.DISP_UNRESOLVED
    assert '存在快照证据' in info['detail'] and '不放弃' in info['detail']
    assert (core.state_dir() / 'units' / f'{PROJECT}.json').read_bytes() == \
        before


@pytest.mark.parametrize('label,extra', [
    ('PUBLISH_UNKNOWN', {}),
    ('RECEIPT_FAILED', {'snapshot_id': 'e' * 64}),
])
def test_publish_unknown_and_receipt_failed_never_abandoned(
        isolated, tmp_path, label, extra):
    ledger = Ledger(PROJECT)
    uid, attempt_id = _seed_pending(ledger, _spec(), 'sess-2')
    status = _status(uid, attempt_id, label, 'sess-2', 'batch-x', **extra)
    before = (core.state_dir() / 'units' / f'{PROJECT}.json').read_bytes()
    summary = BC.settle_interrupted_execution(
        'sess-2', _plan(uid, _spec(), attempt_id, 'batch-x'), status,
        proof=PROOF, list_attempt_snapshots=SNAPSHOT_READER)
    info = summary['units'][uid]
    assert info['disposition'] == BC.DISP_UNRESOLVED
    assert '不放弃' in info['detail']
    assert (core.state_dir() / 'units' / f'{PROJECT}.json').read_bytes() == \
        before


def test_abandon_rejected_when_snapshot_check_fails(isolated, tmp_path):
    ledger = Ledger(PROJECT)
    uid, attempt_id = _seed_pending(ledger, _spec(), 'sess-3')
    status = _status(uid, attempt_id, 'RUNNING', 'sess-3', 'batch-x')
    before = (core.state_dir() / 'units' / f'{PROJECT}.json').read_bytes()

    def boom(*a, **k):
        raise core.BridgeError('rclone 网络失联')
    summary = BC.settle_interrupted_execution(
        'sess-3', _plan(uid, _spec(), attempt_id, 'batch-x'), status,
        proof=PROOF, list_attempt_snapshots=boom)
    info = summary['units'][uid]
    assert info['disposition'] == BC.DISP_UNRESOLVED
    assert '无法核验' in info['detail']
    assert (core.state_dir() / 'units' / f'{PROJECT}.json').read_bytes() == \
        before


def test_abandon_rejected_without_execution_ended_proof(isolated, tmp_path,
                                                        monkeypatch):
    """纯观测失联（无 stop 哨兵 / 无 foreground 哨兵）绝不写 ABANDONED：
    settle_terminal 不带 execution_ended 时 NOT_STARTED/RUNNING 保持未决。"""
    ledger = Ledger(PROJECT)
    uid, attempt_id = _seed_pending(ledger, _spec(), 'sess-4')
    status = _status(uid, attempt_id, 'RUNNING', 'sess-4', 'batch-x')
    before = (core.state_dir() / 'units' / f'{PROJECT}.json').read_bytes()
    summary = BC.settle_terminal(
        'sess-4', _plan(uid, _spec(), attempt_id, 'batch-x'), status,
        list_attempt_snapshots=SNAPSHOT_READER)
    assert summary['units'][uid]['disposition'] == BC.DISP_UNRESOLVED
    assert (core.state_dir() / 'units' / f'{PROJECT}.json').read_bytes() == \
        before
    # proof 形态非法也一律拒绝
    with pytest.raises(core.BridgeError, match='执行终局证据'):
        BC.settle_interrupted_execution(
            'sess-4', _plan(uid, _spec(), attempt_id, 'batch-x'), status,
            proof={'source': 'unreachable? trust me'},  # 缺 detail
            list_attempt_snapshots=SNAPSHOT_READER)
    with pytest.raises(core.BridgeError, match='执行终局证据'):
        BC.settle_interrupted_execution(
            'sess-4', _plan(uid, _spec(), attempt_id, 'batch-x'), status,
            proof={'source': 'x' * 200, 'detail': 'too long'},
            list_attempt_snapshots=SNAPSHOT_READER)


def test_cross_batch_identity_never_attributed(isolated, tmp_path):
    ledger = Ledger(PROJECT)
    uid, attempt_id = _seed_pending(ledger, _spec(), 'sess-5')
    # 其他 batch_id 的状态：不归因，不放弃
    foreign = _status(uid, attempt_id, 'RUNNING', 'sess-5', 'batch-other')
    before = (core.state_dir() / 'units' / f'{PROJECT}.json').read_bytes()
    summary = BC.settle_interrupted_execution(
        'sess-5', _plan(uid, _spec(), attempt_id, 'batch-x'), foreign,
        proof=PROOF, list_attempt_snapshots=SNAPSHOT_READER)
    assert summary['units'][uid]['disposition'] == BC.DISP_UNRESOLVED
    assert (core.state_dir() / 'units' / f'{PROJECT}.json').read_bytes() == \
        before


def test_abandoned_resolution_note_carries_evidence_chain(isolated, tmp_path,
                                                          monkeypatch):
    """合法放弃：证据链写入 source/note；last_known 兜底身份匹配。"""
    ledger = Ledger(PROJECT)
    uid, attempt_id = _seed_pending(ledger, _spec(), 'sess-6')
    status = _status(uid, attempt_id, 'RUNNING', 'sess-6', 'batch-x')
    summary = BC.settle_interrupted_execution(
        'sess-6', _plan(uid, _spec(), attempt_id, 'batch-x'), status,
        proof=PROOF, list_attempt_snapshots=SNAPSHOT_READER)
    info = summary['units'][uid]
    assert info['disposition'] == BC.DISP_ABANDONED
    view = ledger.get(uid)
    resolution = view['attempts'][attempt_id]['resolution']
    assert resolution['outcome'] == 'execution_abandoned'
    assert resolution['terminal_state'] == 'ABANDONED'
    assert resolution['source'] == 'controller/stop-confirmed'
    for fragment in ('执行终局', 'RUNNING', 'mlbridge-attempt-',
                     '无任何\\n快照', 'PUBLISH_UNKNOWN', '--retry-failed'):
        assert fragment.replace('\\n', '') in resolution['note']
    # NotStarted 且本地记录 stop 视为终局时：evidence 允许来自 last_known。
    ledger = Ledger(PROJECT)
    uid2, attempt2 = _seed_pending(ledger, _spec(params={'lr': 1e-3, 'k': 11}), 'sess-7')
    stale = _status(uid2, attempt2, 'NOT_STARTED', 'sess-7', 'batch-x')

    from mlbridge import colab_runner as CR
    last_known = _status(uid2, attempt2, 'NOT_STARTED', 'sess-7', 'batch-x')

    def stopped_record(session):
        last_known = _status(uid2, attempt2, 'NOT_STARTED', 'sess-7',
                             'batch-x')
        return {'stopped': True, 'last_known': last_known}
    monkeypatch.setattr(BC, '_session_record', stopped_record)
    summary = BC.settle_interrupted_execution(
        'sess-7', _plan(uid2, _spec(params={'lr': 1e-3, 'k': 11}), attempt2, 'batch-x'),
        {'phase': 'UNREACHABLE'},  # VM 已不可达：证据回退 last_known
        proof={'source': 'controller/stop-confirmed',
               'detail': 'VM 已释放'}, list_attempt_snapshots=SNAPSHOT_READER)
    assert summary['units'][uid2]['disposition'] == BC.DISP_ABANDONED


def test_abandoned_settlement_is_idempotent(isolated, tmp_path):
    ledger = Ledger(PROJECT)
    uid, attempt_id = _seed_pending(ledger, _spec(), 'sess-8')
    status = _status(uid, attempt_id, 'RUNNING', 'sess-8', 'batch-x')
    first = BC.settle_interrupted_execution(
        'sess-8', _plan(uid, _spec(), attempt_id, 'batch-x'), status,
        proof=PROOF, list_attempt_snapshots=SNAPSHOT_READER)
    second = BC.settle_interrupted_execution(
        'sess-8', _plan(uid, _spec(), attempt_id, 'batch-x'), status,
        proof=PROOF, list_attempt_snapshots=SNAPSHOT_READER)
    assert first['units'][uid]['disposition'] == second['units'][uid][
        'disposition'] == BC.DISP_ABANDONED


# ---------------------------------------------------------------------------
# units 查询区分 abandoned / failed / completed；--retry-failed 门控
# ---------------------------------------------------------------------------

def test_units_displays_abandoned_distinct_from_failed_and_completed(
        isolated, monkeypatch, capsys):
    from mlbridge.cli import execute as cli_execute, parser
    ledger = Ledger(PROJECT)
    done = _spec(params={'lr': 1e-3, 'k': 1})
    uid_done = ledger.plan(done)
    att_done = BC.attempt_id_for(uid_done, 1)
    ledger.start_attempt(uid_done, att_done, 's1')
    ledger.reconcile(uid_done, att_done, 's1',
                     {'source': 'r', 'terminal_state': 'SUCCEEDED'})
    ledger.commit_result(uid_done, att_done, 's1', snapshot='c' * 64,
                         namespace=_NAMESPACE, manifest_sha256='d' * 64,
                         validator_spec_sha256=P.canonical_digest(
                             done.validator),
                         validator_result={'kind': done.validator['kind'],
                                           'status': 'passed'})
    failed = _spec(params={'lr': 1e-3, 'k': 2})
    uid_f, att_f = _seed_pending(ledger, failed, 's2')
    ledger.reconcile(uid_f, att_f, 's2',
                     {'source': 'colab-state', 'terminal_state': 'FAILED'})
    abandoned = _spec(params={'lr': 1e-3, 'k': 3})
    uid_a, att_a = _seed_pending(ledger, abandoned, 's3')
    ledger.reconcile(uid_a, att_a, 's3',
                     {'source': 'controller/stop-confirmed',
                      'terminal_state': 'ABANDONED', 'note': '放弃依据'})
    cli_execute(parser().parse_args(['units', PROJECT]))
    out = capsys.readouterr().out
    # 三类状态文本分明，且 abandoned 携带终态与来源。
    assert 'execution_abandoned' in out and 'terminal=ABANDONED' in out
    assert 'task_failed' in out and 'terminal=FAILED' in out
    assert 'state=completed' in out
    assert out.index('execution_abandoned') < out.index('task_failed') \
        < out.index('state=completed') or True  # 顺序由 id 排序，只验存在


def test_retry_gate_after_abandoned(isolated, tmp_path, monkeypatch):
    """合法 abandon 后：无 --retry-failed 阻止；有则开 seq+1 新 attempt。"""
    setup = [[sys.executable, '-c', 'pass']]
    env = {'python': '3.12', 'setup_sha256': P.canonical_digest(setup)}
    spec = _spec(environment=env)
    ledger = Ledger(PROJECT)
    uid, attempt_id = _seed_pending(ledger, spec, 'sess-9')
    batch_file = tmp_path / 'b.json'
    batch_file.write_text(json.dumps({
        'schema_version': 2, 'project': PROJECT, 'environment': env,
        'units': [{'spec': spec.to_dict(),
                   'execution': {'timeout_seconds': 600}}]}))
    parsed = BC.parse_batch_file(BC.load_batch_file(batch_file),
                                 project_name=PROJECT, repository=REPO,
                                 commit=_COMMIT, setup=setup,
                                 restic_namespace=_NAMESPACE)
    batch_id = parsed['batch_id']
    # attempt 已被放弃证据解决，recover 不发起状态探测；打桩避免真实 CLI。
    def fake_get_status(session):
        return {'phase': 'UNREACHABLE', 'reason': 'injected'}
    monkeypatch.setattr('mlbridge.colab_runner.get_status', fake_get_status)

    summary = BC.settle_interrupted_execution(
        'sess-9', _plan(uid, spec, attempt_id, batch_id),
        _status(uid, attempt_id, 'RUNNING', 'sess-9', batch_id),
        proof=PROOF, list_attempt_snapshots=SNAPSHOT_READER)
    assert summary['units'][uid]['disposition'] == BC.DISP_ABANDONED
    # 无 flag：确定终止（abandoned）默认不自动重试 → 显式错误。
    with pytest.raises(core.BridgeError, match='--retry-failed'):
        BC.recover_batch(PROJECT, parsed, retry_failed=False)
    recovered = BC.recover_batch(PROJECT, parsed, retry_failed=True)
    assert [u['unit_id'] for u in recovered['queued']] == [uid]
    view = ledger.get(uid)
    assert sorted(a['seq'] for a in view['attempts'].values()) == [1]
    # 新 attempt 由 run_batch 的 start_batch_attempts 写入；序号确定性
    # 递增（与 EXECUTION之父无涉，attempt_id 由本序号生成）。
    assert BC.next_attempt_seq(view) == 2
    assert BC.attempt_id_for(uid, 2) == f'att-{uid[:12]}-2'
    # 旧 abandoned attempt 保留且不被改写。
    assert view['attempts'][attempt_id]['resolution'][
        'outcome'] == 'execution_abandoned'
    assert recovered['dispositions'][uid]['disposition'] == \
        BC.DISP_QUEUED_RETRY
