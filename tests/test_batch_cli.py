"""批模式本地控制面（opt-in `run --unit-batch`）的离线全路径测试。

不联网、不 GPU、不真实 colab/restic：`colab_runner.cli` 全量打桩（fake
是纯 stdlib 的 colab CLI 替身：new/exec/upload/download/stop 均只记录
调用；`download` 从预设远端文件表回放），`colab_runner.stage` 只记录
上传集合。覆盖 legacy run 不变、分配前拒绝非法批次、恢复对账/重试
策略/未决语义/units 查询/批终态 watch 对账、对齐窗口自愈（含
KeyboardInterrupt/stop 未确认/重跑需 --retry-failed）与崩溃窗口
（reconcile 后 commit 前中断）补提交自愈。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import mlbridge.protocol as P
from mlbridge import batch_worker as BW
from mlbridge import colab_runner, core
from mlbridge import batch_control as BC
from mlbridge import cli as cli_mod
from mlbridge.ledger import Ledger

_REPO = 'https://github.com/demo/demo.git'
_NAMESPACE = 'rclone:gdrive:data/demo'


# ---------------------------------------------------------------------------
# 基础设施（与 test_colab 同风格：隔离 XDG、fake colab CLI）
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv('XDG_STATE_HOME', str(tmp_path / 'state'))
    monkeypatch.delenv('MLBRIDGE_GITHUB_USERNAME', raising=False)
    monkeypatch.delenv('MLBRIDGE_GITHUB_TOKEN', raising=False)
    monkeypatch.setattr(colab_runner, '_github_secrets_path',
                        lambda: tmp_path / 'isolated-secrets.env')
    # 单实例守卫默认打桩为空（守卫本体由 test_colab 覆盖）。
    monkeypatch.setattr(colab_runner, 'list_active_sessions', lambda: [])
    monkeypatch.setattr(colab_runner.time, 'sleep', lambda s: None)
    # 安全网：未经过 _install 的测试绝不允许触碰真实 colab CLI。
    def _no_real_colab(*args, **kwargs):
        raise AssertionError(f'测试未打桩而触到真实 colab 调用: {args[:2]}')
    monkeypatch.setattr(colab_runner, 'cli', _no_real_colab)
    # 同样的安全网针对 core.proc：只放行本地 git 元数据查询（不联网），
    # 未来任何 restic/rclone/colab 等网络类真子进程调用都会炸测试。
    real_proc = core.proc

    def _guarded_proc(cmd, *args, **kwargs):
        if cmd and cmd[0] == 'git':
            return real_proc(cmd, *args, **kwargs)
        raise AssertionError(
            '测试守卫：core.proc 只允许本地 git；实际请求 '
            f'{cmd[:3]}（网络类命令必须打桩）')
    monkeypatch.setattr(core, 'proc', _guarded_proc)


def _prepare_project(tmp_path, monkeypatch):
    """独立 XDG 配置 + 已 push 的 git 工作区（真实提交，无网络）。"""
    root = tmp_path / 'proj'
    root.mkdir()
    subprocess.run(['git', 'init', '-q', str(root)], check=True)
    subprocess.run(['git', '-C', str(root), 'config', 'user.name', 'T'], check=True)
    subprocess.run(['git', '-C', str(root), 'config', 'user.email',
                    't@example.com'], check=True)
    (root / 'train.py').write_text('print("train")\n')
    (root / 'requirements.txt').write_text('numpy==1\n')
    subprocess.run(['git', '-C', str(root), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(root), 'commit', '-qm', 'init'], check=True)
    branch = subprocess.check_output(
        ['git', '-C', str(root), 'branch', '--show-current'], text=True).strip()
    commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                                     text=True).strip()
    subprocess.run(['git', '-C', str(root), 'remote', 'add', 'origin', _REPO],
                   check=True)
    subprocess.run(['git', '-C', str(root), 'update-ref',
                    f'refs/remotes/origin/{branch}', commit], check=True)
    subprocess.run(['git', '-C', str(root), 'branch', '--set-upstream-to',
                    f'origin/{branch}'], check=True, capture_output=True)
    cfg = tmp_path / 'cfg'
    (cfg / 'mlbridge').mkdir(parents=True)
    monkeypatch.setenv('XDG_CONFIG_HOME', str(cfg))
    scratch = tmp_path / 'creds'
    scratch.mkdir()
    rconf = scratch / 'rclone.conf'
    passwd = scratch / 'passwd'
    rconf.write_text('[gdrive]\ntype=drive\n')
    passwd.write_text('topsecret')
    (cfg / 'mlbridge' / 'settings.toml').write_text(
        '[storage]\nremote="gdrive"\nprefix="data"\n'
        f'password_file="{passwd}"\nrclone_config="{rconf}"\n\n'
        '[colab]\ndefault_gpu="CPU"\n')
    core.register('demo', str(root), _REPO)
    setup = [[sys.executable, '-c', "import pathlib\n"
             "pathlib.Path('SETUP.log').open('a').write('x')"]]
    # 覆写 register 的默认 job setup，保证批文件注入指纹与 CLI 提取一致。
    (core.project_file('demo')).write_text(
        'name = "demo"\n'
        f'workspace = {json.dumps(str(root))}\n'
        f'repository = {json.dumps(_REPO)}\n'
        '\n[jobs.example]\n'
        'command = ["python", "train.py"]\n'
        f'setup = {json.dumps(setup)}\n'
        'outputs = ["outputs"]\n')
    return {'workspace': root, 'commit': commit, 'setup': setup}


LEGACY_STAGE_SET = ['mlbridge_worker.py', 'mlbridge_job.json',
                    'mlbridge_rclone.conf', 'mlbridge_restic_password']


class FakeColab:
    """colab CLI 全量打桩：所有调用只记录；`download` 从远端文件表回放。"""

    def __init__(self):
        self.calls: list[tuple] = []
        self.remote: dict[tuple[str, str], dict] = {}
        self.staged: list[str] = []
        self.job_payloads: list[dict] = []

    def cli(self, *args, capture=False, timeout=None):
        self.calls.append(args)
        if args[0] == 'download':
            session, remote_path, dest = args[2], args[3], Path(args[4])
            payload = self.remote.get((session, remote_path))
            if payload is None:
                raise core.BridgeError(f'download failed: {remote_path}')
            dest.write_text(json.dumps(payload, ensure_ascii=False))
        return subprocess.CompletedProcess(args, 0)

    def set_state(self, session, state):
        self.remote[(session, '/content/mlbridge_state.json')] = state

    def receipt(self, session, batch_id, uid, payload):
        self.remote[(session,
                     f'/content/mlbridge_receipts/{batch_id}/{uid}.json')] = payload

    def count(self, verb):
        return sum(1 for c in self.calls if c[0] == verb)

    def new_sessions(self):
        return [c[2] for c in self.calls if c[0] == 'new']


def _install(monkeypatch, fake):
    monkeypatch.setattr(colab_runner, 'cli', fake.cli)

    def shutdown(session):  # 与真实 shutdown 等价：cli 成功后标 stopped
        fake.cli('stopped', session)
        record = colab_runner.get_local(session)
        record['stopped'] = True
        colab_runner.store_local(session, record)
    monkeypatch.setattr(colab_runner, 'shutdown', shutdown)

    def stage(session, src, name):
        fake.staged.append(name)
        if name == 'mlbridge_job.json':
            fake.job_payloads.append(json.loads(Path(src).read_text()))
    monkeypatch.setattr(colab_runner, 'stage', stage)


def _run(monkeypatch, fake, argv):
    return cli_mod.execute(cli_mod.parser().parse_args(argv))


def _argv(path, *extra):
    return ['run', 'demo', '--unit-batch', str(path), *extra]


# ---------------------------------------------------------------------------
# 批定义构造
# ---------------------------------------------------------------------------

def _spec(command=None, outputs=None, params=None, validator=None,
          inputs=None, project=None, commit=None, repository=None,
          environment=None):
    """批文件中的 spec：可缺省 project/repository/commit/environment。
    当前契约：spec 的字段集固定；额外字段一律按未知字段拒绝。"""
    raw = {
        'inputs': inputs if inputs is not None else {'data': 'a' * 64},
        'command': command if command is not None else ['python', 'train.py'],
        'params': {'lr': 1e-3} if params is None else params,
        'outputs': outputs if outputs is not None else ['outputs/'],
        'validator': validator if validator is not None else
        {'kind': 'command', 'command': ['python', '-c', 'pass'],
         'timeout_seconds': 30},
    }
    if project:
        raw['project'] = project
    if commit:
        raw['commit'] = commit
    if repository:
        raw['repository'] = repository
    if environment:
        raw['environment'] = environment
    return raw


def _batch_file(units_raw, *, path, project='demo', environment=None,
                batch_id=None, timeout=600, limits=None, publish=None,
                raw_text=None):
    payload = {
        'schema_version': 2,
        'project': project,
        'environment': environment if environment is not None
        else {'python': '3.12'},
        'units': [{'spec': raw, 'execution': {'timeout_seconds': timeout}}
                  for raw in units_raw],
    }
    if batch_id is not None:
        payload['unit_batch_id'] = batch_id
    if limits is not None:
        payload['manifest_limits'] = limits
    if publish is not None:
        payload['publish_timeout_seconds'] = publish
    target = path if path is not None else Path('batch.json')
    target.write_text(raw_text if raw_text is not None
                      else json.dumps(payload, ensure_ascii=False, indent=2))
    return target


def _full_unit(raw, ctx):
    """与 parse_batch_file 的注入完全一致的 WorkUnit（预置台账用）。"""
    env = {'python': '3.12', 'setup_sha256': P.canonical_digest(ctx['setup'])}
    return P.WorkUnit.from_dict({**raw, 'schema_version': 2, 'project': 'demo',
                                 'repository': _REPO,
                                 'commit': ctx['commit'],
                                 'environment': env})


def _receipt(ctx, unit, uid, attempt_id, session, batch_id, snapshot='e' * 64):
    validator = unit.validator
    return {
        # unit RECEIPT format is versioned independently of the unit spec
        'schema_version': BW.RECEIPT_SCHEMA_VERSION,
        'batch_id': batch_id, 'unit_id': uid,
        'attempt_id': attempt_id, 'session': session,
        'namespace': _NAMESPACE,
        'snapshot_id': snapshot, 'manifest_sha256': 'd' * 64,
        'manifest_path': f'.mlbridge-metadata/{batch_id}/{uid}/manifest.json',
        'validator_spec_sha256': P.canonical_digest(validator),
        'validator_result': {'kind': validator['kind'], 'status': 'passed'},
        'phase': 'published', 'finished': '2026-10-26T00:00:00+00:00',
    }


def _outcome_state(session, batch_id, outcomes):
    """模拟远端 /content/mlbridge_state.json 的批模式视图。"""
    phase = 'SUCCEEDED' if all(o.get('outcome') == 'SUCCEEDED'
                               for o in outcomes.values()) else 'FAILED'
    return {'phase': phase, 'session': session, 'job_id': batch_id,
            'batch': {'batch_id': batch_id, 'session': session,
                      'units_total': len(outcomes), 'current_unit': None,
                      'outcomes': outcomes}}


def _outcome(uid, attempt_id, kind, **extra):
    return {'unit_id': uid, 'attempt_id': attempt_id, 'outcome': kind,
            **extra}


def _preseed_pending(ctx, raw, session, seq=1):
    """在台账中造 pending unit + 未决 attempt（模拟上次中断）。"""
    unit = _full_unit(raw, ctx)
    ledger = Ledger('demo')
    uid = ledger.plan(unit)
    attempt_id = BC.attempt_id_for(uid, seq)
    ledger.start_attempt(uid, attempt_id, session)
    return unit, ledger, uid, attempt_id


def _preseed_failed(ctx, raw, session, seq=1, detail='saw it die'):
    unit = _full_unit(raw, ctx)
    ledger = Ledger('demo')
    uid = ledger.plan(unit)
    attempt_id = BC.attempt_id_for(uid, seq)
    ledger.start_attempt(uid, attempt_id, session)
    ledger.reconcile(uid, attempt_id, session,
                     {'source': f'colab-state/{session}',
                      'terminal_state': 'FAILED', 'note': detail})
    return unit, ledger, uid, attempt_id


def _commit_unit(unit, ledger, uid, attempt_id, session, snapshot='c' * 64):
    ledger.reconcile(uid, attempt_id, session,
                     {'source': f'colab-receipt/{session}',
                      'terminal_state': 'SUCCEEDED'})
    ledger.commit_result(uid, attempt_id, session, snapshot=snapshot,
                         namespace=_NAMESPACE, manifest_sha256='d' * 64,
                         validator_spec_sha256=P.canonical_digest(unit.validator),
                         validator_result={'kind': unit.validator['kind'],
                                           'status': 'passed'})


def _ledger_file(tmp_path):
    return core.state_dir() / 'units' / 'demo.json'


# ---------------------------------------------------------------------------
# 批次身份：内容派生、文件名无关、显式值必须等于派生值（GPU 分配前拒绝）
# ---------------------------------------------------------------------------

def test_batch_id_content_derived_and_filename_agnostic(tmp_path, monkeypatch):
    """同一批定义以任意文件名 → 同一派生 batch_id；文件名绝不参与身份。"""
    ctx = _prepare_project(tmp_path, monkeypatch)
    raw = _spec()
    a = BC.parse_batch_file(BC.load_batch_file(
        _batch_file([raw], path=tmp_path / 'a.json')),
        project_name='demo', repository=_REPO, commit=ctx['commit'],
        setup=ctx['setup'], restic_namespace=_NAMESPACE)
    b = BC.parse_batch_file(BC.load_batch_file(
        _batch_file([raw], path=tmp_path / 'b.json')),
        project_name='demo', repository=_REPO, commit=ctx['commit'],
        setup=ctx['setup'], restic_namespace=_NAMESPACE)
    assert a['batch_id'] == b['batch_id']
    assert a['batch_id'].startswith('batch-') and len(a['batch_id']) <= 63


def test_batch_id_changes_with_definition_and_explicit_must_match(
        tmp_path, monkeypatch):
    ctx = _prepare_project(tmp_path, monkeypatch)
    fake = FakeColab()
    _install(monkeypatch, fake)
    assert _run(monkeypatch, fake,
                _argv(_batch_file([_spec(params={'lr': 1e-3, 'v': 1})], path=tmp_path / 'a.json'),
                      '--detach')) == 0
    assert _run(monkeypatch, fake,
                _argv(_batch_file([_spec(params={'lr': 1e-3, 'v': 2})], path=tmp_path / 'a.json'),
                      '--detach')) == 0
    ids = [payload['unit_batch']['batch_id'] for payload in fake.job_payloads]
    assert ids[0] != ids[1]
    # 同文件名不同内容身份不同（身份不锚定文件名）。显式 wrong id：
    bad = _batch_file([_spec(params={'lr': 1e-3, 'v': 3})], batch_id='not-the-id',
                      path=tmp_path / 'bad-id.json')
    n0 = len(fake.calls)
    with pytest.raises(core.BridgeError, match='批次身份|不一致'):
        _run(monkeypatch, fake, _argv(bad, '--detach'))
    assert len(fake.calls) == n0  # 分配前拒绝：无任何 colab 调用


# ---------------------------------------------------------------------------
# 非法批次：GPU 分配前 fail closed（无任何 colab 调用）
# ---------------------------------------------------------------------------

def test_invalid_batches_rejected_before_allocation(tmp_path, monkeypatch):
    ctx = _prepare_project(tmp_path, monkeypatch)

    # (name, 批文件构造, 预期消息片段)
    spec = _spec()
    cases = []
    cases.append(('missing_file', 'no-such.json', None, '无法读取批定义文件'))
    cases.append(('nan_literal', _batch_file(
        [], path=tmp_path / 'nan.json',
        raw_text=json.dumps({
            'schema_version': 2, 'project': 'demo',
            'environment': {'python': '3.12'},
            'units': [{'spec': spec, 'execution': {'timeout_seconds': 600}}],
        }).replace('600', 'NaN')), None, '非法 JSON|NaN'))
    # unknown spec schema_version inside an otherwise valid batch file
    unknown_ver = {
        'schema_version': 2, 'project': 'demo',
        'environment': {'python': '3.12'},
        'units': [{'spec': dict(spec, schema_version=1),
                   'execution': {'timeout_seconds': 600}}],
    }
    cases.append(('unknown_spec_schema_version', _batch_file(
        [], raw_text=json.dumps(unknown_ver), path=tmp_path / 'v1.json'),
        None, 'schema_version'))
    # 缺必需字段必须被拒绝
    missing_field = _spec()
    missing_field.pop('outputs')
    cases.append(('field_missing', _batch_file([missing_field],
                                               path=tmp_path / 'nos.json'),
                  None, '缺少字段'))
    cases.append(('cross_project', _batch_file([_spec(project='other')],
                                               project='other',
                                               path=tmp_path / 'cross.json'),
                  None, '跨 project|不一致'))
    bad_path = _spec(outputs=['../escape'])
    cases.append(('bad_outputs_path', _batch_file([bad_path],
                                                  path=tmp_path / 'path.json'),
                  None, 'outputs|非法'))
    tampered = _batch_file([spec], path=tmp_path / 'tamper.json')
    payload = json.loads(tampered.read_text())
    payload['units'][0]['unit_id'] = 'f' * 64
    tampered.write_text(json.dumps(payload))
    cases.append(('tampered_unit_id', tampered, None, 'unit_id|不一致'))
    wrong_repo = _spec(repository='https://github.com/other/other.git')
    cases.append(('repo_mismatch', _batch_file([wrong_repo],
                                               path=tmp_path / 'repo.json'),
                  None, '仓库|repository'))

    for name, path, _, _frag in cases:
        fake = FakeColab()
        _install(monkeypatch, fake)
        argv = ['run', 'demo', '--unit-batch', str(path)]
        argv += ['--detach'] if '--json' not in argv else []
        with pytest.raises(core.BridgeError):
            _run(monkeypatch, fake, argv)
        assert fake.calls == [], name        # 无 sessions/new/download/stop
        assert fake.staged == [], name


# ---------------------------------------------------------------------------
# legacy `run`：上传集合/参数完全不变
# ---------------------------------------------------------------------------

def test_legacy_run_upload_set_and_params_unchanged(tmp_path, monkeypatch):
    _prepare_project(tmp_path, monkeypatch)
    fake = FakeColab()
    _install(monkeypatch, fake)
    assert _run(monkeypatch, fake, ['run', 'demo', '--allow-empty-input',
                                    '--detach', '--gpu', 'CPU']) == 0
    assert fake.staged == LEGACY_STAGE_SET
    assert not any('unit_batch' in json.dumps(payload)
                   for payload in fake.job_payloads)
    assert fake.count('new') == 1
    record = colab_runner.get_local(fake.new_sessions()[0])
    assert 'unit_batch_plan' not in record
    assert record['stopped'] is False


def test_retry_failed_without_batch_is_rejected(tmp_path, monkeypatch):
    _prepare_project(tmp_path, monkeypatch)
    fake = FakeColab()
    _install(monkeypatch, fake)
    # legacy run 不接受批专属开关（不会静默改变语义）。
    with pytest.raises(core.BridgeError, match='--retry-failed|unit-batch'):
        _run(monkeypatch, fake, ['run', 'demo', '--allow-empty-input',
                                 '--detach', '--retry-failed'])


# ---------------------------------------------------------------------------
# 批 staging：增量 batch_worker.py + plan['unit_batch']；缺失 fail closed
# ---------------------------------------------------------------------------

def test_batch_run_stages_batch_worker_and_payload(tmp_path, monkeypatch):
    ctx = _prepare_project(tmp_path, monkeypatch)
    fake = FakeColab()
    _install(monkeypatch, fake)
    path = _batch_file([_spec()], limits={'max_files': 500},
                       publish=900, path=tmp_path / 'b.json')
    assert _run(monkeypatch, fake, _argv(path, '--detach')) == 0
    # 批件 stage 集 = legacy 集合 + 显式增量 batch_worker.py
    # （插在 job 之后、凭据之前增量注入；legacy 成员的相对顺序不变）。
    assert fake.staged == ['mlbridge_worker.py', 'mlbridge_job.json',
                           'batch_worker.py', 'mlbridge_rclone.conf',
                           'mlbridge_restic_password']
    plan = fake.job_payloads[0]
    assert plan['job_id'] == plan['session']
    assert plan['commit'] == ctx['commit']
    assert plan['inputs'] == ['a' * 64]
    assert plan['setup'] == ctx['setup']
    assert plan['restic_repository'] == _NAMESPACE
    content = plan['unit_batch']
    assert content['batch_id'].startswith('batch-')
    assert content['schema_version'] == 2
    assert content['manifest_limits'] == {'max_files': 500,
                                           'max_bytes': 1 << 30,
                                           'timeout_seconds': 600}
    assert content['publish_timeout_seconds'] == 900
    item = content['units'][0]
    unit = P.WorkUnit.from_dict(item['spec'])
    assert item['unit_id'] == unit.identity
    assert item['attempt_id'] == f"att-{unit.identity[:12]}-1"
    assert item['execution'] == {'timeout_seconds': 600}
    unit_env = unit.environment
    assert unit_env['setup_sha256'] == P.canonical_digest(plan['setup'])
    # 本地任务记录携带远端计划（后续 watch 逐单元对账依赖）。
    record = colab_runner.get_local(plan['session'])
    assert record['unit_batch_plan']['unit_batch']['batch_id'] == content['batch_id']
    assert record['stopped'] is False


def test_batch_worker_missing_is_fail_closed(tmp_path, monkeypatch):
    """batch_worker.py 本地缺失：任何分配之前拒绝，不 stage、不 new。"""
    ctx = _prepare_project(tmp_path, monkeypatch)
    fake = FakeColab()
    _install(monkeypatch, fake)
    path = _batch_file([_spec()], path=tmp_path / 'b.json')
    monkeypatch.setattr(colab_runner, '__file__',
                        str(tmp_path / 'elsewhere' / 'colab_runner.py'))
    with pytest.raises(core.BridgeError, match='batch_worker'):
        _run(monkeypatch, fake, _argv(path, '--detach'))
    assert not any(c[0] == 'new' for c in fake.calls)
    assert fake.staged == []


def test_on_assign_failure_does_not_stage_and_stops_instance(tmp_path,
                                                             monkeypatch):
    """分配成功后的台账对齐失败：不 staging、不删除凭据，停止刚分配实例。"""
    ctx = _prepare_project(tmp_path, monkeypatch)
    fake = FakeColab()
    _install(monkeypatch, fake)
    path = _batch_file([_spec()], path=tmp_path / 'b.json')
    parsed = BC.parse_batch_file(BC.load_batch_file(path),
                                 project_name='demo', repository=_REPO,
                                 commit=ctx['commit'], setup=ctx['setup'],
                                 restic_namespace=_NAMESPACE)
    content = BC.build_unit_batch_content(parsed, {parsed['units'][0]['unit_id']:
                                                   'att-x-1'})
    plan = {'commit': ctx['commit'], 'inputs': parsed['inputs'],
            'setup': ctx['setup'], 'unit_batch': content}

    def boom(session):
        raise core.BridgeError('注入：台账对齐失败')

    with pytest.raises(core.BridgeError, match='台账对齐失败'):
        colab_runner.launch('demo', 'example', [], 'CPU', unit_batch=plan,
                            on_assign=boom)
    assert fake.count('new') == 1                     # `colab new` 已发生
    assert fake.staged == []                          # 绝不进入 staging
    assert any(c[0] == 'stopped' for c in fake.calls)  # 兑底停止刚分配实例


# ---------------------------------------------------------------------------
# 恢复对账：部分 completed / 全部 completed / 未决 attempt / 确定终态
# ---------------------------------------------------------------------------

def test_all_completed_reports_zero_instances(tmp_path, monkeypatch):
    """全部 unit 已 completed：一次性零实例报告，不调用任何 colab 命令。"""
    ctx = _prepare_project(tmp_path, monkeypatch)
    raw = _spec()
    unit = _full_unit(raw, ctx)
    ledger = Ledger('demo')
    uid = ledger.plan(unit)
    attempt_id = BC.attempt_id_for(uid, 1)
    ledger.start_attempt(uid, attempt_id, 'sess-old')
    _commit_unit(unit, ledger, uid, attempt_id, 'sess-old')
    fake = FakeColab()
    _install(monkeypatch, fake)
    path = _batch_file([raw], path=tmp_path / 'b.json')
    code = _run(monkeypatch, fake, _argv(path))
    assert code == 0
    assert fake.calls == []           # 零实例：连 sessions 都不需要
    assert fake.staged == []


def test_partial_completed_only_pending_runs(tmp_path, monkeypatch):
    """批内部分 completed：只把未完成单元放进远端计划；已完成不重算。"""
    ctx = _prepare_project(tmp_path, monkeypatch)
    done_raw = _spec(params={'only': 'done'}, outputs=['done-out/'])
    pend_raw = _spec(params={'only': 'pending'}, outputs=['pend-out/'])
    done_unit = _full_unit(done_raw, ctx)
    ledger = Ledger('demo')
    uid_done = ledger.plan(done_unit)
    att = BC.attempt_id_for(uid_done, 1)
    ledger.start_attempt(uid_done, att, 'sess-old')
    _commit_unit(done_unit, ledger, uid_done, att, 'sess-old')
    res_before = ledger.get(uid_done)['result']

    fake = FakeColab()
    _install(monkeypatch, fake)
    path = _batch_file([done_raw, pend_raw], path=tmp_path / 'b.json')
    assert _run(monkeypatch, fake, _argv(path, '--detach')) == 0
    payload = fake.job_payloads[0]
    uids = [item['unit_id'] for item in payload['unit_batch']['units']]
    assert uids == [P.unit_id(_full_unit(pend_raw, ctx))]  # 只有未完成 unit
    assert ledger.get(uid_done)['result'] == res_before    # 成果无任何改写


def test_unresolved_attempt_blocks_with_session(tmp_path, monkeypatch):
    """未决 attempt：阻止新分配；报告其 session；不改台账、不 stop。"""
    ctx = _prepare_project(tmp_path, monkeypatch)
    raw = _spec()
    unit, ledger, uid, attempt_id = _preseed_pending(ctx, raw, 'mlb-prev-0001')
    before = _ledger_file(tmp_path).read_bytes()
    fake = FakeColab()
    _install(monkeypatch, fake)
    # 无远端状态（download 失败 → UNREACHABLE）：无证据 → 停止。
    with pytest.raises(core.BridgeError, match='mlb-prev-0001'):
        _run(monkeypatch, fake, _argv(_batch_file([raw],
                                                  path=tmp_path / 'b.json')))
    assert fake.count('new') == 0
    assert not any(c[0] == 'stopped' for c in fake.calls)  # 绝不自动 stop
    assert _ledger_file(tmp_path).read_bytes() == before   # 台账不改


def test_unknown_attempt_session_state_no_evidence(tmp_path, monkeypatch):
    """远端 session 存在但无批逐单元 outcome（如 UNREACHABLE/废状态）：
    同样按 UNRESOLVED 停止，不把资源失联写成 task_failed。"""
    ctx = _prepare_project(tmp_path, monkeypatch)
    raw = _spec()
    unit, ledger, uid, attempt_id = _preseed_pending(ctx, raw, 'mlb-prev-0002')
    fake = FakeColab()
    _install(monkeypatch, fake)
    # 一个无 batch 字段的状态（“资源在但证据无效”）。
    fake.set_state('mlb-prev-0002', {'phase': 'RUNNING', 'session': 'mlb-prev-0002'})
    before = _ledger_file(tmp_path).read_bytes()
    with pytest.raises(core.BridgeError, match='未决 attempt|UNRESOLVED|不裁决'):
        _run(monkeypatch, fake, _argv(_batch_file([raw],
                                                  path=tmp_path / 'b.json')))
    assert fake.count('new') == 0
    assert _ledger_file(tmp_path).read_bytes() == before


def test_success_evidence_reconciles_and_commits(tmp_path, monkeypatch, capsys):
    """确定成功状态 + 契约回执 → reconcile(task_succeeded)+commit_result；
    台账显式记录 worker 自报，不宣称本地独立验收。"""
    ctx = _prepare_project(tmp_path, monkeypatch)
    raw = _spec()
    unit, ledger, uid, attempt_id = _preseed_pending(ctx, raw, 'mlb-prev-0003')
    # 用 parse 后真实 batch_id 构造状态/回执。
    parsed = BC.parse_batch_file(BC.load_batch_file(_batch_file([raw],
                             path=tmp_path / 'b.json')),
                             project_name='demo', repository=_REPO,
                             commit=ctx['commit'], setup=ctx['setup'],
                             restic_namespace=_NAMESPACE)
    batch_id = parsed['batch_id']
    fake = FakeColab()
    _install(monkeypatch, fake)
    fake.set_state('mlb-prev-0003', _outcome_state(
        'mlb-prev-0003', batch_id,
        {uid: _outcome(uid, attempt_id, 'SUCCEEDED')}))
    fake.receipt('mlb-prev-0003', batch_id, uid,
                 _receipt(ctx, unit, uid, attempt_id, 'mlb-prev-0003', batch_id))
    code = _run(monkeypatch, fake, _argv(_batch_file([raw], path=tmp_path / 'b.json')))
    assert code == 0
    view = Ledger('demo').get(uid)
    assert view['state'] == 'completed'
    assert view['result']['snapshot'] == 'e' * 64
    resolution = view['attempts'][attempt_id]['resolution']
    assert resolution['outcome'] == 'task_succeeded'
    assert resolution['source'] == 'colab-receipt/mlb-prev-0003'
    assert '自报' in resolution['note'] and '独立' in resolution['note']
    out = capsys.readouterr().out
    assert '"allocated": false' in out
    assert fake.count('new') == 0  # 全部对账完成 → 零实例报告


def test_task_failed_no_auto_retry_then_retry_failed_new_attempt(
        tmp_path, monkeypatch):
    """确定失败 → reconcile(task_failed)；默认不自动重试（报错且不分配）；
    显式 --retry-failed 才开新 attempt（seq+1）进入远端计划。"""
    ctx = _prepare_project(tmp_path, monkeypatch)
    raw = _spec()
    path = _batch_file([raw], path=tmp_path / 'b.json')
    parsed = BC.parse_batch_file(BC.load_batch_file(path),
                                 project_name='demo', repository=_REPO,
                                 commit=ctx['commit'], setup=ctx['setup'],
                                 restic_namespace=_NAMESPACE)
    batch_id = parsed['batch_id']
    fake = FakeColab()
    _install(monkeypatch, fake)
    unit, ledger, uid, attempt_id = _preseed_pending(ctx, raw, 'mlb-prev-0004')
    ledger_file_before = _ledger_file(tmp_path).read_bytes()
    fake.set_state('mlb-prev-0004', _outcome_state(
        'mlb-prev-0004', batch_id,
        {uid: _outcome(uid, attempt_id, 'TASK_FAILED',
                       stage='command', detail='退出码 7')}))

    # 1) 默认策略：task_failed 已确定，必须显式 --retry-failed。
    with pytest.raises(core.BridgeError, match='不自动重试|--retry-failed'):
        _run(monkeypatch, fake, _argv(path))
    view = Ledger('demo').get(uid)
    assert view['attempts'][attempt_id]['resolution']['outcome'] == 'task_failed'
    assert fake.count('new') == 0
    assert (view['attempts'][attempt_id]['resolution']['terminal_state']
            == 'FAILED')

    # 2) 显式 --retry-failed：新 attempt（seq=2）并分配（fake new 成功）。
    fake.set_state('mlb-prev-0004', _outcome_state(
        'mlb-prev-0004', batch_id,
        {uid: _outcome(uid, attempt_id, 'TASK_FAILED',
                       stage='command', detail='退出码 7')}))
    assert _run(monkeypatch, fake, _argv(path, '--retry-failed', '--detach')) == 0
    view = Ledger('demo').get(uid)
    assert len(view['attempts']) == 2
    second = BC.attempt_id_for(uid, 2)
    assert second in view['attempts']
    assert view['attempts'][second]['resolution'] is None  # 新 attempt 未解决
    payload = fake.job_payloads[0]
    assert payload['unit_batch']['units'][0]['attempt_id'] == second
    assert fake.count('new') == 1
    # 旧失败 attempt 保留在台账（可追溯）。
    assert attempt_id in view['attempts']
    assert ledger_file_before != _ledger_file(tmp_path).read_bytes()


# ---------------------------------------------------------------------------
# 不确定 outcome：PUBLISH_UNKNOWN / RECEIPT_FAILED —— 绝不转失败也不当成功
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('kind', ['publish_unknown', 'receipt_failed',
                                  'not_started'])
def test_uncertain_outcomes_never_reconciled(tmp_path, monkeypatch, kind):
    ctx = _prepare_project(tmp_path, monkeypatch)
    raw = _spec()
    path = _batch_file([raw], path=tmp_path / 'b.json')
    parsed = BC.parse_batch_file(BC.load_batch_file(path),
                                 project_name='demo', repository=_REPO,
                                 commit=ctx['commit'], setup=ctx['setup'],
                                 restic_namespace=_NAMESPACE)
    batch_id = parsed['batch_id']
    outcome_kind = {'publish_unknown': 'PUBLISH_UNKNOWN',
                    'receipt_failed': 'RECEIPT_FAILED',
                    'not_started': 'NOT_STARTED'}[kind]
    fake = FakeColab()
    _install(monkeypatch, fake)
    unit, ledger, uid, attempt_id = _preseed_pending(ctx, raw, 'mlb-prev-0005')
    fake.set_state('mlb-prev-0005', _outcome_state(
        'mlb-prev-0005', batch_id,
        {uid: _outcome(uid, attempt_id, outcome_kind,
                       **({'snapshot_id': 'e' * 64}
                          if kind == 'receipt_failed' else {}))}))
    before = _ledger_file(tmp_path).read_bytes()
    with pytest.raises(core.BridgeError, match=outcome_kind.split('_')[0].upper()):
        _run(monkeypatch, fake, _argv(path))
    assert fake.count('new') == 0
    view = Ledger('demo').get(uid)
    assert view['state'] == 'pending'                      # 不转失败/折算成功
    assert view['attempts'][attempt_id]['resolution'] is None
    assert _ledger_file(tmp_path).read_bytes() == before   # 台账零改写


def test_receipt_missing_field_or_identity_mismatch_rejected_no_ledger_write(
        tmp_path, monkeypatch):
    """回执缺字段/身份不匹配：拒绝匿名回执，台账零改写；不分配新实例。"""
    ctx = _prepare_project(tmp_path, monkeypatch)
    raw = _spec()
    path = _batch_file([raw], path=tmp_path / 'b.json')
    parsed = BC.parse_batch_file(BC.load_batch_file(path),
                                 project_name='demo', repository=_REPO,
                                 commit=ctx['commit'], setup=ctx['setup'],
                                 restic_namespace=_NAMESPACE)
    batch_id = parsed['batch_id']
    fake = FakeColab()
    _install(monkeypatch, fake)
    unit, ledger, uid, attempt_id = _preseed_pending(ctx, raw, 'mlb-prev-0006')
    fake.set_state('mlb-prev-0006', _outcome_state(
        'mlb-prev-0006', batch_id,
        {uid: _outcome(uid, attempt_id, 'SUCCEEDED')}))
    receipt = _receipt(ctx, unit, uid, attempt_id, 'mlb-prev-0006', batch_id)

    # 1) 缺字段
    broken = {k: v for k, v in receipt.items() if k != 'manifest_sha256'}
    fake.receipt('mlb-prev-0006', batch_id, uid, broken)
    before = _ledger_file(tmp_path).read_bytes()
    with pytest.raises(core.BridgeError, match='回执|缺字段|manifest_sha256'):
        _run(monkeypatch, fake, _argv(path))
    assert _ledger_file(tmp_path).read_bytes() == before
    assert fake.count('new') == 0

    # 2) 身份不匹配（attempt_id 被替换）
    tampered = dict(receipt)
    tampered['attempt_id'] = 'att-other'
    fake.receipt('mlb-prev-0006', batch_id, uid, tampered)
    with pytest.raises(core.BridgeError, match='回执|身份'):
        _run(monkeypatch, fake, _argv(path))
    assert _ledger_file(tmp_path).read_bytes() == before

    # 3) 匿名（回执文件缺失：download 失败）
    fake.remote.pop(('mlb-prev-0006',
                     f'/content/mlbridge_receipts/{batch_id}/{uid}.json'))
    with pytest.raises(core.BridgeError, match='回执|不可用|无证据'):
        _run(monkeypatch, fake, _argv(path))
    assert _ledger_file(tmp_path).read_bytes() == before


def test_batch_failed_does_not_pollute_completed_unit(tmp_path, monkeypatch):
    """批整体 FAILED 只是控制面事实：已 completed 的 unit 结果绝不被改写。"""
    ctx = _prepare_project(tmp_path, monkeypatch)
    done_raw = _spec(params={'is': 'done'}, outputs=['done-out/'])
    fail_raw = _spec(params={'is': 'fail'}, command=['python', 'boom.py'],
                     outputs=['fail-out/'])
    done_unit = _full_unit(done_raw, ctx)
    ledger = Ledger('demo')
    uid_done = ledger.plan(done_unit)
    att_done = BC.attempt_id_for(uid_done, 1)
    ledger.start_attempt(uid_done, att_done, 'sess-old')
    _commit_unit(done_unit, ledger, uid_done, att_done, 'sess-old',
                 snapshot='c' * 64)
    result_before = ledger.get(uid_done)['result']

    path = _batch_file([done_raw, fail_raw], path=tmp_path / 'b.json')
    fake = FakeColab()
    _install(monkeypatch, fake)
    assert _run(monkeypatch, fake, _argv(path, '--detach')) == 0
    payload = fake.job_payloads[0]
    assert [item['unit_id'] for item in payload['unit_batch']['units']] == \
        [P.unit_id(_full_unit(fail_raw, ctx))]   # 只有失败（pending）unit 重跑
    assert Ledger('demo').get(uid_done)['result'] == result_before
    assert ledger.get(uid_done)['state'] == 'completed'


# ---------------------------------------------------------------------------
# units 只读查询：稳定输出、completed 与 UNRESOLVED 分明
# ---------------------------------------------------------------------------

def test_units_query_stable_and_distinct(tmp_path, monkeypatch, capsys):
    """units 查询：重复运行输出逐字节稳定；未决 attempt 与 completed 分明。"""
    ctx = _prepare_project(tmp_path, monkeypatch)
    done_raw = _spec(params={'is': 'done'})
    pend_raw = _spec(params={'is': 'open', 'v': 11}, outputs=['open-out/'])
    done_unit = _full_unit(done_raw, ctx)
    ledger = Ledger('demo')
    uid_done = ledger.plan(done_unit)
    att_done = BC.attempt_id_for(uid_done, 1)
    ledger.start_attempt(uid_done, att_done, 'sess-old')
    _commit_unit(done_unit, ledger, uid_done, att_done, 'sess-old',
                 snapshot='c' * 64)
    _preseed_pending(ctx, pend_raw, 'mlb-open-0001')

    outputs = []
    for _ in range(2):
        code = cli_mod.execute(cli_mod.parser().parse_args(['units', 'demo']))
        assert code == 0
        outputs.append(capsys.readouterr().out)
    assert outputs[0] == outputs[1]                       # 稳定
    json_outs = []
    for _ in range(2):
        cli_mod.execute(cli_mod.parser().parse_args(['units', 'demo', '--json']))
        json_outs.append(capsys.readouterr().out)
    assert json_outs[0] == json_outs[1]
    rep = json.loads(json_outs[0])
    assert set(rep) == {'project', 'units'}
    assert [u for u in rep['units']] == sorted(rep['units'])   # 排序稳定
    done_view = rep['units'][uid_done]
    assert done_view['state'] == 'completed'
    assert done_view['result']['snapshot'] == 'c' * 64
    human = outputs[0]
    assert f'UNRESOLVED' in human                          # 未决分明
    assert 'state=completed' in human
    assert f"{uid_done[:12]} state=completed" in human


# ---------------------------------------------------------------------------
# watch 批终态逐单元对账（run 非分离 / 独立 `mlbridge watch`）
# ---------------------------------------------------------------------------

def test_watch_batch_success_commits_and_stops(tmp_path, monkeypatch,
                                               capsys):
    """批 run 完成后 watch：全部单元确定成功且 auto_stop → commit + stop。"""
    ctx = _prepare_project(tmp_path, monkeypatch)
    raw = _spec()
    path = _batch_file([raw], path=tmp_path / 'b.json')
    session_holder = {}

    fake = FakeColab()
    _install(monkeypatch, fake)

    # 自定义 new 侧：记录会话并在分配后注册远端 state/receipt。
    orig_cli = fake.cli

    def cli_with_remote(*args, **kwargs):
        result = orig_cli(*args, **kwargs)
        if args[0] == 'new':
            session = args[2]
            parsed = BC.parse_batch_file(BC.load_batch_file(path),
                                         project_name='demo',
                                         repository=_REPO,
                                         commit=ctx['commit'],
                                         setup=ctx['setup'],
                                         restic_namespace=_NAMESPACE)
            bid = parsed['batch_id']
            uid = parsed['units'][0]['unit_id']
            attempt_id = f"att-{uid[:12]}-1"
            session_holder.update(session=session, bid=bid, uid=uid,
                                  attempt_id=attempt_id)
            fake.set_state(session, _outcome_state(
                session, bid, {uid: _outcome(uid, attempt_id, 'SUCCEEDED')}))
            fake.receipt(session, bid, uid, _receipt(
                ctx, parsed['units'][0]['spec'],
                uid, attempt_id, session, bid))
        return result
    monkeypatch.setattr(colab_runner, 'cli', cli_with_remote)
    code = _run(monkeypatch, fake, _argv(path))
    assert code == 0
    s = session_holder['session']
    view = Ledger('demo').get(session_holder['uid'])
    assert view['state'] == 'completed'
    assert fake.count('stopped') == 1                      # auto_stop 生效
    record = colab_runner.get_local(s)
    assert record['stopped'] is True


def test_watch_keeps_instance_when_unresolved_or_failed(tmp_path, monkeypatch):
    """FAILED/PUBLISH_UNKNOWN/未决：watch 不自动 stop，实例保留；
    exit code 非零；绝不把已 completed 的单元改写。"""
    ctx = _prepare_project(tmp_path, monkeypatch)
    raw = _spec()
    path = _batch_file([raw], path=tmp_path / 'b.json')
    fake = FakeColab()
    _install(monkeypatch, fake)
    holder = {}

    def cli_with_remote(*args, **kwargs):
        result = fake.cli(*args, **kwargs)
        if args[0] == 'new':
            session = args[2]
            parsed = BC.parse_batch_file(BC.load_batch_file(path),
                                         project_name='demo',
                                         repository=_REPO,
                                         commit=ctx['commit'],
                                         setup=ctx['setup'],
                                         restic_namespace=_NAMESPACE)
            uid = parsed['units'][0]['unit_id']
            attempt_id = f"att-{uid[:12]}-1"
            holder.update(session=session, uid=uid, attempt_id=attempt_id)
            # 批终态 FAILED：该 unit PUBLISH_UNKNOWN（发布不确定）。
            fake.set_state(session, _outcome_state(
                session, parsed['batch_id'],
                {uid: _outcome(uid, attempt_id, 'PUBLISH_UNKNOWN')}))
        return result
    monkeypatch.setattr(colab_runner, 'cli', cli_with_remote)
    code = _run(monkeypatch, fake, _argv(path))
    assert code == 1
    record = colab_runner.get_local(holder['session'])
    assert record['stopped'] is not True                  # 不自动 stop
    view = Ledger('demo').get(holder['uid'])
    assert view['state'] == 'pending'                     # 不转失败/不当成功
    assert view['attempts'][holder['attempt_id']]['resolution'] is None

    # 独立 watch：同样不 stop；仍保留可后续对账事实。
    code = cli_mod.execute(cli_mod.parser().parse_args(['watch', holder['session']]))
    assert code == 1
    assert sum(1 for c in fake.calls if c[0] == 'stopped') == 0


def test_no_retry_failed_flag_with_pending_only_units_runs_normally(
        tmp_path, monkeypatch):
    """全新 pending 批次：不需要 --retry-failed；分配后 ledger attempt 与
    远端 plan 对齐（_id 一致），surrogate 检查 attempt_id 完全匹配。"""
    ctx = _prepare_project(tmp_path, monkeypatch)
    raw = _spec()
    path = _batch_file([raw], path=tmp_path / 'b.json')
    fake = FakeColab()
    _install(monkeypatch, fake)
    assert _run(monkeypatch, fake, _argv(path, '--detach')) == 0
    payload = fake.job_payloads[0]
    unit_id = payload['unit_batch']['units'][0]['unit_id']
    attempt_id = payload['unit_batch']['units'][0]['attempt_id']
    ledger = Ledger('demo')
    view = ledger.get(unit_id)
    assert attempt_id in view['attempts']                # attempt 已对齐
    assert view['attempts'][attempt_id]['session'] == payload['session']


# ---------------------------------------------------------------------------
# 对齐窗口自愈：任何异常（含 KeyboardInterrupt）都停止实例；仅在 stop 确认
# 后写 controller 本地事实证据；stop 未确认不写证据；重跑需 --retry-failed
# ---------------------------------------------------------------------------

def _flaky_start_attempt(monkeypatch, real_start, limit):
    """包装 Ledger.start_attempt：完成成功 N 次写入后在窗口中途注入异常。"""
    state = {'n': 0}

    def flaky(self, uid, attempt_id, session):
        real_start(self, uid, attempt_id, session)   # 事务已提交，再注入
        state['n'] += 1
        if state['n'] >= limit:
            raise KeyboardInterrupt('注入：对齐窗口中途 Ctrl-C')
    monkeypatch.setattr(Ledger, 'start_attempt', flaky)
    return state


def test_alignment_abort_keyboardinterrupt_stops_and_writes_evidence(
        tmp_path, monkeypatch, capsys):
    """on_assign 窗口中途 KeyboardInterrupt：停止实例、不 staging；
    已写入的 attempt 获得 controller 本地事实证据（绝不伪造远端终态）；
    未到达的 unit 只有 plan 登记而无 attempt。"""
    ctx = _prepare_project(tmp_path, monkeypatch)
    a_raw = _spec(params={'who': 'a'}, outputs=['a-out/'])
    b_raw = _spec(params={'who': 'b', 'v': 13}, outputs=['b-out/'])
    path = _batch_file([a_raw, b_raw], path=tmp_path / 'b.json')
    fake = FakeColab()
    _install(monkeypatch, fake)
    _flaky_start_attempt(monkeypatch, Ledger.start_attempt, limit=1)

    with pytest.raises(KeyboardInterrupt):
        _run(monkeypatch, fake, _argv(path, '--detach'))
    session = fake.new_sessions()[0]
    assert fake.count('new') == 1                      # new 已分配
    assert fake.staged == []                           # 绝不进入 staging
    assert not any(c[0] in ('exec', 'upload') for c in fake.calls)
    assert any(c[0] == 'stopped' for c in fake.calls)  # 刚分配实例已停止
    # start_batch_attempts 按 sorted(uid) 顺序写入；中断发生在第一笔写入
    # 之后的窗口中途：首个 unit 的 attempt 已写入并获得 controller 本地
    # 事实证据（绝不允许伪造远端终态），第二个 unit 未到达、无 attempt。
    uid_a, uid_b = _full_unit(a_raw, ctx).identity, _full_unit(b_raw, ctx).identity
    first, second = sorted([uid_a, uid_b])
    att_first = BC.attempt_id_for(first, 1)
    res = Ledger('demo').get(first)['attempts'][att_first]['resolution']
    assert res['outcome'] == 'task_failed'
    assert res['terminal_state'] == 'FAILED'
    assert res['source'] == 'controller/alignment-abort'
    assert 'staging 未开始' in res['note'] and '已停止' in res['note']
    assert '远端' in res['note'] and '终态' in res['note']
    view_second = Ledger('demo').get(second)
    assert view_second['attempts'] == {} and view_second['state'] == 'pending'
    # 本地任务记录保留 session 名且确认停止（后续可核实）。
    record = colab_runner.get_local(session)
    assert record['stopped'] is True
    err = capsys.readouterr().err
    assert '已停止' in err and '--retry-failed' in err


def test_alignment_abort_stop_unconfirmed_writes_no_evidence(
        tmp_path, monkeypatch, capsys):
    """stop 未确认（instance_stopped=false）：绝不写证据（实例状态未知）；
    原始异常照常传播；session 名保留在本地记录供手动停止。"""
    ctx = _prepare_project(tmp_path, monkeypatch)
    raw = _spec()
    path = _batch_file([raw], path=tmp_path / 'b.json')
    fake = FakeColab()
    _install(monkeypatch, fake)

    def stop_boom(session):
        raise core.BridgeError('注入：stop 通道失联')
    monkeypatch.setattr(colab_runner, 'shutdown', stop_boom)
    real_start = Ledger.start_attempt

    def flaky(self, uid, attempt_id, session):
        real_start(self, uid, attempt_id, session)
        raise core.BridgeError('注入：台账对齐失败')
    monkeypatch.setattr(Ledger, 'start_attempt', flaky)

    with pytest.raises(core.BridgeError, match='台账对齐失败'):
        _run(monkeypatch, fake, _argv(path, '--detach'))
    assert fake.count('new') == 1
    assert fake.staged == []
    uid = _full_unit(raw, ctx).identity
    att = BC.attempt_id_for(uid, 1)
    view = Ledger('demo').get(uid)
    assert view['attempts'][att]['resolution'] is None   # 关键：未写证据
    record = colab_runner.get_local(fake.new_sessions()[0])
    assert record.get('stopped') is not True             # session 名保留
    assert 'instance_stopped=false' in capsys.readouterr().err


def test_keyboardinterrupt_outside_window_writes_no_evidence(
        tmp_path, monkeypatch):
    """窗口尚未打开（分配 `colab new` 阶段被取消）：台账只有 plan
    登记，既无 attempt 也无 abort 证据——不得把非对齐失败写成 abort。"""
    ctx = _prepare_project(tmp_path, monkeypatch)
    raw = _spec()
    path = _batch_file([raw], path=tmp_path / 'b.json')
    fake = FakeColab()
    _install(monkeypatch, fake)
    orig_cli = fake.cli

    def cli_ki(*args, **kwargs):
        if args[0] == 'new':
            raise KeyboardInterrupt('注入：分配阶段取消')
        return orig_cli(*args, **kwargs)
    monkeypatch.setattr(colab_runner, 'cli', cli_ki)
    with pytest.raises(KeyboardInterrupt):
        _run(monkeypatch, fake, _argv(path))
    # 台账文件的唯一内容来自 Ledger.plan 登记；无任何 attempt/证据。
    view = Ledger('demo').get(_full_unit(raw, ctx).identity)
    assert view['attempts'] == {}


def test_alignment_abort_evidence_unlocks_only_with_retry_failed(
        tmp_path, monkeypatch):
    """controller/alignment-abort 证据解开的 attempt 属确定失败：
    默认策略不自动重试；须显式 --retry-failed 才开新 attempt（seq 递增）。"""
    ctx = _prepare_project(tmp_path, monkeypatch)
    raw = _spec()
    path = _batch_file([raw], path=tmp_path / 'b.json')
    fake = FakeColab()
    _install(monkeypatch, fake)
    unit, ledger, uid, attempt_id = _preseed_pending(ctx, raw, 'mlb-abort-1')
    ledger.reconcile(uid, attempt_id, 'mlb-abort-1', {
        'source': 'controller/alignment-abort', 'terminal_state': 'FAILED',
        'note': '对齐窗口中止（controller 本地事实）……reopen 需 --retry-failed'})

    with pytest.raises(core.BridgeError, match='--retry-failed'):
        _run(monkeypatch, fake, _argv(path))
    assert fake.count('new') == 0                     # 不分配
    assert _run(monkeypatch, fake, _argv(path, '--retry-failed',
                                         '--detach')) == 0
    assert fake.count('new') == 1
    view = ledger.get(uid)
    assert BC.attempt_id_for(uid, 2) in view['attempts']  # 新 attempt
    assert view['attempts'][attempt_id]['resolution']['outcome'] == 'task_failed'


def test_state_evidence_identity_requires_matching_batch_id():
    """F5：同 (session, uid, attempt_id) 的他批状态不得归因本批 attempt。"""
    uid = 'a' * 64
    outcome = {'unit_id': uid, 'attempt_id': 'att-x-1', 'outcome': 'SUCCEEDED'}
    status = {'batch': {'session': 's1', 'batch_id': 'batch-foreign',
                        'outcomes': {uid: outcome}}}
    assert BC._state_evidence_identity(
        's1', status, uid, 'att-x-1', 'batch-me') is None   # 跨批拒绝
    assert BC._state_evidence_identity(
        'other', status, uid, 'att-x-1', 'batch-foreign') is None
    got = BC._state_evidence_identity('s1', status, uid, 'att-x-1',
                                      'batch-foreign')
    assert got is outcome


def test_other_batch_state_cannot_resolve_attempt(tmp_path, monkeypatch):
    """集成：远端回放另一批次的同元组状态（batch_id 不同）→ 不归因，
    未决停止、台账零改写、不新建 attempt。"""
    ctx = _prepare_project(tmp_path, monkeypatch)
    raw = _spec()
    path = _batch_file([raw], path=tmp_path / 'b.json')
    parsed = BC.parse_batch_file(BC.load_batch_file(path),
                                 project_name='demo', repository=_REPO,
                                 commit=ctx['commit'], setup=ctx['setup'],
                                 restic_namespace=_NAMESPACE)
    foreign_batch_id = ('batch-legend' + parsed['batch_id'][len('batch-legend'):])
    assert foreign_batch_id != parsed['batch_id']
    fake = FakeColab()
    _install(monkeypatch, fake)
    unit, ledger, uid, attempt_id = _preseed_pending(ctx, raw, 'mlb-xbatch-1')
    fake.set_state('mlb-xbatch-1', _outcome_state(
        'mlb-xbatch-1', foreign_batch_id,
        {uid: _outcome(uid, attempt_id, 'SUCCEEDED')}))
    before = _ledger_file(tmp_path).read_bytes()
    with pytest.raises(core.BridgeError, match='未决 attempt|UNRESOLVED|不裁决'):
        _run(monkeypatch, fake, _argv(path))
    assert fake.count('new') == 0
    assert _ledger_file(tmp_path).read_bytes() == before   # 台账零改写


# ---------------------------------------------------------------------------
# 崩溃窗口自愈：reconcile(task_succeeded) 已写、commit_result 未写（上次
# 运行在两步之间中断）→ 先按同一 session/batch_id 回执补提交（校验照旧）
# ---------------------------------------------------------------------------

def _preseed_crash_window(tmp_path, monkeypatch, fake, ctx, raw, path,
                          session='mlb-crash-11', *, snapshot='e' * 64):
    """真实崩溃窗口快照：上一次运行在 _commit_from_receipt 的 reconcile
    与 commit_result 之间中断 —— resolve 证据与其写入格式完全一致
    （含 snapshot 摘要的 note），只是 result 尚未提交。"""
    parsed = BC.parse_batch_file(BC.load_batch_file(path),
                                 project_name='demo', repository=_REPO,
                                 commit=ctx['commit'], setup=ctx['setup'],
                                 restic_namespace=_NAMESPACE)
    unit, ledger, uid, attempt_id = _preseed_pending(ctx, raw, session)
    ledger.reconcile(uid, attempt_id, session, {
        'source': f'colab-receipt/{session}', 'terminal_state': 'SUCCEEDED',
        'note': f'worker 自报回执 published（snapshot={snapshot}）；'
                '自报而非本地独立产物验收'})
    fake.receipt(session, parsed['batch_id'], uid,
                 _receipt(ctx, unit, uid, attempt_id, session,
                          parsed['batch_id'], snapshot=snapshot))
    return parsed, unit, ledger, uid, attempt_id


def test_crash_between_reconcile_and_commit_self_heals(
        tmp_path, monkeypatch, capsys):
    """崩溃窗口：上次运行 reconcile 后 commit 前中断 → 本本次启动先按
    同 session/batch_id 回执补提交（全部身份/摘要校验照旧），提交成功
    则零实例跳过，绝不重复执行同一 attempt。"""
    ctx = _prepare_project(tmp_path, monkeypatch)
    raw = _spec()
    path = _batch_file([raw], path=tmp_path / 'b.json')
    fake = FakeColab()
    _install(monkeypatch, fake)
    parsed, unit, ledger, uid, attempt_id = _preseed_crash_window(
        tmp_path, monkeypatch, fake, ctx, raw, path)
    code = _run(monkeypatch, fake, _argv(path))
    assert code == 0
    assert fake.count('new') == 0            # 全部补提交完成 → 零实例
    view = ledger.get(uid)
    assert view['state'] == 'completed'
    assert view['result']['snapshot'] == 'e' * 64
    assert view['result']['attempt_id'] == attempt_id
    assert view['attempts'][attempt_id]['resolution']['outcome'] \
        == 'task_succeeded'
    out = capsys.readouterr().out
    assert '"allocated": false' in out


def test_crash_window_heal_still_validates_receipt_fully(
        tmp_path, monkeypatch):
    """回执身份/摘要不符时补提交同样拒绝：校验绝不放宽；台账保持
    task_succeeded 未决成果（等待人工对账），不重复执行、不新建 attempt。"""
    ctx = _prepare_project(tmp_path, monkeypatch)
    raw = _spec()
    path = _batch_file([raw], path=tmp_path / 'b.json')
    fake = FakeColab()
    _install(monkeypatch, fake)
    parsed, unit, ledger, uid, attempt_id = _preseed_crash_window(
        tmp_path, monkeypatch, fake, ctx, raw, path, session='mlb-crash-12')
    # 篡改回执 snapshot_id（非 hex）→ 必须拒绝。
    broken = _receipt(ctx, unit, uid, attempt_id, 'mlb-crash-12',
                      parsed['batch_id'])
    broken['snapshot_id'] = 'z' * 64
    fake.receipt('mlb-crash-12', parsed['batch_id'], uid, broken)
    with pytest.raises(core.BridgeError, match='回执'):
        _run(monkeypatch, fake, _argv(path))
    view = ledger.get(uid)
    assert view['state'] == 'pending'
    assert view['result'] is None

    # 再换一份 attempt_id 不符的回执 → 同样拒绝（身份校验照旧）。
    wrong = _receipt(ctx, unit, uid, attempt_id, 'mlb-crash-12',
                     parsed['batch_id'])
    wrong['attempt_id'] = 'att-elsewhere-1'
    fake.receipt('mlb-crash-12', parsed['batch_id'], uid, wrong)
    with pytest.raises(core.BridgeError, match='回执|身份'):
        _run(monkeypatch, fake, _argv(path))
    view = ledger.get(uid)
    assert view['state'] == 'pending' and view['result'] is None


# ---------------------------------------------------------------------------
# F4：--retry-failed 重跑入队 unit 的报告 disposition = queued_retry
# ---------------------------------------------------------------------------

def test_retry_failed_report_shows_queued_retry(tmp_path, monkeypatch, capsys):
    """--retry-failed 开新 attempt 的 unit 在报告中显示独立 disposition
    `queued_retry`，不再冒用上一 attempt 的 reconciled_failed。"""
    ctx = _prepare_project(tmp_path, monkeypatch)
    raw = _spec()
    path = _batch_file([raw], path=tmp_path / 'b.json')
    fake = FakeColab()
    _install(monkeypatch, fake)
    unit, ledger, uid, attempt_id = _preseed_failed(ctx, raw, 'mlb-retry-9')
    assert _run(monkeypatch, fake, _argv(path, '--retry-failed',
                                         '--detach')) == 0
    out = capsys.readouterr().out
    assert '"queued_retry"' in out
    assert 'reconciled_failed' not in out


def test_core_proc_guard_blocks_real_subprocess_calls():
    """autouse 守卫自检：core.proc 的网络类命令调用会直接炸测试
    （防止未来在某条批路径上悄悄加回真实 restic/rclone/colab 调用）。"""
    with pytest.raises(AssertionError, match='测试守卫'):
        core.proc(['restic', 'snapshots'])
    with pytest.raises(AssertionError, match='测试守卫'):
        core.proc(['rclone', 'about', 'gdrive:'])
