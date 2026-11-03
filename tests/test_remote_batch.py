"""第二阶段切片：remote_worker 的远端 unit_batch 顺序执行（离线全路径）。

真实小子进程 + fake restic，不联网、不 torch、不下载：
- 两 unit 真实 Python command + validator，setup 恰一次，每 unit 一个快照；
- 失败语义（前序回执保留 / 后续 NOT_STARTED / PUBLISH_UNKNOWN / RECEIPT_FAILED）；
- 副作用前的 fail-closed 预校验（错误身份/schema/绑定/重叠/保留路径）；
- 所有权/BASE 执行锁：真实双进程下并发重复/无效 plan worker 被拒绝、
  不改 HEALTH worker 的 STATE/凭据；claim 初始 NOT_STARTED；
- 失败注入后的批记录/STATE/回执一致性与终态写顺序；
- 进程组组活性清理（抗 TERM 孙进程 + leader 先退出也收尾）、
  单文件/多输出 manifest 预算、元数据链 symlink 拒绝、
  有界发布诊断与身份门控终态。

批量 worker（batch_worker.py）是 protocol.WorkUnit 的本地镜像，镜像一致性
由 test_batch_identity_matches_protocol / test_batch_rejects_mismatched_id 保证。
"""
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from mlbridge import batch_worker as BW
from mlbridge import remote_worker
import mlbridge.protocol as P

FAKE_RESTIC = r'''#!/bin/sh
# restic 调用形如: restic -r REPO <backup|restore|version> ...；行记录全部参数。
sub=""
for a in "$@"; do
  case "$a" in backup|restore|version) sub="$a"; break;; esac
done
printf '%s\n' "$*" >> "${FAKE_RESTIC_LOG:?FAKE_RESTIC_LOG required}"
case "$sub" in
  version)
    echo "restic 0.17.0 compiled with go1.22.5 on linux/amd64"
    ;;
  restore)
    # 输入快照恢复在本测试中不写内容（唯一要求：成功返回）。
    exit 0
    ;;
  backup)
    if [ -n "${FAKE_RESTIC_NOISE:-}" ]; then
      i=0
      while [ $i -lt 20000 ]; do
        echo "noise-line-for-stderr-0123456789abcdef0123456789" >&2
        echo "noise-line-for-stdout-0123456789abcdef0123456789"
        i=$((i+1))
      done
    fi
    if [ -n "${FAKE_RESTIC_LEFTOVER:-}" ]; then
      "${FAKE_RESTIC_LEFTOVER}" "${FAKE_RESTIC_GC_PID}" &
    fi
    if [ -n "${FAKE_RESTIC_SLEEP:-}" ]; then sleep "$FAKE_RESTIC_SLEEP"; fi
    n=$(cat "${FAKE_RESTIC_COUNT:?FAKE_RESTIC_COUNT required}" 2>/dev/null || echo 0)
    n=$((n+1))
    printf '%s' "$n" > "${FAKE_RESTIC_COUNT:?FAKE_RESTIC_COUNT required}"
    if [ "${FAKE_RESTIC_RESPONSE:-summary}" != "summary" ]; then
      echo "not-json"; exit 0
    fi
    printf '{"message_type":"summary","snapshot_id":"b%063d"}\n' "$n"
    ;;
esac
exit 0
'''

# 真实小子进程：unit1 写 outputs1/a.txt；unit2 写 outputs2/b.txt。
WRITE_ONE = ("import pathlib\n"
             "p=pathlib.Path('outputs1')\np.mkdir(parents=True)\n"
             "(p/'a.txt').write_text('ONE')\n")
WRITE_TWO = ("import pathlib\n"
             "p=pathlib.Path('outputs2')\np.mkdir(parents=True)\n"
             "(p/'b.txt').write_text('TWO')\n")
CHECK_ONE = ("import pathlib, sys\n"
             "sys.exit(0 if pathlib.Path('outputs1/a.txt').read_text()=='ONE' else 1)\n")
CHECK_TWO = ("import pathlib, sys\n"
             "sys.exit(0 if pathlib.Path('outputs2/b.txt').read_text()=='TWO' else 1)\n")


def _command_validator(command, timeout=30.0):
    return {"kind": "command", "command": command,
            "timeout_seconds": timeout}


def make_repo(tmp_path):
    repo = tmp_path / 'git-src'
    repo.mkdir()
    subprocess.run(['git', 'init', '-q', str(repo)], check=True)
    subprocess.run(['git', '-C', str(repo), 'config', 'user.name', 'T'], check=True)
    subprocess.run(['git', '-C', str(repo), 'config', 'user.email', 't@e.c'], check=True)
    (repo / 'readme.md').write_text('demo workspace for batch tests\n')
    subprocess.run(['git', '-C', str(repo), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(repo), 'commit', '-qm', 'init'], check=True)
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'],
                                     text=True).strip()
    return repo, commit


def make_spec(repo, commit, setup, *, command, outputs, validator,
              project='demo', env_extra=None, params=None):
    """用 llbridge.protocol 构造合法 canonical spec，再拆成 raw dict。"""
    env = {'setup_sha256': P.canonical_digest(setup), 'interpreter': 'sys'}
    if env_extra:
        env.update(env_extra)
    return P.WorkUnit(project=project, repository=str(repo), commit=commit,
                      inputs={'data': 'a' * 64}, environment=env,
                      command=command, params={'k': 'v'} if params is None else params,
                      outputs=outputs, validator=validator).to_dict()


def make_item(spec, attempt_id='att-1', timeout=60, *, unit_id=None):
    """默认 unit_id 由 protocol 计算；预校验篡改 case 用 raw item，
    由调用方显式传 unit_id（或占位 id）。"""
    if unit_id is None:
        unit_id = P.unit_id(spec)
    return {'unit_id': unit_id, 'attempt_id': attempt_id,
            'spec': spec, 'execution': {'timeout_seconds': timeout}}


def make_plan(repo, commit, items, *, setup, batch_id='bt1', **batch_extra):
    plan = {'job_id': 'bt-job', 'session': 'bt-job', 'project': 'demo',
            'repository': str(repo), 'commit': commit,
            'restic_repository': 'rclone:gdrive:store/demo',
            'inputs': ['a' * 64], 'setup': setup,
            'unit_batch': {'batch_id': batch_id, 'schema_version': 2,
                           'units': items, **batch_extra}}
    return plan


DEFAULT_SETUP = [[sys.executable, '-c',
                  "import pathlib\n"
                  "p=pathlib.Path('SETUP.log'); p.open('a').write('x')"]]


def harness(tmp_path, monkeypatch, plan):
    base = tmp_path / 'remote-root'
    base.mkdir()
    (base / 'mlbridge_rclone.conf').write_text('[gdrive]\ntype=drive\n')
    (base / 'mlbridge_restic_password').write_text('test')
    (base / 'mlbridge_job.json').write_text(json.dumps(plan))
    fakebin = tmp_path / 'fakebin'
    fakebin.mkdir()
    count = tmp_path / 'restic-count.txt'
    count.write_text('0')
    restic_log = tmp_path / 'restic.log'
    restic_log.write_text('')
    fake_restic = fakebin / 'restic'
    fake_restic.write_text(FAKE_RESTIC)
    fake_restic.chmod(0o755)
    fake_rclone = fakebin / 'rclone'
    fake_rclone.write_text('#!/bin/sh\nexit 0\n')
    fake_rclone.chmod(0o755)
    monkeypatch.setenv('PATH', str(fakebin) + os.pathsep + os.environ['PATH'])
    monkeypatch.setenv('FAKE_RESTIC_LOG', str(restic_log))
    monkeypatch.setenv('FAKE_RESTIC_COUNT', str(count))
    monkeypatch.delenv('FAKE_RESTIC_SLEEP', raising=False)
    monkeypatch.delenv('FAKE_RESTIC_RESPONSE', raising=False)
    monkeypatch.setattr(remote_worker, 'BASE', base)
    monkeypatch.setattr(remote_worker, 'STATE', base / 'mlbridge_state.json')
    monkeypatch.setattr(remote_worker, 'PLAN', base / 'mlbridge_job.json')

    def run():
        remote_worker.main()
        return json.loads((base / 'mlbridge_state.json').read_text())

    return {'base': base, 'run': run, 'restic_log': restic_log,
            'count': count, 'ws': base / 'mlbridge-workspace'}


def restic_calls(log_path):
    lines = log_path.read_text().splitlines()
    return ([l for l in lines if ' backup ' in l],
            [l for l in lines if ' restore ' in l])


def receipt_dirs(base, batch_id='bt1'):
    return base / 'mlbridge_receipts' / batch_id


def batch_outcomes(state):
    return state['batch']['outcomes']


# ---------------------------------------------------------------------------
# 完整双单元流程
# ---------------------------------------------------------------------------

def test_batch_two_units_publish_and_receipts(tmp_path, monkeypatch):
    repo, commit = make_repo(tmp_path)
    spec1 = make_spec(repo, commit, DEFAULT_SETUP,
                      command=[sys.executable, '-c', WRITE_ONE],
                      outputs=['outputs1'],
                      validator=_command_validator([sys.executable, '-c', CHECK_ONE]))
    spec2 = make_spec(repo, commit, DEFAULT_SETUP,
                      command=[sys.executable, '-c', WRITE_TWO],
                      outputs=['outputs2'],
                      validator=_command_validator([sys.executable, '-c', CHECK_TWO]))
    items = [make_item(spec1, 'att-one'), make_item(spec2, 'att-two')]
    plan = make_plan(repo, commit, items, setup=DEFAULT_SETUP)
    ctx = harness(tmp_path, monkeypatch, plan)

    state = ctx['run']()
    assert state['phase'] == 'SUCCEEDED'

    backups, restores = restic_calls(ctx['restic_log'])
    assert len(backups) == 2 and len(restores) == 1
    # setup 恰唯一执行一次（同一时间只有一个 clone/restore/setup）。
    assert (ctx['ws'] / 'SETUP.log').read_text() == 'x'
    # 两个 unit 各自的真实 Python 输出存在，trees 不重叠。
    assert (ctx['ws'] / 'outputs1' / 'a.txt').read_text() == 'ONE'
    assert (ctx['ws'] / 'outputs2' / 'b.txt').read_text() == 'TWO'

    outcomes = batch_outcomes(state)
    assert [outcomes[i['unit_id']]['outcome'] for i in items] == ['SUCCEEDED', 'SUCCEEDED']
    snapshots = [outcomes[i['unit_id']]['snapshot_id'] for i in items]
    assert all(len(s) == 64 and s[0] == 'b' for s in snapshots)
    assert snapshots[0] != snapshots[1]

    for spec, item in ((spec1, items[0]), (spec2, items[1])):
        receipt_path = receipt_dirs(ctx['base']) / (item['unit_id'] + '.json')
        assert receipt_path.is_file()
        receipt = json.loads(receipt_path.read_text())
        assert receipt['schema_version'] == BW.RECEIPT_SCHEMA_VERSION
        assert receipt['batch_id'] == 'bt1'
        assert receipt['unit_id'] == item['unit_id'] == P.unit_id(spec)
        assert receipt['attempt_id'] == item['attempt_id']
        assert receipt['session'] == plan['session']
        assert receipt['namespace'] == plan['restic_repository']
        assert receipt['snapshot_id'] == outcomes[item['unit_id']]['snapshot_id']
        assert receipt['validator_result'] == {'kind': 'command', 'status': 'passed'}
        assert receipt['validator_spec_sha256'] == P.canonical_digest(
            P.WorkUnit.from_dict(spec).validator)
        assert receipt['phase'] == 'published'
        manifest_rel = receipt['manifest_path']
        assert manifest_rel == f'.mlbridge-metadata/bt1/{item["unit_id"]}/manifest.json'
        manifest_bytes = (ctx['ws'] / manifest_rel).read_bytes()
        assert receipt['manifest_sha256'] == hashlib.sha256(manifest_bytes).hexdigest()
        manifest = json.loads(manifest_bytes)
        assert manifest['files'] == {
            ('outputs1/a.txt' if item is items[0] else 'outputs2/b.txt'):
                hashlib.sha256(b'ONE' if item is items[0] else b'TWO').hexdigest()}
        assert manifest['schema_version'] == 2 and manifest['attempt_id'] == item['attempt_id']
    # 凭据清理照旧。
    assert not (ctx['base'] / 'mlbridge_rclone.conf').exists()
    assert not (ctx['base'] / 'mlbridge_restic_password').exists()


def test_batch_state_records_progress_file(tmp_path, monkeypatch):
    repo, commit = make_repo(tmp_path)
    spec1 = make_spec(repo, commit, DEFAULT_SETUP,
                      command=[sys.executable, '-c', WRITE_ONE],
                      outputs=['outputs1'],
                      validator=_command_validator([sys.executable, '-c', CHECK_ONE]))
    plan = make_plan(repo, commit, [make_item(spec1, 'att-one')], setup=DEFAULT_SETUP)
    ctx = harness(tmp_path, monkeypatch, plan)
    state = ctx['run']()
    record = json.loads((ctx['base'] / 'mlbridge_batch_state_bt1.json').read_text())
    assert record['status'] == 'succeeded'
    assert record['batch_id'] == 'bt1' and record['session'] == 'bt-job'
    uid = list(record['units'])[0]
    assert record['units'][uid]['attempt_id'] == 'att-one'
    assert record['unit_outcomes'][uid]['snapshot_id'] == \
        batch_outcomes(state)[uid]['snapshot_id']


def test_batch_identity_matches_protocol():
    """batch_worker 是 protocol 镜像：规范化输出与 unit id 必须一致。"""
    spec = make_spec('/tmp/some-repo', 'a' * 40, DEFAULT_SETUP,
                     command=['python', 'run.py'],
                     outputs=['outs'], validator=_command_validator(['ls']))
    batch = remote_worker._batch_module()
    plan = make_plan('/tmp/some-repo', 'a' * 40, [make_item(spec, 'att-id')],
                     setup=DEFAULT_SETUP)
    validated = batch.validate_batch(plan)
    item = validated['units'][0]
    unit = P.WorkUnit.from_dict(spec)
    assert item['spec'] == unit.to_dict()
    assert item['unit_id'] == unit.identity == P.unit_id(spec)
    assert item['validator_spec_sha256'] == P.canonical_digest(unit.validator)
    assert validated['setup_sha256'] == P.canonical_digest(DEFAULT_SETUP)


# ---------------------------------------------------------------------------
# 失败语义
# ---------------------------------------------------------------------------

def test_second_unit_command_failure_preserves_first_receipt(tmp_path, monkeypatch):
    repo, commit = make_repo(tmp_path)
    ok_spec = make_spec(repo, commit, DEFAULT_SETUP,
                        command=[sys.executable, '-c', WRITE_ONE],
                        outputs=['outputs1'],
                        validator=_command_validator([sys.executable, '-c', CHECK_ONE]))
    # unit2/unit3 的 command 显式失败；unit3 显式声明 third 树（不动 outputs1）。
    fail_spec = make_spec(repo, commit, DEFAULT_SETUP,
                          command=[sys.executable, '-c', 'raise SystemExit(7)'],
                          outputs=['outputs2'],
                          validator=_command_validator([sys.executable, '-c', CHECK_TWO]))
    third_spec = make_spec(repo, commit, DEFAULT_SETUP,
                           command=[sys.executable, '-c', WRITE_TWO],
                           outputs=['other-tree'],
                           validator=_command_validator([sys.executable, '-c', CHECK_TWO]))
    items = [make_item(ok_spec, 'att-1'), make_item(fail_spec, 'att-2'),
             make_item(third_spec, 'att-3')]
    plan = make_plan(repo, commit, items, setup=DEFAULT_SETUP)
    ctx = harness(tmp_path, monkeypatch, plan)

    state = ctx['run']()
    assert state['phase'] == 'FAILED'
    outcomes = batch_outcomes(state)
    assert outcomes[items[0]['unit_id']]['outcome'] == 'SUCCEEDED'
    failed = outcomes[items[1]['unit_id']]
    assert failed['outcome'] == 'TASK_FAILED'
    assert failed['stage'] == 'command'
    assert '退出码 7' in failed['detail']
    assert outcomes[items[2]['unit_id']] == {
        'unit_id': items[2]['unit_id'],
        'outcome': 'NOT_STARTED',
        'attempt_id': 'att-3',
        'note': '前序单元失败；本批未启动该单元',
    }
    batch_record = json.loads((ctx['base'] / 'mlbridge_batch_state_bt1.json').read_text())
    assert batch_record['status'] == 'failed'

    backups, _ = restic_calls(ctx['restic_log'])
    assert len(backups) == 1  # 失败后不再 publish/继续单元
    assert (receipt_dirs(ctx['base']) / (items[0]['unit_id'] + '.json')).is_file()
    assert not (receipt_dirs(ctx['base']) / (items[1]['unit_id'] + '.json')).exists()
    assert not (ctx['ws'] / 'outputs2').exists()  # unit2 命令没有真实产物
    # 现场保留供排查（不清理旧 clone）。
    assert (ctx['ws'] / 'outputs1' / 'a.txt').read_text() == 'ONE'


def test_validator_failure_no_backup(tmp_path, monkeypatch):
    repo, commit = make_repo(tmp_path)
    bad_val = _command_validator([sys.executable, '-c',
                                  "import sys; sys.exit(3)"])
    spec = make_spec(repo, commit, DEFAULT_SETUP,
                     command=[sys.executable, '-c', WRITE_ONE],
                     outputs=['outputs1'], validator=bad_val)
    spec2 = make_spec(repo, commit, DEFAULT_SETUP,
                      command=[sys.executable, '-c', WRITE_TWO],
                      outputs=['outputs2'],
                      validator=_command_validator([sys.executable, '-c', CHECK_TWO]))
    items = [make_item(spec, 'att-v1'), make_item(spec2, 'att-v2')]
    plan = make_plan(repo, commit, items, setup=DEFAULT_SETUP)
    ctx = harness(tmp_path, monkeypatch, plan)

    state = ctx['run']()
    assert state['phase'] == 'FAILED'
    failed = batch_outcomes(state)[items[0]['unit_id']]
    assert failed['outcome'] == 'TASK_FAILED' and failed['stage'] == 'validator'
    assert batch_outcomes(state)[items[1]['unit_id']]['outcome'] == 'NOT_STARTED'
    backups, _ = restic_calls(ctx['restic_log'])
    assert len(backups) == 0  # validator 失败不 publish
    assert not receipt_dirs(ctx['base']).exists()
    assert (ctx['ws'] / 'outputs1' / 'a.txt').read_text() == 'ONE'  # 现场保留


def test_backup_without_valid_snapshot_is_publish_unknown(tmp_path, monkeypatch):
    repo, commit = make_repo(tmp_path)
    spec1 = make_spec(repo, commit, DEFAULT_SETUP,
                      command=[sys.executable, '-c', WRITE_ONE],
                      outputs=['outputs1'],
                      validator=_command_validator([sys.executable, '-c', CHECK_ONE]))
    spec2 = make_spec(repo, commit, DEFAULT_SETUP,
                      command=[sys.executable, '-c', WRITE_TWO],
                      outputs=['outputs2'],
                      validator=_command_validator([sys.executable, '-c', CHECK_TWO]))
    items = [make_item(spec1, 'att-p1'), make_item(spec2, 'att-p2')]
    plan = make_plan(repo, commit, items, setup=DEFAULT_SETUP)
    ctx = harness(tmp_path, monkeypatch, plan)
    # harness 返回后再设置（harness 内部会 清理/unset FAKE_* 变量）。
    monkeypatch.setenv('FAKE_RESTIC_RESPONSE', 'garbage')

    state = ctx['run']()
    assert state['phase'] == 'FAILED'
    unknown = batch_outcomes(state)[items[0]['unit_id']]
    assert unknown['outcome'] == 'PUBLISH_UNKNOWN'
    assert unknown['stage'] == 'publish'
    assert batch_outcomes(state)[items[1]['unit_id']]['outcome'] == 'NOT_STARTED'
    backups, _ = restic_calls(ctx['restic_log'])
    assert len(backups) == 1  # 不自动重算/不重传/不再继续
    assert not receipt_dirs(ctx['base']).exists()
    batch_record = json.loads((ctx['base'] / 'mlbridge_batch_state_bt1.json').read_text())
    assert batch_record['status'] == 'failed'


def test_backup_timeout_is_publish_unknown(tmp_path, monkeypatch):
    repo, commit = make_repo(tmp_path)
    spec1 = make_spec(repo, commit, DEFAULT_SETUP,
                      command=[sys.executable, '-c', WRITE_ONE],
                      outputs=['outputs1'],
                      validator=_command_validator([sys.executable, '-c', CHECK_ONE]))
    plan = make_plan(repo, commit, [make_item(spec1, 'att-p1')], setup=DEFAULT_SETUP,
                     publish_timeout_seconds=0.6)
    ctx = harness(tmp_path, monkeypatch, plan)
    monkeypatch.setenv('FAKE_RESTIC_SLEEP', '5')

    state = ctx['run']()
    assert state['phase'] == 'FAILED'
    unknown = batch_outcomes(state)[list(batch_outcomes(state))[0]]
    assert unknown['outcome'] == 'PUBLISH_UNKNOWN'
    assert '超时' in unknown['detail']
    assert not receipt_dirs(ctx['base']).exists()


def test_receipt_write_failure_is_not_swallowed(tmp_path, monkeypatch):
    repo, commit = make_repo(tmp_path)
    spec1 = make_spec(repo, commit, DEFAULT_SETUP,
                      command=[sys.executable, '-c', WRITE_ONE],
                      outputs=['outputs1'],
                      validator=_command_validator([sys.executable, '-c', CHECK_ONE]))
    unit_id = P.unit_id(spec1)
    plan = make_plan(repo, commit, [make_item(spec1, 'att-r1')], setup=DEFAULT_SETUP)
    ctx = harness(tmp_path, monkeypatch, plan)
    # 预先占用回执目录路径：mkdir 将失败 → 模拟回执写入失败。
    blocked = ctx['base'] / 'mlbridge_receipts' / 'bt1'
    blocked.parent.mkdir(parents=True)
    blocked.write_text('occupied\n')

    state = ctx['run']()
    assert state['phase'] == 'FAILED'
    outcome = batch_outcomes(state)[unit_id]
    assert outcome['outcome'] == 'RECEIPT_FAILED'
    assert outcome['stage'] == 'receipt'
    assert len(outcome['snapshot_id']) == 64  # 回执丢失但已确认 snapshot 必须可见
    assert outcome['attempt_id'] == 'att-r1'


# ---------------------------------------------------------------------------
# 输出完整性 / 进程组 / 上限
# ---------------------------------------------------------------------------

def test_preexisting_outputs_rejected_before_command(tmp_path, monkeypatch):
    repo, commit = make_repo(tmp_path)
    # git tracked 文件占据 unit 输出路径 → 输出预检拒绝（不运行命令）。
    (repo / 'outputs1').write_text('tracked residue\n')
    subprocess.run(['git', '-C', str(repo), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(repo), 'commit', '-qm', 'add residue'], check=True)
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'],
                                     text=True).strip()
    marker_cmd = [sys.executable, '-c',
                  "import pathlib; pathlib.Path('UNIT_RAN').write_text('y')"]
    spec = make_spec(repo, commit, DEFAULT_SETUP, command=marker_cmd,
                     outputs=['outputs1'],
                     validator=_command_validator(['true']))
    plan = make_plan(repo, commit, [make_item(spec, 'att-e1')], setup=DEFAULT_SETUP)
    ctx = harness(tmp_path, monkeypatch, plan)

    state = ctx['run']()
    assert state['phase'] == 'FAILED'
    failed = batch_outcomes(state)[list(batch_outcomes(state))[0]]
    assert failed['stage'] == 'outputs_precheck'
    assert '拒绝复用' in failed['detail']
    assert not (ctx['ws'] / 'UNIT_RAN').exists()  # command 未被执行
    assert not receipt_dirs(ctx['base']).exists()


def test_output_symlink_ancestor_rejected(tmp_path, monkeypatch):
    repo, commit = make_repo(tmp_path)
    # unit1 创建穿越链 symlink（在声明 outputs 之外）；unit2 输出落在链上。
    create_link = ("import os, pathlib\n"
                   "p=pathlib.Path('other')\np.mkdir(parents=True)\n"
                   "(p/'done.txt').write_text('ok')\n"
                   "os.makedirs('chain', exist_ok=True)\n"
                   "os.symlink('/etc', 'chain/link')\n")
    spec1 = make_spec(repo, commit, DEFAULT_SETUP,
                      command=[sys.executable, '-c', create_link],
                      outputs=['other'],
                      validator=_command_validator([sys.executable, '-c',
                                                    "import pathlib,sys\n"
                                                    "sys.exit(0 if pathlib.Path"
                                                    "('other/done.txt').exists() else 1)"]))
    spec2 = make_spec(repo, commit, DEFAULT_SETUP,
                      command=[sys.executable, '-c', WRITE_TWO],
                      outputs=['chain/link/escapes'],
                      validator=_command_validator(['true']))
    items = [make_item(spec1, 'att-1'), make_item(spec2, 'att-2')]
    plan = make_plan(repo, commit, items, setup=DEFAULT_SETUP)
    ctx = harness(tmp_path, monkeypatch, plan)

    state = ctx['run']()
    assert state['phase'] == 'FAILED'
    outcomes = batch_outcomes(state)
    assert outcomes[items[0]['unit_id']]['outcome'] == 'SUCCEEDED'
    failed = outcomes[items[1]['unit_id']]
    assert failed['stage'] == 'outputs_precheck'
    assert '符号链接' in failed['detail']


def test_manifest_limit_max_files_rejects(tmp_path, monkeypatch):
    repo, commit = make_repo(tmp_path)
    three_files = ("import pathlib\n"
                   "p=pathlib.Path('outputs1')\np.mkdir(parents=True)\n"
                   "for i in range(3):\n"
                   "    (p/f'f{i}.txt').write_text('x')\n")
    spec = make_spec(repo, commit, DEFAULT_SETUP,
                     command=[sys.executable, '-c', three_files],
                     outputs=['outputs1'],
                     validator=_command_validator(['true']))
    plan = make_plan(repo, commit, [make_item(spec, 'att-m1')], setup=DEFAULT_SETUP,
                     manifest_limits={'max_files': 2})
    ctx = harness(tmp_path, monkeypatch, plan)

    state = ctx['run']()
    assert state['phase'] == 'FAILED'
    failed = batch_outcomes(state)[list(batch_outcomes(state))[0]]
    assert failed['stage'] == 'outputs_check'
    assert 'max_files' in failed['detail']
    backups, _ = restic_calls(ctx['restic_log'])
    assert len(backups) == 0  # 拒绝而非静默漏项，不 publish


def test_command_timeout_kills_process_group(tmp_path, monkeypatch):
    repo, commit = make_repo(tmp_path)
    marker = 'mlbridge-timeout-marker-2907'
    sleeper = (f"import subprocess, sys\n"
               f"subprocess.Popen([sys.executable, '-c', "
               f"'import time; time.sleep(30)  # {marker}'])\n"
               f"import time; time.sleep(30)  # {marker}\n")
    spec = make_spec(repo, commit, DEFAULT_SETUP,
                     command=[sys.executable, '-c', sleeper],
                     outputs=['outputs1'],
                     validator=_command_validator(['true']))
    plan = make_plan(repo, commit, [make_item(spec, 'att-t1', timeout=1.2)],
                     setup=DEFAULT_SETUP)
    ctx = harness(tmp_path, monkeypatch, plan)

    def matching(marker_text):
        hits = []
        for pid_dir in Path('/proc').iterdir():
            if not pid_dir.name.isdigit():
                continue
            try:
                cmdline = (pid_dir / 'cmdline').read_bytes().decode('utf-8', 'replace')
            except OSError:
                continue
            if marker_text in cmdline and str(os.getpid()) != pid_dir.name:
                hits.append(pid_dir.name)
        return hits

    before = matching(marker)
    try:
        state = ctx['run']()
        assert state['phase'] == 'FAILED'
        failed = batch_outcomes(state)[list(batch_outcomes(state))[0]]
        assert failed['stage'] == 'command' and '超时' in failed['detail']
        # 进程组收尾：标记进程（含 worker 启动子进程）不再存在。
        assert matching(marker) == []
    finally:
        # 兜底清理，防止失败时残留进程
        for proc in matching(marker):
            try:
                os.kill(int(proc), 15)
            except OSError:
                pass


def test_duplicate_worker_refused(tmp_path, monkeypatch):
    """同批两次 worker：第二次在副作用前被 BASE flock + 既有状态记录拒绝。"""
    repo, commit = make_repo(tmp_path)
    spec = make_spec(repo, commit, DEFAULT_SETUP,
                     command=[sys.executable, '-c', WRITE_ONE],
                     outputs=['outputs1'],
                     validator=_command_validator([sys.executable, '-c', CHECK_ONE]))
    plan = make_plan(repo, commit, [make_item(spec, 'att-dup')], setup=DEFAULT_SETUP)
    ctx = harness(tmp_path, monkeypatch, plan)

    ctx['run']()
    state_path = ctx['base'] / 'mlbridge_batch_state_bt1.json'
    before = state_path.read_bytes()
    backups_before, _ = restic_calls(ctx['restic_log'])

    with pytest.raises(RuntimeError, match='拒绝重复执行'):
        ctx['run']()
    assert state_path.read_bytes() == before      # batch 状态记录未被改写
    backups_after, restores_after = restic_calls(ctx['restic_log'])
    assert backups_after == backups_before        # 没有任何重复 backup
    assert len(restores_after) == 1               # 也没有重复 restore/clone


# ---------------------------------------------------------------------------
# 预校验 fail closed：任何副作用（clone/restore/setup/凭据 chmod）之前拒绝
# ---------------------------------------------------------------------------

def expect_precheck_rejected(tmp_path, monkeypatch, plan):
    ctx = harness(tmp_path, monkeypatch, plan)
    with pytest.raises(RuntimeError):
        ctx['run']()
    state = json.loads((ctx['base'] / 'mlbridge_state.json').read_text())
    assert state['phase'] == 'FAILED'
    assert not (ctx['base'] / 'mlbridge-workspace').exists()
    backups, restores = restic_calls(ctx['restic_log'])
    assert not backups and not restores
    assert not (ctx['base'] / 'mlbridge_receipts').exists()
    assert not (ctx['base'] / 'mlbridge_batch_state_bt1.json').exists()
    return state


def _reject_cases():
    """每个 case 返回 (id, plan builder(repo, commit))；全部在 clone 前拒绝。"""
    def tampered_unit_id(repo, commit):
        spec = make_spec(repo, commit, DEFAULT_SETUP,
                         command=[sys.executable, '-c', WRITE_ONE],
                         outputs=['outputs1'],
                         validator=_command_validator(['true']))
        items = [make_item(spec, 'att-1')]
        plan = make_plan(repo, commit, items, setup=DEFAULT_SETUP)
        plan['unit_batch']['units'][0]['unit_id'] = 'f' * 64
        return plan

    def uppercase_unit_id(repo, commit):
        spec = make_spec(repo, commit, DEFAULT_SETUP,
                         command=[sys.executable, '-c', WRITE_ONE],
                         outputs=['outputs1'],
                         validator=_command_validator(['true']))
        items = [make_item(spec, 'att-1')]
        plan = make_plan(repo, commit, items, setup=DEFAULT_SETUP)
        plan['unit_batch']['units'][0]['unit_id'] = items[0]['unit_id'].upper()
        return plan

    def duplicate_attempt(repo, commit):
        s1 = make_spec(repo, commit, DEFAULT_SETUP,
                       command=[sys.executable, '-c', WRITE_ONE],
                       outputs=['outputs1'], validator=_command_validator(['true']))
        s2 = make_spec(repo, commit, DEFAULT_SETUP,
                       command=[sys.executable, '-c', WRITE_TWO],
                       outputs=['outputs2'], validator=_command_validator(['true']))
        items = [make_item(s1, 'att-shared'), make_item(s2, 'att-shared')]
        return make_plan(repo, commit, items, setup=DEFAULT_SETUP)

    def overlapping_outputs(repo, commit):
        s1 = make_spec(repo, commit, DEFAULT_SETUP,
                       command=[sys.executable, '-c', WRITE_ONE],
                       outputs=['outputs1'], validator=_command_validator(['true']))
        s2 = make_spec(repo, commit, DEFAULT_SETUP,
                       command=[sys.executable, '-c', WRITE_TWO],
                       outputs=['outputs1'], validator=_command_validator(['true']))
        items = [make_item(s1, 'att-1'), make_item(s2, 'att-2')]
        return make_plan(repo, commit, items, setup=DEFAULT_SETUP)

    def escaped_output_spec(repo, commit):
        legal = make_spec(repo, commit, DEFAULT_SETUP,
                          command=[sys.executable, '-c', WRITE_ONE],
                          outputs=['outputs1'], validator=_command_validator(['true']))
        raw = dict(legal)
        raw['outputs'] = ['../escape']
        # 保留原 unit_id：spec 路径合法性校验先于身份重算，必须先失败。
        items = [{'unit_id': P.unit_id(legal), 'attempt_id': 'att-1',
                  'spec': raw, 'execution': {'timeout_seconds': 60}}]
        return make_plan(repo, commit, items, setup=DEFAULT_SETUP)

    def git_component_output(repo, commit):
        legal = make_spec(repo, commit, DEFAULT_SETUP,
                          command=[sys.executable, '-c', WRITE_ONE],
                          outputs=['outputs1'], validator=_command_validator(['true']))
        raw = dict(legal)
        raw['outputs'] = ['not-git', 'x/.git/y']
        items = [{'unit_id': P.unit_id(legal), 'attempt_id': 'att-1',
                  'spec': raw, 'execution': {'timeout_seconds': 60}}]
        return make_plan(repo, commit, items, setup=DEFAULT_SETUP)

    def unknown_validator_kind(repo, commit):
        spec = make_spec(repo, commit, DEFAULT_SETUP,
                         command=[sys.executable, '-c', WRITE_ONE],
                         outputs=['outputs1'],
                         validator={'kind': 'regex', 'pattern': 'x'})
        items = [make_item(spec, 'att-1')]
        return make_plan(repo, commit, items, setup=DEFAULT_SETUP)

    def missing_setup_binding(repo, commit):
        spec = make_spec(repo, commit, DEFAULT_SETUP,
                         command=[sys.executable, '-c', WRITE_ONE],
                         outputs=['outputs1'], validator=_command_validator(['true']),
                         env_extra={})
        spec['environment'] = {'interpreter': 'sys'}  # 去掉 setup_sha256 绑定
        items = [make_item(spec, 'att-1')]
        return make_plan(repo, commit, items, setup=DEFAULT_SETUP)

    def commit_mismatch(repo, commit):
        spec = make_spec(repo, commit, DEFAULT_SETUP,
                         command=[sys.executable, '-c', WRITE_ONE],
                         outputs=['outputs1'], validator=_command_validator(['true']))
        plan = make_plan(repo, commit, [make_item(spec, 'att-1')],
                         setup=DEFAULT_SETUP)
        plan['commit'] = 'b' * 40
        return plan

    def inputs_order_mismatch(repo, commit):
        spec = make_spec(repo, commit, DEFAULT_SETUP,
                         command=[sys.executable, '-c', WRITE_ONE],
                         outputs=['outputs1'], validator=_command_validator(['true']))
        plan = make_plan(repo, commit, [make_item(spec, 'att-1')],
                         setup=DEFAULT_SETUP)
        plan['inputs'] = ['b' * 64]
        return plan

    def batch_schema_version_3(repo, commit):
        spec = make_spec(repo, commit, DEFAULT_SETUP,
                         command=[sys.executable, '-c', WRITE_ONE],
                         outputs=['outputs1'], validator=_command_validator(['true']))
        plan = make_plan(repo, commit, [make_item(spec, 'att-1')],
                         setup=DEFAULT_SETUP, )
        plan['unit_batch']['schema_version'] = 3
        return plan

    def unknown_version_spec(repo, commit):
        spec = make_spec(repo, commit, DEFAULT_SETUP,
                         command=[sys.executable, '-c', WRITE_ONE],
                         outputs=['outputs1'], validator=_command_validator(['true']))
        raw = dict(spec)
        raw['schema_version'] = 1  # 未知版本必须被拒绝
        items = [{'unit_id': P.unit_id(spec), 'attempt_id': 'att-1',
                  'spec': raw, 'execution': {'timeout_seconds': 60}}]
        return make_plan(repo, commit, items, setup=DEFAULT_SETUP)

    def bad_session(repo, commit):
        spec = make_spec(repo, commit, DEFAULT_SETUP,
                         command=[sys.executable, '-c', WRITE_ONE],
                         outputs=['outputs1'], validator=_command_validator(['true']))
        plan = make_plan(repo, commit, [make_item(spec, 'att-1')],
                         setup=DEFAULT_SETUP)
        plan['session'] = 'bad session!'
        return plan

    def bad_attempt_id(repo, commit):
        spec = make_spec(repo, commit, DEFAULT_SETUP,
                         command=[sys.executable, '-c', WRITE_ONE],
                         outputs=['outputs1'], validator=_command_validator(['true']))
        return make_plan(repo, commit, [make_item(spec, 'bad attempt!')],
                         setup=DEFAULT_SETUP)

    def environment_mismatch_across_units(repo, commit):
        s1 = make_spec(repo, commit, DEFAULT_SETUP,
                       command=[sys.executable, '-c', WRITE_ONE],
                       outputs=['outputs1'], validator=_command_validator(['true']))
        s2 = make_spec(repo, commit, DEFAULT_SETUP,
                       command=[sys.executable, '-c', WRITE_TWO],
                       outputs=['outputs2'], validator=_command_validator(['true']),
                       env_extra={'toolchain': 'different'})
        items = [make_item(s1, 'att-1'), make_item(s2, 'att-2')]
        return make_plan(repo, commit, items, setup=DEFAULT_SETUP)

    def reserved_metadata_output(repo, commit):
        spec = make_spec(repo, commit, DEFAULT_SETUP,
                         command=[sys.executable, '-c', WRITE_ONE],
                         outputs=['.mlbridge-metadata'], validator=_command_validator(['true']))
        return make_plan(repo, commit, [make_item(spec, 'att-1')],
                         setup=DEFAULT_SETUP)

    def empty_outputs(repo, commit):
        spec = make_spec(repo, commit, DEFAULT_SETUP,
                         command=[sys.executable, '-c', 'pass'],
                         outputs=[], validator=_command_validator(['true']))
        return make_plan(repo, commit, [make_item(spec, 'att-1')],
                         setup=DEFAULT_SETUP)

    def unknown_batch_field(repo, commit):
        spec = make_spec(repo, commit, DEFAULT_SETUP,
                         command=[sys.executable, '-c', WRITE_ONE],
                         outputs=['outputs1'], validator=_command_validator(['true']))
        return make_plan(repo, commit, [make_item(spec, 'att-1')],
                         setup=DEFAULT_SETUP, someday_field={'x': 1})

    def missing_execution_timeout(repo, commit):
        spec = make_spec(repo, commit, DEFAULT_SETUP,
                         command=[sys.executable, '-c', WRITE_ONE],
                         outputs=['outputs1'], validator=_command_validator(['true']))
        item = make_item(spec, 'att-1')
        del item['execution']['timeout_seconds']
        return make_plan(repo, commit, [item], setup=DEFAULT_SETUP)

    def null_unit_batch(repo, commit):
        spec = make_spec(repo, commit, DEFAULT_SETUP,
                         command=[sys.executable, '-c', WRITE_ONE],
                         outputs=['outputs1'], validator=_command_validator(['true']))
        plan = make_plan(repo, commit, [make_item(spec, 'att-1')],
                         setup=DEFAULT_SETUP)
        plan['unit_batch'] = None  # 显式 null：键存在即走批校验并拒绝
        return plan

    return [
        ('tampered_unit_id', tampered_unit_id),
        ('uppercase_unit_id', uppercase_unit_id),
        ('duplicate_attempt', duplicate_attempt),
        ('overlapping_outputs', overlapping_outputs),
        ('escaped_output_spec', escaped_output_spec),
        ('git_component_output', git_component_output),
        ('unknown_validator_kind', unknown_validator_kind),
        ('missing_setup_binding', missing_setup_binding),
        ('commit_mismatch', commit_mismatch),
        ('inputs_order_mismatch', inputs_order_mismatch),
        ('batch_schema_version_3', batch_schema_version_3),
        ('unknown_spec_schema_version', unknown_version_spec),
        ('bad_session', bad_session),
        ('bad_attempt_id', bad_attempt_id),
        ('environment_mismatch_across_units', environment_mismatch_across_units),
        ('reserved_metadata_output', reserved_metadata_output),
        ('empty_outputs', empty_outputs),
        ('unknown_batch_field', unknown_batch_field),
        ('missing_execution_timeout', missing_execution_timeout),
        ('null_unit_batch', null_unit_batch),
    ]


@pytest.mark.parametrize('case_id, builder', _reject_cases(), ids=[c[0] for c in _reject_cases()])
def test_batch_rejected_before_any_side_effect(case_id, builder, tmp_path, monkeypatch):
    repo, commit = make_repo(tmp_path)
    expect_precheck_rejected(tmp_path, monkeypatch, builder(repo, commit))


def test_unknown_spec_schema_version_rejected(tmp_path):
    """An unknown spec schema_version is refused with a plain version error."""
    spec = make_spec('/tmp/some-repo', 'a' * 40, DEFAULT_SETUP,
                     command=['python', 'run.py'],
                     outputs=['outs'], validator=_command_validator(['ls']))
    raw = dict(spec)
    raw['schema_version'] = 1
    batch = remote_worker._batch_module()
    with pytest.raises(batch.BatchError, match='schema_version'):
        batch._normalize_spec(raw)


def test_v2_spec_without_seed_passes_and_params_seed_is_ordinary():
    """v2 契约：spec 无 seed 字段正常通过；params 里的 seed 键与其它
    params 键同等对待（不做整数校验等任何特殊处理）。"""
    spec = make_spec('/tmp/some-repo', 'a' * 40, DEFAULT_SETUP,
                     command=['python', 'run.py'],
                     outputs=['outs'], validator=_command_validator(['ls']),
                     params={'seed': 'not-an-int', 'lr': 0.5})
    batch = remote_worker._batch_module()
    fields, identity = batch._normalize_spec(spec)
    assert fields['params'] == {'seed': 'not-an-int', 'lr': 0.5}
    unit = P.WorkUnit.from_dict(spec)
    assert identity == unit.identity
    # params 里的 seed 与其它 params 键一样参与身份，但无任何专属规则
    other = dict(spec)
    other['params'] = {'seed': 'not-an-int', 'lr': 0.6}
    assert batch._normalize_spec(other)[1] != identity


def test_batch_worker_missing_is_fail_closed(tmp_path, monkeypatch):
    """launch 未 stage batch_worker.py 时：opt-in 分支清晰报错且无任何副作用。"""
    repo, commit = make_repo(tmp_path)
    spec = make_spec(repo, commit, DEFAULT_SETUP,
                     command=[sys.executable, '-c', WRITE_ONE],
                     outputs=['outputs1'], validator=_command_validator(['true']))
    plan = make_plan(repo, commit, [make_item(spec, 'att-1')], setup=DEFAULT_SETUP)
    ctx = harness(tmp_path, monkeypatch, plan)
    # __file__ 指向无 sibling 的位置 → 模拟 "未 stage"
    monkeypatch.setattr(remote_worker, '__file__',
                        str(tmp_path / 'elsewhere' / 'mlbridge_worker.py'))
    with pytest.raises(RuntimeError, match='stage'):
        ctx['run']()
    state = json.loads((ctx['base'] / 'mlbridge_state.json').read_text())
    assert state['phase'] == 'FAILED'
    assert not (ctx['base'] / 'mlbridge-workspace').exists()


# ---------------------------------------------------------------------------
# 所有权：BASE 执行锁（真实双进程）；claim 初始 NOT_STARTED
# ---------------------------------------------------------------------------

_SUBPROCESS_WORKER = '''\
import pathlib
import sys

sys.path.insert(0, sys.argv[2])
from mlbridge import remote_worker  # noqa: E402

base = pathlib.Path(sys.argv[1])
remote_worker.BASE = base
remote_worker.STATE = base / "mlbridge_state.json"
remote_worker.PLAN = base / "mlbridge_job.json"
remote_worker.main()
'''


def _wait_for_file(path, timeout=60.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.05)
    raise AssertionError(f'等待超时: {path}')


def _pid_gone(pid, timeout=15.0) -> bool:
    """有界确认 pid 不再存活；zombie（Z/X）按已处理对待。绝不全系统扫描。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            raw = Path(f'/proc/{pid}/stat').read_bytes()
        except OSError:
            return True
        close = raw.rfind(b')')
        state = raw[close + 2:].split()[0] if close >= 0 else b'Z'
        if state in (b'Z', b'X'):
            return True
        time.sleep(0.05)
    return False


_BLOCKER = (
    "import pathlib, sys, time\n"
    "started, release = sys.argv[1], sys.argv[2]\n"
    "pathlib.Path(started).write_text('ready')\n"
    "deadline = time.monotonic() + 90\n"
    "while not pathlib.Path(release).exists() and time.monotonic() < deadline:\n"
    "    time.sleep(0.05)\n"
    "p = pathlib.Path('outputs1'); p.mkdir(parents=True)\n"
    "(p / 'a.txt').write_text('ONE')\n")


def _resident_worker_setup(tmp_path, monkeypatch, repo, commit, unit1_cmd,
                           batch_id='bt-conc'):
    """准备 BASE/fakebin/PATH 并返回 (plan, base, fake 输出路径)。"""
    spec1 = make_spec(repo, commit, DEFAULT_SETUP, command=unit1_cmd,
                      outputs=['outputs1'], validator=_command_validator(['true']))
    spec2 = make_spec(repo, commit, DEFAULT_SETUP,
                      command=[sys.executable, '-c', WRITE_TWO],
                      outputs=['outputs2'],
                      validator=_command_validator([sys.executable, '-c', CHECK_TWO]))
    items = [make_item(spec1, 'att-a1'), make_item(spec2, 'att-a2')]
    plan = make_plan(repo, commit, items, setup=DEFAULT_SETUP, batch_id=batch_id)
    base = tmp_path / 'remote-root'
    base.mkdir()
    (base / 'mlbridge_rclone.conf').write_text('[gdrive]\ntype=drive\n')
    (base / 'mlbridge_restic_password').write_text('test')
    (base / 'mlbridge_job.json').write_text(json.dumps(plan))
    fakebin = tmp_path / 'fakebin'
    fakebin.mkdir()
    count = tmp_path / 'restic-count.txt'
    count.write_text('0')
    restic_log = tmp_path / 'restic.log'
    restic_log.write_text('')
    restic = fakebin / 'restic'
    restic.write_text(FAKE_RESTIC)
    restic.chmod(0o755)
    rclone = fakebin / 'rclone'
    rclone.write_text('#!/bin/sh\nexit 0\n')
    rclone.chmod(0o755)
    monkeypatch.setenv('PATH', str(fakebin) + os.pathsep + os.environ['PATH'])
    monkeypatch.setenv('FAKE_RESTIC_LOG', str(restic_log))
    monkeypatch.setenv('FAKE_RESTIC_COUNT', str(count))
    monkeypatch.setattr(remote_worker, 'BASE', base)
    monkeypatch.setattr(remote_worker, 'STATE', base / 'mlbridge_state.json')
    monkeypatch.setattr(remote_worker, 'PLAN', base / 'mlbridge_job.json')
    return {'plan': plan, 'items': items, 'base': base,
            'restic_log': restic_log, 'count': count}


def _spawn_resident(tmp_path, base):
    src_root = Path(remote_worker.__file__).parent.parent
    script = tmp_path / 'mlbridge_resident_worker.py'
    script.write_text(_SUBPROCESS_WORKER)
    stdout = tmp_path / 'resident-out.log'
    stderr = tmp_path / 'resident-err.log'
    return subprocess.Popen(
        [sys.executable, str(script), str(base), str(src_root)],
        stdout=stdout.open('wb'), stderr=stderr.open('wb'))


def test_concurrent_duplicate_and_invalid_worker_share_nothing(tmp_path, monkeypatch):
    """真实双进程：A 执行中，重复/无效 plan 的 B 在执行锁处被拒绝，
    不写 A 的 STATE、不删凭据；A 的终态保留。"""
    repo, commit = make_repo(tmp_path)
    started = tmp_path / 'workerA_started'
    release = tmp_path / 'workerA_release'
    unit1_cmd = [sys.executable, '-c', _BLOCKER, str(started), str(release)]
    cfg = _resident_worker_setup(tmp_path, monkeypatch, repo, commit, unit1_cmd)
    base = cfg['base']
    record_path = base / 'mlbridge_batch_state_bt-conc.json'

    proc = _spawn_resident(tmp_path, base)
    try:
        _wait_for_file(started)
        # A 已通过 claim 与执行锁。claim 只把实际启动的 unit 记为 RUNNING
        # 并记录 current_unit；其余仍为 NOT_STARTED（绝不预标全部 RUNNING）。
        record_initial = json.loads(record_path.read_text())
        others = [v for k, v in record_initial['units'].items()
                  if k != cfg['items'][0]['unit_id']]
        assert all(u['outcome'] == 'NOT_STARTED' for u in others)
        assert (record_initial['units'][cfg['items'][0]['unit_id']]
                ['outcome'] == 'RUNNING')
        assert record_initial['current_unit'] == cfg['items'][0]['unit_id']

        state_before = (base / 'mlbridge_state.json').read_bytes()
        record_before = record_path.read_bytes()
        assert (base / 'mlbridge_rclone.conf').is_file()
        assert (base / 'mlbridge_restic_password').is_file()

        # 同一 plan 的重复 worker：执行锁处非阻塞拒绝。
        with pytest.raises(RuntimeError, match='执行锁'):
            remote_worker.main()
        # 无效 plan（unit_batch 显式 null）：校验在锁内，同样先被拒绝，
        # 无法把任何状态写入共享 STATE。
        invalid_plan = tmp_path / 'invalid_plan.json'
        invalid_plan.write_text(json.dumps({'job_id': 'other-job',
                                            'unit_batch': None}))
        monkeypatch.setattr(remote_worker, 'PLAN', invalid_plan)
        with pytest.raises(RuntimeError, match='执行锁'):
            remote_worker.main()

        # B 们什么都没改：STATE/批记录逐字节不变，凭据仍在。
        assert (base / 'mlbridge_state.json').read_bytes() == state_before
        assert record_path.read_bytes() == record_before
        assert (base / 'mlbridge_rclone.conf').is_file()
        assert (base / 'mlbridge_restic_password').is_file()

        release.write_text('go')
        assert proc.wait(timeout=90) == 0
        state = json.loads((base / 'mlbridge_state.json').read_text())
        assert state['phase'] == 'SUCCEEDED'
        outcomes = state['batch']['outcomes']
        assert [outcomes[i['unit_id']]['outcome'] for i in cfg['items']] == \
            ['SUCCEEDED', 'SUCCEEDED']
        record = json.loads(record_path.read_text())
        assert record['status'] == 'succeeded'
        assert record['unit_outcomes'][cfg['items'][0]['unit_id']]['snapshot_id']
    finally:
        if proc.poll() is None:
            (tmp_path / 'workerA_release').write_text('go')
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()


# ---------------------------------------------------------------------------
# 进程组：组活性收尾（抗 TERM 孙进程 / leader 先退出 / restic 泄留）
# ---------------------------------------------------------------------------

GRANDCHILD_CODE = (
    "import os, pathlib, signal, sys, time\n"
    "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
    "pathlib.Path(sys.argv[1]).write_text(str(os.getpid()))\n"
    "pathlib.Path(sys.argv[2]).write_text('ready')\n"
    "time.sleep(60)\n")


def _spawner_command(grandchild_setup_code, pid_file, ready_file):
    """leader：spawn 抗 TERM 孙进程 -> 等就绪握手 -> 按指定行为收尾。"""
    leader = (
        "import pathlib, subprocess, sys, time\n"
        "subprocess.Popen([sys.executable, '-c', sys.argv[1], sys.argv[2],"
        " sys.argv[3]])\n"
        "deadline = time.monotonic() + 30\n"
        "while time.monotonic() < deadline:\n"
        "    if pathlib.Path(sys.argv[3]).exists() and"
        " pathlib.Path(sys.argv[2]).exists():\n"
        "        break\n"
        "    time.sleep(0.02)\n"
        + grandchild_setup_code)
    return [sys.executable, '-c', leader, GRANDCHILD_CODE,
            str(pid_file), str(ready_file)]


def _patch_batch_constants(monkeypatch, grace=0.4, kill=0.4):
    """批模块每次按文件重新加载，所以先把模块加载一次再注入常量。"""
    mod = remote_worker._batch_module()
    monkeypatch.setattr(mod, 'TERMINATION_GRACE', grace)
    monkeypatch.setattr(mod, 'KILL_GRACE', kill)
    monkeypatch.setattr(remote_worker, '_batch_module', lambda: mod)


def test_term_resistant_grandchild_killed_on_command_timeout(tmp_path, monkeypatch):
    """超时：TERM 后抗杀孙进程仍在（leader 无关），升级 KILL 并确认组空。"""
    repo, commit = make_repo(tmp_path)
    pid_file = tmp_path / 'grandchild.pid'
    ready_file = tmp_path / 'grandchild.ready'
    command = _spawner_command("time.sleep(60)\n", pid_file, ready_file)
    plan = make_plan(repo, commit,
                     [make_item(make_spec(repo, commit, DEFAULT_SETUP,
                                          command=command, outputs=['outputs1'],
                                          validator=_command_validator(['true'])),
                                'att-tg', timeout=1.2)],
                     setup=DEFAULT_SETUP)
    ctx = harness(tmp_path, monkeypatch, plan)
    _patch_batch_constants(monkeypatch, 0.4, 0.4)
    try:
        state = ctx['run']()
        assert state['phase'] == 'FAILED'
        failed = batch_outcomes(state)[list(batch_outcomes(state))[0]]
        assert failed['stage'] == 'command' and '超时' in failed['detail']
        assert '已终止并收尾整个进程组' in failed['detail']
        # 同步握手产物：孙进程确实启动过
        assert ready_file.read_text() == 'ready'
        pid = int(pid_file.read_text())
        assert _pid_gone(pid)
    finally:
        if pid_file.exists():
            try:
                os.kill(int(pid_file.read_text()), 9)
            except (OSError, ValueError):
                pass


def test_leader_exit_with_leftover_grandchild_fails_unit(tmp_path, monkeypatch):
    """leader 正常退出但组内遗留抗 TERM 孙进程：确认组空才算干净；
    残留即清理并按失败处理，且绝不进入 validator/backup。"""
    repo, commit = make_repo(tmp_path)
    pid_file = tmp_path / 'grandchild.pid'
    ready_file = tmp_path / 'grandchild.ready'
    after_ready = ("p = pathlib.Path('outputs1'); p.mkdir(parents=True)\n"
                   "(p / 'a.txt').write_text('ONE')\n")
    command = _spawner_command(after_ready, pid_file, ready_file)
    plan = make_plan(repo, commit,
                     [make_item(make_spec(repo, commit, DEFAULT_SETUP,
                                          command=command, outputs=['outputs1'],
                                          validator=_command_validator(['true'])),
                                'att-lg')],
                     setup=DEFAULT_SETUP)
    ctx = harness(tmp_path, monkeypatch, plan)
    _patch_batch_constants(monkeypatch, 0.4, 0.4)
    try:
        state = ctx['run']()
        assert state['phase'] == 'FAILED'
        failed = batch_outcomes(state)[list(batch_outcomes(state))[0]]
        assert failed['stage'] == 'command'
        assert '存活成员' in failed['detail'] and '已清理' in failed['detail']
        # 同步握手：孙进程已启动
        assert ready_file.exists()
        assert _pid_gone(int(pid_file.read_text()))
        backups, _ = restic_calls(ctx['restic_log'])
        assert len(backups) == 0  # 不把带遗留写入者的现场当作成功去备份
    finally:
        if pid_file.exists():
            try:
                os.kill(int(pid_file.read_text()), 9)
            except (OSError, ValueError):
                pass


# ---------------------------------------------------------------------------
# manifest 预算：单文件/多输出总额/时间
# ---------------------------------------------------------------------------

def _plan_multi_outputs_budget(tmp_path, repo, commit, files, limits):
    """写 files{相对路径: 内容} 的单 unit plan；outputs 为各顶层目录。"""
    seen_dirs: set[str] = set()
    cmds = []
    for rel, content in sorted(files.items()):
        dir_part = rel.split('/')[0]
        if dir_part not in seen_dirs:
            cmds.append(f"import pathlib\npathlib.Path({dir_part!r}).mkdir(parents=True)\n")
            seen_dirs.add(dir_part)
        cmds.append(f"import pathlib\npathlib.Path({rel!r}).write_bytes(b'x' * {len(content)})\n")
    outputs = sorted(seen_dirs)
    command = [sys.executable, '-c', "".join(cmds)]
    return make_plan(repo, commit,
                     [make_item(make_spec(repo, commit, DEFAULT_SETUP,
                                          command=command, outputs=outputs,
                                          validator=_command_validator(['true'])),
                                'att-bud')],
                     setup=DEFAULT_SETUP, manifest_limits=limits)


def _plan_single_file_budget(tmp_path, repo, commit, rel, size, limits):
    """单文件声明输出的单 unit plan。"""
    command = [sys.executable, '-c',
               "import pathlib\n"
               f"pathlib.Path({rel!r}).parent.mkdir(parents=True, exist_ok=True)\n"
               f"pathlib.Path({rel!r}).write_bytes(b'x' * {size})\n"]
    return make_plan(repo, commit,
                     [make_item(make_spec(repo, commit, DEFAULT_SETUP,
                                          command=command, outputs=[rel],
                                          validator=_command_validator(['true'])),
                                'att-bud')],
                     setup=DEFAULT_SETUP, manifest_limits=limits)


def test_single_file_output_byte_budget_rejected(tmp_path, monkeypatch):
    """单文件声明输出同样受共享 max_bytes 预算限制。"""
    repo, commit = make_repo(tmp_path)
    plan = _plan_single_file_budget(tmp_path, repo, commit,
                                    'outputs_single/a.bin', 200, {'max_bytes': 100})
    ctx = harness(tmp_path, monkeypatch, plan)
    state = ctx['run']()
    assert state['phase'] == 'FAILED'
    failed = batch_outcomes(state)[list(batch_outcomes(state))[0]]
    assert failed['stage'] == 'outputs_check' and 'max_bytes' in failed['detail']
    backups, _ = restic_calls(ctx['restic_log'])
    assert not backups and not receipt_dirs(ctx['base']).exists()


def test_single_file_output_time_budget_rejected(tmp_path, monkeypatch):
    """哈希逐块检查时间预算：超微预算在首块前拒绝，无静默漏项。"""
    repo, commit = make_repo(tmp_path)
    plan = _plan_single_file_budget(tmp_path, repo, commit,
                                    'outputs_single/a.bin', 16,
                                    {'timeout_seconds': 1e-9})
    ctx = harness(tmp_path, monkeypatch, plan)
    state = ctx['run']()
    assert state['phase'] == 'FAILED'
    failed = batch_outcomes(state)[list(batch_outcomes(state))[0]]
    assert failed['stage'] == 'outputs_check' and '时间预算' in failed['detail']
    backups, _ = restic_calls(ctx['restic_log'])
    assert not backups


def test_multi_output_total_byte_budget_rejected(tmp_path, monkeypatch):
    """目录/多输出共享同一字节预算：各自不超、总额超限也拒绝。"""
    repo, commit = make_repo(tmp_path)
    files = {'o1/g1.bin': 'a' * 60, 'o2/g2.bin': 'b' * 60}
    plan = _plan_multi_outputs_budget(tmp_path, repo, commit, files,
                                      {'max_bytes': 100})
    ctx = harness(tmp_path, monkeypatch, plan)
    state = ctx['run']()
    assert state['phase'] == 'FAILED'
    failed = batch_outcomes(state)[list(batch_outcomes(state))[0]]
    assert failed['stage'] == 'outputs_check' and 'max_bytes' in failed['detail']
    backups, _ = restic_calls(ctx['restic_log'])
    assert not backups


# ---------------------------------------------------------------------------
# 元数据链：command 造父 symlink / unit symlink 均拒绝，目标不被写
# ---------------------------------------------------------------------------

def test_metadata_parent_symlink_by_command_rejected(tmp_path, monkeypatch):
    repo, commit = make_repo(tmp_path)
    outside = tmp_path / 'outside-escape'
    outside.mkdir()
    command = [sys.executable, '-c',
               "import os, pathlib, sys\n"
               f"os.symlink({str(outside)!r}, '.mlbridge-metadata')\n"
               "p = pathlib.Path('outputs1'); p.mkdir(parents=True)\n"
               "(p / 'a.txt').write_text('ONE')\n"]
    spec = make_spec(repo, commit, DEFAULT_SETUP,
                     command=command, outputs=['outputs1'],
                     validator=_command_validator(['true']))
    plan = make_plan(repo, commit, [make_item(spec, 'att-md')], setup=DEFAULT_SETUP)
    ctx = harness(tmp_path, monkeypatch, plan)
    state = ctx['run']()
    assert state['phase'] == 'FAILED'
    failed = batch_outcomes(state)[list(batch_outcomes(state))[0]]
    assert failed['stage'] == 'metadata_check' and '符号链接' in failed['detail']
    # 越界目标保持为空：manifest 从未写到 workspace 之外
    assert list(outside.iterdir()) == []
    backups, _ = restic_calls(ctx['restic_log'])
    assert not backups


def test_metadata_batch_symlink_by_validator_rejected(tmp_path, monkeypatch):
    """validator 造批层 symlink：写入前整链重验拒绝，越界目标保持为空。"""
    repo, commit = make_repo(tmp_path)
    outside = tmp_path / 'outside-batch-escape'
    outside.mkdir()
    validator = [sys.executable, '-c',
                 "import os\n"
                 "os.makedirs('.mlbridge-metadata', exist_ok=True)\n"
                 f"os.symlink({str(outside)!r}, '.mlbridge-metadata/bt-md')\n"]
    command = [sys.executable, '-c', WRITE_ONE]
    spec = make_spec(repo, commit, DEFAULT_SETUP,
                     command=command, outputs=['outputs1'],
                     validator=_command_validator(validator))
    plan = make_plan(repo, commit, [make_item(spec, 'att-md')],
                     setup=DEFAULT_SETUP, batch_id='bt-md')
    ctx = harness(tmp_path, monkeypatch, plan)
    state = ctx['run']()
    assert state['phase'] == 'FAILED'
    failed = batch_outcomes(state)[list(batch_outcomes(state))[0]]
    assert failed['stage'] == 'metadata_check' and '符号链接' in failed['detail']
    assert list(outside.iterdir()) == []
    assert not (receipt_dirs(ctx['base'], 'bt-md')).exists()
    backups, _ = restic_calls(ctx['restic_log'])
    assert not backups


def test_metadata_chain_rejects_unit_symlink_and_special_file(tmp_path):
    """直接对链检查做单测：unit 路径 symlink / 普通文件占位均拒绝。
    （端到端预残留的 workspace 会被 clone 重置，链级在 _check 直接验证。）"""
    mod = remote_worker._batch_module()
    batch, uid = 'bt-md', 'a' * 64
    outside = tmp_path / 'outside-direct'
    outside.mkdir()
    root = tmp_path / 'ws'
    unit_link = root / '.mlbridge-metadata' / batch / uid
    unit_link.parent.mkdir(parents=True)
    os.symlink(str(outside), str(unit_link))
    with pytest.raises(mod.UnitFailure) as exc:
        mod._check_metadata_chain(root, uid, mod._metadata_parts(batch, uid))
    assert exc.value.stage == 'metadata_check'
    assert '符号链接' in exc.value.detail

    plain = tmp_path / 'ws2'
    (plain / '.mlbridge-metadata' / batch).mkdir(parents=True)
    (plain / '.mlbridge-metadata' / batch / uid).write_text('occupied')
    with pytest.raises(mod.UnitFailure) as exc:
        mod._check_metadata_chain(plain, uid, mod._metadata_parts(batch, uid))
    assert '不是普通目录' in exc.value.detail


# ---------------------------------------------------------------------------
# 终态写顺序与失败注入：STATE/批记录/回执一致性，绝不吞错
# ---------------------------------------------------------------------------

def _load_batch_module_once(monkeypatch):
    mod = remote_worker._batch_module()
    monkeypatch.setattr(remote_worker, '_batch_module', lambda: mod)
    return mod


def test_success_record_persistence_failure_keeps_succeeded(tmp_path, monkeypatch):
    """全部成功后批记录终态写失败：不得把 SUCCEEDED 改写成 FAILED，
    逐单元回执是权威；显式上抛并标注控制面持久化不确定，绝不吞掉。"""
    repo, commit = make_repo(tmp_path)
    spec = make_spec(repo, commit, DEFAULT_SETUP,
                     command=[sys.executable, '-c', WRITE_ONE],
                     outputs=['outputs1'],
                     validator=_command_validator([sys.executable, '-c', CHECK_ONE]))
    plan = make_plan(repo, commit, [make_item(spec, 'att-pf1')], setup=DEFAULT_SETUP)
    ctx = harness(tmp_path, monkeypatch, plan)
    mod = _load_batch_module_once(monkeypatch)
    real_update = mod._update_batch_state

    def update(path, *, status, outcomes, detail=None, **kwargs):
        if status == 'succeeded':
            raise mod.BatchError('注入：批记录持久化失败')
        return real_update(path, status=status, outcomes=outcomes,
                           detail=detail, **kwargs)

    monkeypatch.setattr(mod, '_update_batch_state', update)
    with pytest.raises(mod.BatchError, match='持久化失败'):
        ctx['run']()
    state = json.loads((ctx['base'] / 'mlbridge_state.json').read_text())
    assert state['phase'] == 'SUCCEEDED'  # 计算事实不被控制面失败改写
    assert 'control_warning' in state and '持久化' in state['control_warning']
    uid = P.unit_id(spec)
    assert state['batch']['outcomes'][uid]['outcome'] == 'SUCCEEDED'
    assert (receipt_dirs(ctx['base']) / f'{uid}.json').is_file()  # 回执权威
    record = json.loads((ctx['base'] / 'mlbridge_batch_state_bt1.json').read_text())
    assert record['status'] == 'running'  # 终态未持久＝控制面不确定
    assert record['unit_outcomes'][uid]['outcome'] == 'SUCCEEDED'


def test_failure_record_persistence_failure_keeps_failed(tmp_path, monkeypatch):
    """真实 unit 失败后批记录终态写失败：STATE 保持 FAILED 与逐单元
    outcome；错误显式上抛（不吞掉），批记录保留最后进度供对账。"""
    repo, commit = make_repo(tmp_path)
    fail_spec = make_spec(repo, commit, DEFAULT_SETUP,
                          command=[sys.executable, '-c', 'raise SystemExit(7)'],
                          outputs=['outputs2'], validator=_command_validator(['true']))
    third_spec = make_spec(repo, commit, DEFAULT_SETUP,
                           command=[sys.executable, '-c', WRITE_TWO],
                           outputs=['other-tree'], validator=_command_validator(['true']))
    items = [make_item(fail_spec, 'att-pf2'), make_item(third_spec, 'att-pf3')]
    plan = make_plan(repo, commit, items, setup=DEFAULT_SETUP)
    ctx = harness(tmp_path, monkeypatch, plan)
    mod = _load_batch_module_once(monkeypatch)
    real_update = mod._update_batch_state

    def update(path, *, status, outcomes, detail=None, **kwargs):
        if status == 'failed':
            raise mod.BatchError('注入：批记录持久化失败')
        return real_update(path, status=status, outcomes=outcomes,
                           detail=detail, **kwargs)

    monkeypatch.setattr(mod, '_update_batch_state', update)
    with pytest.raises(mod.BatchError, match='持久化失败'):
        ctx['run']()
    state = json.loads((ctx['base'] / 'mlbridge_state.json').read_text())
    assert state['phase'] == 'FAILED'
    assert 'control_warning' in state
    outcomes = state['batch']['outcomes']
    assert outcomes[items[0]['unit_id']]['outcome'] == 'TASK_FAILED'
    assert outcomes[items[1]['unit_id']]['outcome'] == 'NOT_STARTED'
    record = json.loads((ctx['base'] / 'mlbridge_batch_state_bt1.json').read_text())
    assert record['status'] == 'running'
    assert record['units'][items[0]['unit_id']]['outcome'] == 'RUNNING'
    assert not (receipt_dirs(ctx['base']) / f'{items[0]["unit_id"]}.json').exists()


# ---------------------------------------------------------------------------
# 发布诊断：有界 stderr 尾部摘要（不含 token/env），backup 子进程同样清理
# ---------------------------------------------------------------------------

def test_publish_diag_bounded_summary_without_token(tmp_path, monkeypatch):
    repo, commit = make_repo(tmp_path)
    spec = make_spec(repo, commit, DEFAULT_SETUP,
                     command=[sys.executable, '-c', WRITE_ONE],
                     outputs=['outputs1'],
                     validator=_command_validator([sys.executable, '-c', CHECK_ONE]))
    plan = make_plan(repo, commit, [make_item(spec, 'att-nz')], setup=DEFAULT_SETUP,
                     publish_timeout_seconds=20.0)
    ctx = harness(tmp_path, monkeypatch, plan)
    monkeypatch.setenv('FAKE_RESTIC_RESPONSE', 'garbage')
    monkeypatch.setenv('FAKE_RESTIC_NOISE', '1')  # stdout/stderr 各 ~0.9MB
    state = ctx['run']()
    assert state['phase'] == 'FAILED'
    unknown = batch_outcomes(state)[list(batch_outcomes(state))[0]]
    assert unknown['outcome'] == 'PUBLISH_UNKNOWN'
    assert 'tags 对账' in unknown['detail']
    assert len(unknown['detail']) < 4000  # 只保留有界 stderr 尾部
    assert 'token' not in unknown['detail'].lower()
    assert not receipt_dirs(ctx['base']).exists()


_LEFTOVER_SH = """#!/bin/sh
trap '' TERM
echo $$ > "$1"
while :; do sleep 1; done
"""


def test_publish_timeout_kills_leftover_grandchild(tmp_path, monkeypatch):
    """restic backup 子进程泄留抗 TERM 孙进程：KILL 升级有位确认组空。"""
    repo, commit = make_repo(tmp_path)
    spec = make_spec(repo, commit, DEFAULT_SETUP,
                     command=[sys.executable, '-c', WRITE_ONE],
                     outputs=['outputs1'],
                     validator=_command_validator([sys.executable, '-c', CHECK_ONE]))
    plan = make_plan(repo, commit, [make_item(spec, 'att-gc')], setup=DEFAULT_SETUP,
                     publish_timeout_seconds=0.8)
    ctx = harness(tmp_path, monkeypatch, plan)
    leftover = tmp_path / 'term_ignoring_looper.sh'
    leftover.write_text(_LEFTOVER_SH)
    leftover.chmod(0o755)
    gc_pid_file = tmp_path / 'restic_grandchild.pid'
    monkeypatch.setenv('FAKE_RESTIC_SLEEP', '5')
    monkeypatch.setenv('FAKE_RESTIC_LEFTOVER', str(leftover))
    monkeypatch.setenv('FAKE_RESTIC_GC_PID', str(gc_pid_file))
    _patch_batch_constants(monkeypatch, 0.4, 0.4)
    state = ctx['run']()
    assert state['phase'] == 'FAILED'
    unknown = batch_outcomes(state)[list(batch_outcomes(state))[0]]
    assert unknown['outcome'] == 'PUBLISH_UNKNOWN' and '超时' in unknown['detail']
    assert gc_pid_file.exists()
    assert _pid_gone(int(gc_pid_file.read_text()))


# ---------------------------------------------------------------------------
# 终态身份门控：legacy 准备失败/入口失败 vs 残留终态
# ---------------------------------------------------------------------------

def test_legacy_prep_failure_overwrites_residual_success(tmp_path, monkeypatch):
    """有身份的新 legacy 在准备前失败：旧 job 的 SUCCEEDED 绝不被当成本
    job 的结果，也可能残留 snapshot_id 混入。"""
    repo, commit = make_repo(tmp_path)
    base = tmp_path / 'legacy-base'
    base.mkdir()
    (base / 'mlbridge_state.json').write_text(json.dumps(
        {'phase': 'SUCCEEDED', 'job_id': 'old-job', 'snapshot_id': 'a' * 64}))
    (base / 'mlbridge_job.json').write_text(json.dumps({
        'job_id': 'new-job', 'session': 'new-job', 'project': 'demo',
        'repository': str(repo), 'commit': commit,
        'restic_repository': 'rclone:gdrive:x', 'inputs': [], 'setup': [],
        'command': ['true'], 'outputs': []}))
    monkeypatch.setattr(remote_worker, 'BASE', base)
    monkeypatch.setattr(remote_worker, 'STATE', base / 'mlbridge_state.json')
    monkeypatch.setattr(remote_worker, 'PLAN', base / 'mlbridge_job.json')
    with pytest.raises(FileNotFoundError):  # 准备前 _base_env 失败
        remote_worker.main()
    state = json.loads((base / 'mlbridge_state.json').read_text())
    assert state['phase'] == 'FAILED' and state['job_id'] == 'new-job'
    assert 'snapshot_id' not in state  # fresh 覆盖，不继承残留成果字段


def test_batch_prep_failure_overwrites_residual_success(tmp_path, monkeypatch):
    """批模式同规则：残留旧成功不冒充新 job 结果。"""
    repo, commit = make_repo(tmp_path)
    spec = make_spec(repo, commit, DEFAULT_SETUP,
                     command=[sys.executable, '-c', WRITE_ONE],
                     outputs=['outputs1'], validator=_command_validator(['true']))
    plan = make_plan(repo, commit, [make_item(spec, 'att-res')], setup=DEFAULT_SETUP)
    ctx = harness(tmp_path, monkeypatch, plan)
    (ctx['base'] / 'mlbridge_state.json').write_text(json.dumps(
        {'phase': 'SUCCEEDED', 'job_id': 'old-job', 'snapshot_id': 'a' * 64}))
    (ctx['base'] / 'mlbridge_rclone.conf').unlink()  # 准备前 ._base_env 失败
    with pytest.raises(FileNotFoundError):
        ctx['run']()
    state = json.loads((ctx['base'] / 'mlbridge_state.json').read_text())
    assert state['phase'] == 'FAILED' and state['job_id'] == 'bt-job'
    assert 'snapshot_id' not in state


def test_entry_failure_without_identity_never_claims_others_state(tmp_path, monkeypatch):
    """入口失败且身份无法确认：输出错误但不把别的 job STATE 当本 job
    结果，不伪造身份写入。"""
    base = tmp_path / 'entry-base'
    base.mkdir()
    (base / 'mlbridge_state.json').write_text(json.dumps(
        {'phase': 'SUCCEEDED', 'job_id': 'old-job', 'snapshot_id': 'a' * 64}))
    (base / 'mlbridge_job.json').write_text('{not json')
    monkeypatch.setattr(remote_worker, 'BASE', base)
    monkeypatch.setattr(remote_worker, 'STATE', base / 'mlbridge_state.json')
    monkeypatch.setattr(remote_worker, 'PLAN', base / 'mlbridge_job.json')
    before = (base / 'mlbridge_state.json').read_bytes()
    with pytest.raises(ValueError):
        remote_worker.main()
    assert (base / 'mlbridge_state.json').read_bytes() == before


# ---------------------------------------------------------------------------
# 心跳生命周期：main 结束不残留 daemon 心跳线程
# ---------------------------------------------------------------------------

def test_no_heartbeat_thread_leaks_after_main(tmp_path, monkeypatch):
    import threading
    repo, commit = make_repo(tmp_path)
    base = tmp_path / 'hb-base'
    base.mkdir()
    (base / 'mlbridge_rclone.conf').write_text('[gdrive]\ntype=drive\n')
    (base / 'mlbridge_restic_password').write_text('test')
    (base / 'mlbridge_job.json').write_text(json.dumps({
        'job_id': 'hb', 'session': 'hb', 'project': 'demo',
        'repository': str(repo), 'commit': commit,
        'restic_repository': 'rclone:gdrive:x',
        'inputs': [], 'setup': [],
        'command': [sys.executable, '-c', 'pass'], 'outputs': []}))
    fakebin = tmp_path / 'hb-fakebin'
    fakebin.mkdir()
    (fakebin / 'restic').write_text('#!/bin/sh\necho "restic 0.17.0"\n')
    (fakebin / 'restic').chmod(0o755)
    (fakebin / 'rclone').write_text('#!/bin/sh\nexit 0\n')
    (fakebin / 'rclone').chmod(0o755)
    monkeypatch.setenv('PATH', str(fakebin) + os.pathsep + os.environ['PATH'])
    monkeypatch.setattr(remote_worker, 'BASE', base)
    monkeypatch.setattr(remote_worker, 'STATE', base / 'mlbridge_state.json')
    monkeypatch.setattr(remote_worker, 'PLAN', base / 'mlbridge_job.json')
    monkeypatch.setattr(remote_worker, 'HEARTBEAT_INTERVAL', 0.05)

    def alive():
        return [t for t in threading.enumerate()
                if t.name == 'mlbridge-heartbeat' and t.is_alive()]

    remote_worker.main()
    assert alive() == []
    # 第二次调用：主 finally 已清理凭据，重新补齐以便完整跑完；
    # 依旧不残留心跳 daemon。
    (base / 'mlbridge_rclone.conf').write_text('[gdrive]\ntype=drive\n')
    (base / 'mlbridge_restic_password').write_text('test')
    remote_worker.main()
    assert alive() == []


def test_concurrent_legacy_worker_refused_and_cleanup_owner_only(tmp_path, monkeypatch):
    """legacy 同样受唯一 BASE 执行锁约束：执行中拒绝第二个 worker，
    不写/不删执行中 worker 的任何共享文件。"""
    repo, commit = make_repo(tmp_path)
    started = tmp_path / 'legacyA_started'
    release = tmp_path / 'legacyA_release'
    legacy_plan = {
        'job_id': 'leg-dup', 'session': 'leg-dup', 'project': 'demo',
        'repository': str(repo), 'commit': commit,
        'restic_repository': 'rclone:gdrive:store/demo',
        'inputs': [], 'setup': [],
        'command': [sys.executable, '-c', _BLOCKER, str(started), str(release)],
        'outputs': []}
    base = tmp_path / 'legacy-root'
    base.mkdir()
    (base / 'mlbridge_rclone.conf').write_text('[gdrive]\ntype=drive\n')
    (base / 'mlbridge_restic_password').write_text('test')
    (base / 'mlbridge_job.json').write_text(json.dumps(legacy_plan))
    fakebin = tmp_path / 'legacy-fakebin'
    fakebin.mkdir()
    restic = fakebin / 'restic'
    restic.write_text(FAKE_RESTIC)
    restic.chmod(0o755)
    rclone = fakebin / 'rclone'
    rclone.write_text('#!/bin/sh\nexit 0\n')
    rclone.chmod(0o755)
    count = tmp_path / 'legacy-restic-count.txt'
    count.write_text('0')
    restic_log = tmp_path / 'legacy-restic.log'
    restic_log.write_text('')
    monkeypatch.setenv('PATH', str(fakebin) + os.pathsep + os.environ['PATH'])
    monkeypatch.setenv('FAKE_RESTIC_LOG', str(restic_log))
    monkeypatch.setenv('FAKE_RESTIC_COUNT', str(count))
    monkeypatch.setattr(remote_worker, 'BASE', base)
    monkeypatch.setattr(remote_worker, 'STATE', base / 'mlbridge_state.json')
    monkeypatch.setattr(remote_worker, 'PLAN', base / 'mlbridge_job.json')

    proc = _spawn_resident(tmp_path, base)
    try:
        _wait_for_file(started)
        state_before = (base / 'mlbridge_state.json').read_bytes()
        assert (base / 'mlbridge_rclone.conf').is_file()
        assert (base / 'mlbridge_restic_password').is_file()
        # 第二个 legacy worker：锁处非阻塞拒绝，无任何共享写入
        with pytest.raises(RuntimeError, match='执行锁'):
            remote_worker.main()
        assert (base / 'mlbridge_state.json').read_bytes() == state_before
        assert (base / 'mlbridge_rclone.conf').is_file()
        assert (base / 'mlbridge_restic_password').is_file()
        release.write_text('go')
        assert proc.wait(timeout=90) == 0
        state = json.loads((base / 'mlbridge_state.json').read_text())
        assert state['phase'] == 'SUCCEEDED' and state['job_id'] == 'leg-dup'
    finally:
        if proc.poll() is None:
            (tmp_path / 'legacyA_release').write_text('go')
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
