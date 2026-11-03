"""Per-key checkpoint 接口的离线测试：fake restic + 假 plan + 真实 subprocess。

不访问网络、不依赖 Colab。worker 的 VM 根路径用 MLBRIDGE_WORKER_BASE 环
境变量隔离（真实 VM 上未设置，缺省 /content）；协议环境通过
MLBRIDGE_CHECKPOINT_PROTOCOL / MLBRIDGE_CHECKPOINT_WORKER 显式提供，等价
于 worker main() 在 plan.command 环境里导出的契约。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from mlbridge import remote_worker

WORKER_FILE = Path(__file__).resolve().parents[1] / 'src/mlbridge/remote_worker.py'
PY = sys.executable

FAKE_RESTIC = r'''#!/bin/sh
sub=""
for a in "$@"; do
  case "$a" in backup|restore|version) sub="$a"; break;; esac
done
printf '%s\n' "$*" >> "${FAKE_RESTIC_LOG:?required}"
case "$sub" in
  version) echo "restic ${FAKE_RESTIC_VERSION:-0.17.0} compiled with go1.22.5";;
  restore) exit 0;;
  backup)
    sleep "${FAKE_RESTIC_SLEEP:-0}"
    n=$(( $(cat "$FAKE_RESTIC_LOG.count" 2>/dev/null || echo 0) + 1 ))
    echo "$n" > "$FAKE_RESTIC_LOG.count"
    sid=$(printf 'a1%062x' "$n")
    echo "SNAPSHOT=$sid" >> "${FAKE_RESTIC_LOG:?required}"
    case "${FAKE_RESTIC_MODE:-ok}" in
      fail)
        echo "restic: fatal: unable to open config fileserter" >&2
        exit 1;;
      partial|nonzero-summary)
        echo '{"message_type":"summary","snapshot_id":"'"$sid"'"}'
        echo "RESTIC_PASSWORD=${RESTIC_PASSWORD:-none} partial source read" >&2
        if [ "$FAKE_RESTIC_MODE" = partial ]; then exit 3; else exit 1; fi;;
      partial-no-summary)
        echo 'incomplete source read' >&2
        exit 3;;
      no-summary)
        echo '{"message_type":"status","duration by": 1}'
        exit 0;;
      id-int)
        echo '{"message_type":"summary","snapshot_id": 42}'
        exit 0;;
      id-short)
        echo '{"message_type":"summary","snapshot_id": "abcd1234"}'
        exit 0;;
      id-upper)
        echo '{"message_type":"summary","snapshot_id": "'"$(printf 'A1%058dX' 1)"'"}'
        exit 0;;
      ok|*)
        echo '{"message_type":"summary","snapshot_id":"'"$sid"'"}'
        exit 0;;
    esac;;
esac
exit 0
'''


def _make_session(tmp_path, *, restic_mode='ok'):
    base = tmp_path / 'vm'
    ws = base / 'mlbridge-workspace'
    root = ws / 'session' / 'one'
    root.mkdir(parents=True)
    plan = {
        'job_id': 'sess-1', 'session': 'sess-1', 'project': 'demo',
        'repository': 'https://github.com/o/r',
        'commit': 'a' * 40,
        'restic_repository': 'rclone:gdrive:store/demo',
        'inputs': [], 'setup': [],
        'command': ['python', 'train.py'],
        'outputs': ['session/one'],
    }
    (base / 'mlbridge_job.json').write_text(json.dumps(plan))
    (base / 'mlbridge_rclone.conf').write_text('[gdrive]\ntype=drive\n')
    (base / 'mlbridge_restic_password').write_text('not-a-secret')
    fakebin = tmp_path / 'fakebin'
    fakebin.mkdir(exist_ok=True)
    (fakebin / 'restic').write_text(FAKE_RESTIC)
    (fakebin / 'restic').chmod(0o755)
    (fakebin / 'rclone').write_text('#!/bin/sh\nexit 0\n')
    (fakebin / 'rclone').chmod(0o755)
    restic_log = tmp_path / 'restic.log'
    restic_log.write_text('')
    env = dict(os.environ)
    env.update(
        MLBRIDGE_WORKER_BASE=str(base),
        PATH=str(fakebin) + os.pathsep + env['PATH'],
        FAKE_RESTIC_LOG=str(restic_log),
        FAKE_RESTIC_MODE=restic_mode,
        MLBRIDGE_CHECKPOINT_PROTOCOL='v1',
        MLBRIDGE_CHECKPOINT_WORKER=str(WORKER_FILE.resolve()),
    )
    return {'base': base, 'ws': ws, 'root': root, 'env': env,
            'plan': plan, 'restic_log': restic_log}


def _request(s, *, key='row-p1', protocol='v1', commit=None, session='sess-1',
             project='demo', roots=None, metadata=None, response=None,
             provenance=None, extra=None):
    root = s['root'] / key
    root.mkdir(exist_ok=True)
    for name in ('model', 'samples', 'metrics'):
        (root / name).mkdir(exist_ok=True)
        (root / name / 'done.marker').write_text('done\n')
        (root / name / 'artifact.bin').write_bytes(f'{key}:{name}'.encode())
    (s['root'] / f'{key}-pack-metadata.json').write_text(json.dumps(
        {'pins': {'sha256_of_model': 'f' * 64, 'seed': 7},
         'backend': 'Colab', 'gpu': 'whatever-server-provides'}))
    req = {
        'protocol': protocol,
        'project': project,
        'session': session,
        'commit': s['plan']['commit'] if commit is None else commit,
        'key': key,
        'roots': roots if roots is not None else [
            f'session/one/{key}/model', f'session/one/{key}/samples',
            f'session/one/{key}/metrics'],
        'metadata': metadata if metadata is not None else
            f'session/one/{key}-pack-metadata.json',
        'response': response if response is not None else
            f'session/one/{key}-response.json',
        'provenance': provenance if provenance is not None else
            {'backend': 'Colab', 'host': 'gridpure-batch-n', 'gpu': 'assigned-by-colab',
             'catalog_gpus_used': 'A100-or-L4-or-CPU-decided-elsewhere'},
    }
    if extra is not None:
        req.update(extra)
    path = s['root'] / f'{key}-request.json'
    path.write_text(json.dumps(req))
    return path, req


def _run(s, request_path, *extra_args):
    return subprocess.run(
        [PY, str(WORKER_FILE), 'checkpoint', '--request', str(request_path),
         *extra_args], capture_output=True, text=True, env=s['env'], timeout=120)


def _backup_calls(s):
    return [ln for ln in s['restic_log'].read_text().splitlines()
            if ' backup ' in ln or ln.startswith('restic -r rclone')]


def test_success_writes_metadata_full64_and_atomic_response(tmp_path):
    s = _make_session(tmp_path)
    request_path, req = _request(s)
    done = _run(s, request_path)
    assert done.returncode == 0, done.stderr
    response = json.loads((s['ws'] / req['response']).read_text())
    assert response['phase'] == 'SUCCEEDED'
    assert re.fullmatch(r'[a-f0-9]{64}', response['snapshot_id'])
    assert response['session'] == 'sess-1' and response['commit'] == 'a' * 40
    assert response['checkpoint_key'] == 'row-p1'
    assert response['requested_roots'] == req['roots']
    meta_bytes = (s['ws'] / req['metadata']).read_bytes()
    assert response['metadata']['sha256'] == hashlib.sha256(meta_bytes).hexdigest()
    assert response['metadata']['bytes'] == len(meta_bytes)
    assert response['provenance'] == req['provenance']
    # 恰好一次 backup，tag 方案完整
    calls = _backup_calls(s)
    assert len(calls) == 1
    for tag in ('mlbridge-project-demo', 'mlbridge-session-checkpoint-sess-1',
                'mlbridge-checkpoint-v1',
                'mlbridge-checkpoint-key-' + hashlib.sha256(b'row-p1').hexdigest(),
                'mlbridge-checkpoint-meta-'):
        assert tag in calls[0]
    assert req['response'].split('/')[-1] not in calls[0]  # response 不被自己快照
    # marker 终态；凭据文件不被 checkpoint 流程删除（job 仍在运行）
    marker = json.loads((s['base'] / 'mlbridge_checkpoint' / (
        hashlib.sha256(b'row-p1').hexdigest() + '.json')).read_text())
    assert marker['status'] == 'SUCCEEDED' and marker['snapshot_id'] == response['snapshot_id']
    assert (s['base'] / 'mlbridge_rclone.conf').exists()
    assert (s['base'] / 'mlbridge_restic_password').exists()
    # content response 文件是普通文件（原子 replace 写入）
    assert (s['ws'] / req['response']).is_file()


def test_same_key_real_process_race_performs_one_backup(tmp_path):
    s = _make_session(tmp_path)
    req, _ = _request(s)
    # Force both processes past the old check-then-act guard. Only an atomic
    # marker claim after this barrier can prevent two real backup calls.
    barrier = s['base'] / 'barrier'
    barrier.mkdir()
    script = s['base'] / 'race.py'
    script.write_text(
        'import os,sys,time\nfrom pathlib import Path\n'
        f'sys.path.insert(0,{str(WORKER_FILE.parents[1])!r})\n'
        'from mlbridge import remote_worker as w\n'
        'original=w._assert_no_prior_checkpoint\n'
        'def wait_together(key,response):\n'
        ' original(key,response)\n'
        f' directory=Path({str(barrier)!r})\n'
        ' (directory/str(os.getpid())).write_text("ready")\n'
        ' end=time.monotonic()+10\n'
        ' while len(list(directory.iterdir()))<2:\n'
        '  if time.monotonic()>end: raise RuntimeError("barrier timeout")\n'
        '  time.sleep(.01)\n'
        'w._assert_no_prior_checkpoint=wait_together\n'
        'raise SystemExit(w.main(["checkpoint","--request",sys.argv[1]]))\n')
    children = [subprocess.Popen([PY, str(script), str(req)], env=s['env'],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                for _ in range(2)]
    results = []
    try:
        for p in children:
            stdout, stderr = p.communicate(timeout=20)
            results.append((p.returncode, stderr))
    finally:
        for p in children:
            if p.poll() is None:
                p.kill()
                p.communicate(timeout=5)
    assert sorted(rc for rc, _ in results) == [0, 3], results
    assert len(_backup_calls(s)) == 1


@pytest.mark.parametrize('mode, candidate', [('partial', True), ('nonzero-summary', True),
                                            ('partial-no-summary', False)])
def test_nonzero_with_snapshot_evidence_is_ambiguous(tmp_path, mode, candidate):
    s = _make_session(tmp_path, restic_mode=mode)
    s['env']['RESTIC_PASSWORD'] = 'sensitive-test-string'
    reqpath, req = _request(s)
    done = _run(s, reqpath)
    assert done.returncode == 5, (done.stdout, done.stderr)
    body = json.loads((s['ws'] / req['response']).read_text())
    assert body['phase'] == 'AMBIGUOUS'
    assert body['restic_exit_code'] == (1 if mode == 'nonzero-summary' else 3)
    assert 'snapshot_id' not in body  # not a complete, accepted publication
    assert ('candidate_snapshot_id' in body) == candidate
    if candidate:
        assert re.fullmatch('[0-9a-f]{64}', body['candidate_snapshot_id'])
    assert 'sensitive-test-string' not in json.dumps(body)
    assert _run(s, reqpath).returncode == 3
    assert len(_backup_calls(s)) == 1


def test_response_write_failure_updates_marker_without_rebackup(tmp_path):
    s = _make_session(tmp_path)
    reqpath, req = _request(s)
    script = s['base'] / 'response-failure.py'
    script.write_text(
        'import sys\n'
        f'sys.path.insert(0,{str(WORKER_FILE.parents[1])!r})\n'
        'from mlbridge import remote_worker as w\n'
        'real=w._atomic_json\n'
        'def fail_response(path,body,**kw):\n'
        ' if "response.json" in str(path): raise OSError("write failure")\n'
        ' return real(path,body,**kw)\n'
        'w._atomic_json=fail_response\n'
        'raise SystemExit(w.main(["checkpoint","--request",sys.argv[1]]))\n')
    done = subprocess.run([PY, str(script), str(reqpath)], env=s['env'],
                          capture_output=True, text=True, timeout=20)
    assert done.returncode == 5
    marker = json.loads((s['base'] / 'mlbridge_checkpoint' / (
        hashlib.sha256(req['key'].encode()).hexdigest() + '.json')).read_text())
    assert marker['status'] == marker['phase'] == 'AMBIGUOUS'
    assert re.fullmatch('[0-9a-f]{64}', marker['candidate_snapshot_id'])
    assert not (s['ws'] / req['response']).exists()
    assert _run(s, reqpath).returncode == 3
    assert len(_backup_calls(s)) == 1


def test_response_inside_backed_up_root_refused(tmp_path):
    s = _make_session(tmp_path)
    reqpath, req = _request(s, response='session/one/row-p1/model/response.json')
    assert _run(s, reqpath).returncode == 2
    assert _backup_calls(s) == []


def test_conflicting_snapshot_summaries_do_not_choose_latest():
    output = '\n'.join(json.dumps({'message_type': 'summary', 'snapshot_id': value * 64})
                       for value in ('a', 'b'))
    assert remote_worker._summary_snapshot_id(output) is None


def test_checkpoint_env_vars_exported_to_command_by_worker_main(tmp_path, monkeypatch):
    """worker main 必须在 command 环境里显式导出协议版本与 worker 路径。"""
    s = _make_session(tmp_path)
    probe = s['ws'] / 'probe-marker'
    s['plan']['command'] = [
        PY, '-c',
        'import os,sys;'
        "open(sys.argv[1],'w').write("
        "os.environ.get('MLBRIDGE_CHECKPOINT_PROTOCOL','MISSING')+'\\n'+"
        "os.environ.get('MLBRIDGE_CHECKPOINT_WORKER','MISSING'))",
        str(probe)]
    (s['base'] / 'mlbridge_job.json').write_text(json.dumps(s['plan']))
    monkeypatch.setattr(remote_worker, 'BASE', s['base'])
    monkeypatch.setattr(remote_worker, 'STATE', s['base'] / 'mlbridge_state.json')
    monkeypatch.setattr(remote_worker, 'PLAN', s['base'] / 'mlbridge_job.json')
    monkeypatch.setattr(remote_worker, 'CHECKPOINT_LOG',
                        s['base'] / 'mlbridge_checkpoint.log')
    monkeypatch.setenv('MLBRIDGE_WORKER_BASE', str(s['base']))
    # fake PATH + fake restic 的必需变量要真正进入 os.environ，否则 run_job
    # 的 child env 取不到，fake restic 会因缺 FAKE_RESTIC_LOG 退出。
    monkeypatch.setenv('PATH', s['env']['PATH'])
    monkeypatch.setenv('FAKE_RESTIC_LOG', str(s['restic_log']))
    monkeypatch.setenv('FAKE_RESTIC_MODE', 'ok')
    # 不真实 clone https://github.com/o/r（离线）：直接建好 workspace。
    monkeypatch.setattr(remote_worker, 'clone_repository',
                        lambda repository, commit, root:
                        root.mkdir(parents=True, exist_ok=True))
    remote_worker.main()
    exported = probe.read_text()
    assert exported.splitlines()[0] == 'v1'
    assert Path(exported.splitlines()[1]).resolve() == WORKER_FILE.resolve()


@pytest.mark.parametrize('mode', ['fail', 'no-summary', 'id-int', 'id-short', 'id-upper'])
def test_failure_timeout_and_malformed_summary(tmp_path, mode):
    s = _make_session(tmp_path, restic_mode=mode)
    request_path, req = _request(s, key=f'row-{mode}')
    done = _run(s, request_path)
    if mode == 'fail':
        assert done.returncode == 4
    else:
        assert done.returncode == 5, (done.stdout, done.stderr)
    body = json.loads((s['ws'] / req['response']).read_text())
    if mode == 'fail':
        assert body['phase'] == 'FAILED'
        assert 'unable to open config' in body['restic_log_excerpt']
        assert 'rclone:gdrive:store/demo' not in body['restic_log_excerpt']
    else:
        assert body['phase'] == 'AMBIGUOUS'
        assert 'snapshot_id' not in body
    body_ok = json.loads((s['ws'] / req['response']).read_text())
    marker = json.loads((s['base'] / 'mlbridge_checkpoint' / (
        hashlib.sha256(f'row-{mode}'.encode()).hexdigest() + '.json')).read_text())
    assert marker['status'] == body_ok['phase']


def test_backup_timeout_is_ambiguous_and_never_retried(tmp_path):
    s = _make_session(tmp_path)
    s['env']['FAKE_RESTIC_SLEEP'] = '3'
    request_path, req = _request(s, key='row-slow')
    done = _run(s, request_path, '--timeout', '1')
    assert done.returncode == 5
    body = json.loads((s['ws'] / req['response']).read_text())
    assert body['phase'] == 'AMBIGUOUS'
    assert '超时' in body['error']
    assert len(_backup_calls(s)) == 1  # 不自动重试


def test_second_checkpoint_failure_keeps_first_succeeded(tmp_path):
    """p1 成功后 p2 失败、随后会话失败都不会破坏 p1 的独立可用留证。"""
    s = _make_session(tmp_path)
    r1, req1 = _request(s, key='row-p1')
    assert _run(s, r1).returncode == 0
    s['env']['FAKE_RESTIC_MODE'] = 'fail'
    r2, req2 = _request(s, key='row-p2')
    assert _run(s, r2).returncode == 4
    # 第一份 response 与 marker 仍完整独立可读
    first = json.loads((s['ws'] / req1['response']).read_text())
    assert first['phase'] == 'SUCCEEDED'
    second = json.loads((s['ws'] / req2['response']).read_text())
    assert second['phase'] == 'FAILED' and second['error']
    marker1 = json.loads((s['base'] / 'mlbridge_checkpoint' / (
        hashlib.sha256(b'row-p1').hexdigest() + '.json')).read_text())
    assert marker1['status'] == 'SUCCEEDED'
    assert len(_backup_calls(s)) == 2  # 一次/人，不因失败重复


def test_same_key_rerun_refused_no_second_backup(tmp_path):
    s = _make_session(tmp_path)
    r1, req1 = _request(s, key='row-p1')
    assert _run(s, r1).returncode == 0
    calls_before = len(_backup_calls(s))
    done = _run(s, r1)
    assert done.returncode == 3
    assert '已有 checkpoint 记录' in done.stderr
    assert len(_backup_calls(s)) == calls_before  # 绝不重复 backup


def test_crash_window_between_backup_and_response_refuses_rebackup(tmp_path):
    """成功 backup 后、原子 response 前崩溃：同 key 重调必须拒绝，走重发现。"""
    s = _make_session(tmp_path)
    request_path, req = _request(s, key='row-p1')
    env = dict(s['env'])
    env['MLBRIDGE_WORKER_BASE'] = str(s['base'])

    # 一次性动态脚本：调用真实 worker 的 run path，在 response 落盘处崩溃
    crash_script = (
        'import os,sys\n'
        # 必须 insert 包父目录 src，而非包目录 src/mlbridge 本身
        f'sys.path.insert(0, {str(WORKER_FILE.parents[1])!r})\n'
        'from mlbridge import remote_worker as W\n'
        'real = W._atomic_json\n'
        'def crash_after_backup(path, body, **kw):\n'
        "    if 'response.json' in str(path):\n"
        '        os._exit(9)\n'                      # backup 完成后、response 落盘前死亡
        '    return real(path, body, **kw)\n'
        'W._atomic_json = crash_after_backup\n'
        "raise SystemExit(W.main(['checkpoint','--request',sys.argv[1]]))\n"
    )
    script = s['base'] / 'crash_runner.py'
    script.write_text(crash_script)
    done = subprocess.run([PY, str(script), str(request_path)],
                          capture_output=True, text=True, env=env, timeout=120)
    assert done.returncode == 9
    assert not (s['ws'] / req['response']).exists()
    assert len(_backup_calls(s)) == 1  # backup 已发生
    # 同 key 重调：拒绝，不做第二次 backup
    done2 = _run(s, request_path)
    assert done2.returncode == 3
    assert len(_backup_calls(s)) == 1


def test_other_key_not_blocked_by_a_pending_marker(tmp_path):
    s = _make_session(tmp_path)
    rh = hashlib.sha256(b'row-p1').hexdigest()
    pending = s['base'] / 'mlbridge_checkpoint' / f'{rh}.json'
    pending.parent.mkdir(parents=True)
    pending.write_text(json.dumps({'status': 'PENDING', 'session': 'sess-1'}))
    r1, req1 = _request(s, key='row-p1')
    assert _run(s, r1).returncode == 3  # 同 key 仍拒
    r2, req2 = _request(s, key='row-p2')
    done2 = _run(s, r2)
    assert done2.returncode == 0  # 不同 key 独立


def test_env_guard_missing_protocol(tmp_path):
    s = _make_session(tmp_path)
    s['env'].pop('MLBRIDGE_CHECKPOINT_PROTOCOL')
    request_path, req = _request(s, key='row-x')
    done = _run(s, request_path)
    assert done.returncode == 2
    assert 'MLBRIDGE_CHECKPOINT_PROTOCOL' in done.stderr
    assert not (s['ws'] / req['response']).exists()
    assert _backup_calls(s) == []


def test_env_guard_worker_mismatch(tmp_path):
    s = _make_session(tmp_path)
    s['env']['MLBRIDGE_CHECKPOINT_WORKER'] = '/content/other.py'
    request_path, req = _request(s, key='row-x')
    done = _run(s, request_path)
    assert done.returncode == 2
    assert _backup_calls(s) == []


@pytest.mark.parametrize('mutate, needle', [
    ({'commit': 'b' * 40}, 'commit'),
    ({'session': 'sess-9'}, 'session'),
    ({'project': 'other'}, 'project'),
    ({'protocol': 'v2'}, 'protocol'),
    ({'key': ''}, 'key'),  # key 本体也是请求身份：非法值同样 fail closed
])
def test_identity_mismatch_refused(tmp_path, mutate, needle):
    s = _make_session(tmp_path)
    request_path, req = _request(s, key='row-x', extra=mutate)
    done = _run(s, request_path)
    assert done.returncode == 2 and needle in done.stderr
    assert _backup_calls(s) == []


def test_roots_outside_authorized_outputs_refused(tmp_path):
    s = _make_session(tmp_path)
    outside = s['ws'] / 'other'
    outside.mkdir()
    (outside / 'f.bin').write_bytes(b'x')
    request_path, req = _request(
        s, key='row-x', roots=['other', 'session/one/row-x/samples',
                               'session/one/row-x/metrics'])
    done = _run(s, request_path)
    assert done.returncode == 2
    assert '授权' in done.stderr
    assert _backup_calls(s) == []


def test_parent_escape_via_dots_refused(tmp_path):
    s = _make_session(tmp_path)
    request_path, req = _request(
        s, key='row-x',
        roots=['session/one/row-x/model', 'session/one/row-x/samples',
               'session/one/../../../etc'])
    done = _run(s, request_path)
    assert done.returncode == 2
    assert _backup_calls(s) == []


def test_symlink_root_and_alias_path_refused(tmp_path):
    s = _make_session(tmp_path / 'symlink-case')
    alias = s['root'] / 'row-p1' / 'mirror'
    alias.mkdir(parents=True)  # 父 row-p1 此时还未由 _request 创建
    (s['root'] / 'row-p1' / 'to-session-root').symlink_to(
        s['ws'] / 'session' / 'one', target_is_directory=True)
    request_path, req = _request(
        s, key='row-p1',
        roots=['session/one/row-p1/to-session-root', 'session/one/row-p1/samples',
               'session/one/row-p1/metrics'])
    done = _run(s, request_path)
    assert done.returncode == 2
    assert '软链接' in done.stderr
    assert _backup_calls(s) == []
    # 响应路径穿过软链接祖先同样拒绝
    request_path, req = _request(s, key='row-y',
                                 response='session/one/row-p1/to-session-root/resp.json')
    done = _run(s, request_path)
    assert done.returncode == 2
    assert _backup_calls(s) == []


def test_metadata_rules_refused(tmp_path):
    s = _make_session(tmp_path / 'meta-case')
    # metadata 缺失
    request_path, req = _request(s, key='row-m1',
                                 metadata='session/one/missing.json')
    assert _run(s, request_path).returncode == 2
    # metadata 是目录
    request_path, req = _request(s, key='row-m2', metadata='session/one/row-m2')
    assert _run(s, request_path).returncode == 2
    # metadata 是软链接（指向本 session 内已存在的普通文件，则拒绝必来自软链本身）
    link = s['root'] / 'row-m3' / 'meta-link.json'
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(s['root'] / 'row-m1-pack-metadata.json')
    request_path, req = _request(s, key='row-m3',
                                 metadata='session/one/row-m3/meta-link.json')
    assert _run(s, request_path).returncode == 2
    # metadata 超过 4MiB：在 _request 之后再写大内容，否则会被默认 metadata 覆盖
    request_path, req = _request(s, key='row-m4')
    (s['root'] / 'row-m4-pack-metadata.json').write_bytes(
        b'x' * (4 * 1024 * 1024 + 1))
    assert _run(s, request_path).returncode == 2
    assert _backup_calls(s) == []


def test_git_component_and_unknown_fields_refused(tmp_path):
    s = _make_session(tmp_path / 'git-case')
    request_path, req = _request(
        s, key='row-g', roots=['session/one/.git/objects',
                               'session/one/row-g/samples',
                               'session/one/row-g/metrics'])
    assert _run(s, request_path).returncode == 2
    # 未知字段 fail closed（_make_session 不能对同一 base 重复 mkdir）
    s2 = _make_session(tmp_path / 'unknown-field-case')
    request_path, req = _request(s2, key='row-u',
                                 extra={'driver_version': '0.9'})
    done = _run(s2, request_path)
    assert done.returncode == 2
    assert '未知字段' in done.stderr


def test_provenance_required_and_freeform(tmp_path):
    """provenance 必须显式提供且类型受限；值不写死 GPU 型号（原样透传）。"""
    s = _make_session(tmp_path / 'prov-missing')
    request_path, req = _request(s, key='row-nov', extra=None)
    request_path.write_text(json.dumps(
        {k: v for k, v in json.loads(request_path.read_text()).items()
         if k != 'provenance'}))
    done = _run(s, request_path)
    assert done.returncode == 2
    assert '缺少字段' in done.stderr
    # 任意字符串值（包括 CPU-only 场景）都是合法 provenance
    s2 = _make_session(tmp_path / 'prov-cpu')
    request_path, req = _request(s2, key='row-cpu',
                                 provenance={'gpu': 'CPU', 'host': 'some-cpu-host'})
    done = _run(s2, request_path)
    assert done.returncode == 0
    response = json.loads((s2['ws'] / req['response']).read_text())
    assert response['provenance'] == {'gpu': 'CPU', 'host': 'some-cpu-host'}
