"""恢复台账的聚焦测试：状态机、持久化、并发与 fail closed 行为。

全部离线：不读取凭据、不联网；并发用小子进程（flock 串行化）验证，
不使用 sleep 等待。
"""
from __future__ import annotations

import json
import hashlib
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from mlbridge import ledger as L
from mlbridge import protocol as P

SRC_ROOT = str(Path(__file__).resolve().parents[1] / "src")
PROJECT = "alpha"


def unit_raw(**overrides):
    base = {
        "schema_version": 2,
        "project": PROJECT,
        "repository": "https://github.com/user/repo.git",
        "commit": "a" * 40,
        "inputs": {"train": "b" * 64},
        "environment": {"python": "3.11.8"},
        "command": ["python", "train.py"],
        "params": {"lr": 1e-3},
        "outputs": ["outputs/"],
        "validator": {"kind": "metric-threshold"},
    }
    base.update(overrides)
    return base


@pytest.fixture
def state_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    return tmp_path / "state"


@pytest.fixture
def ledger(state_dir):
    return L.Ledger(PROJECT)


@pytest.fixture
def unit_id(ledger):
    return ledger.plan(unit_raw())


VALIDATOR_SPEC = {"kind": "metric-threshold"}
EXPECTED_VALIDATOR_SPEC_SHA256 = hashlib.sha256(json.dumps(
    VALIDATOR_SPEC, sort_keys=True, ensure_ascii=False,
    separators=(",", ":")).encode()).hexdigest()

RESULT_KW = dict(snapshot="c" * 64, namespace="rclone:gdrive:data/alpha",
                 manifest_sha256="d" * 64,
                 validator_spec_sha256=EXPECTED_VALIDATOR_SPEC_SHA256,
                 validator_result={"kind": "metric-threshold", "status": "passed"})


def _succeed(ledger, uid, attempt="att-1", session="sess-1"):
    ledger.start_attempt(uid, attempt, session)
    return ledger.reconcile(uid, attempt, session,
                            {"source": "colab-observe",
                             "terminal_state": "SUCCEEDED"})


def _ledger_path(state_dir):
    return state_dir / "mlbridge" / "units" / f"{PROJECT}.json"


def _child(code, state_dir, argv):
    """在独立进程里跑一段台账调用；返回 (stdout, returncode)。"""
    proc = subprocess.run([sys.executable, "-c", code, *argv], capture_output=True,
                          text=True, env={**os.environ,
                                          "PYTHONPATH": SRC_ROOT,
                                          "XDG_STATE_HOME": str(state_dir)})
    return (proc.stdout.strip() or f"STDERR:{proc.stderr.strip()}", proc.returncode)


# ---- 身份与登记 ---------------------------------------------------------
def test_plan_is_idempotent_and_persists_pending(ledger, unit_id, state_dir):
    assert ledger.plan(unit_raw()) == unit_id
    units = json.loads(_ledger_path(state_dir).read_text())["units"]
    assert len(units) == 1
    restarted = L.Ledger(PROJECT)
    view = restarted.get(unit_id)
    assert view["state"] == "pending"  # 提交前重启仍 pending
    assert view["attempts"] == {}


def test_different_params_plans_as_distinct_unit(ledger, unit_id, state_dir):
    # seed 已不是协议字段：区分 unit 的途径是 params（普通参数）差异
    other = ledger.plan(unit_raw(params={"lr": 1e-3, "k": 8}))
    assert other != unit_id
    assert len(json.loads(_ledger_path(state_dir).read_text())["units"]) == 2
    assert ledger.get(unit_id)["state"] == "pending"


# ---- attempt 生命周期 ---------------------------------------------------
def test_attempt_lifecycle_persists_across_restart(ledger, unit_id, state_dir):
    ledger.start_attempt(unit_id, "att-1", "sess-1")
    ledger.set_phase(unit_id, "att-1", "running")
    view = L.Ledger(PROJECT).get(unit_id)
    assert view["latest_attempt"] == "att-1"
    assert not view["latest_resolved"]
    assert view["attempts"]["att-1"]["session"] == "sess-1"
    assert view["attempts"]["att-1"]["phase"] == "running"
    assert view["attempts"]["att-1"]["observation"] == "ok"


def test_second_attempt_rejected_while_unresolved(ledger, unit_id):
    ledger.start_attempt(unit_id, "att-1", "sess-1")
    with pytest.raises(L.LedgerError, match="未解决"):
        ledger.start_attempt(unit_id, "att-2", "sess-2")


def test_attempt_id_globally_unique(ledger, unit_id):
    other_uid = ledger.plan(unit_raw(params={"lr": 1e-3, "k": 8}))
    ledger.start_attempt(other_uid, "att-x", "sess-1")
    with pytest.raises(L.LedgerError, match="唯一"):
        ledger.start_attempt(unit_id, "att-x", "sess-2")
    assert ledger.get(unit_id)["attempts"] == {}


def test_observation_lost_changes_only_observation(ledger, unit_id):
    ledger.start_attempt(unit_id, "att-1", "sess-1")
    ledger.set_phase(unit_id, "att-1", "running")
    ledger.set_observation(unit_id, "att-1", "lost")
    attempt = ledger.get(unit_id)["attempts"]["att-1"]
    # 观测失联不改变阶段，不转失败/资源丢失，也不解锁新 attempt
    assert attempt["phase"] == "running"
    assert attempt["resolution"] is None
    with pytest.raises(L.LedgerError, match="未解决"):
        ledger.start_attempt(unit_id, "att-2", "sess-1")
    ledger.set_observation(unit_id, "att-1", "ok")  # 重连后允许恢复观测状态
    with pytest.raises(L.LedgerError, match="observation"):
        ledger.set_observation(unit_id, "att-1", "dead")


def test_reconcile_requires_matching_identity_and_definitive_terminal(
        ledger, unit_id):
    ledger.start_attempt(unit_id, "att-1", "sess-1")
    with pytest.raises(L.LedgerError, match="session"):
        ledger.reconcile(unit_id, "att-1", "sess-2",
                         {"source": "observe", "terminal_state": "SUCCEEDED"})
    with pytest.raises(L.LedgerError, match="终态"):
        ledger.reconcile(unit_id, "att-1", "sess-1",
                         {"source": "observe", "terminal_state": "UNREACHABLE"})
    with pytest.raises(L.LedgerError, match="字段"):
        ledger.reconcile(unit_id, "att-1", "sess-1",
                         {"terminal_state": "FAILED"})
    with pytest.raises(L.LedgerError, match="证据|unknown|字段"):
        ledger.reconcile(unit_id, "att-1", "sess-1",
                         {"source": "observe", "terminal_state": "FAILED",
                          "auto_retry": True})


def test_reconcile_terminal_evidence_enables_explicit_retry(ledger, unit_id):
    ledger.start_attempt(unit_id, "att-1", "sess-1")
    ledger.set_observation(unit_id, "att-1", "lost")
    #失败终态证据仍必须匹配 session 身份；台账不自动重试、不生成新 attempt，
    #unit 保持 pending；随后是否重跑完全由调用者显式决定。
    assert ledger.reconcile(unit_id, "att-1", "sess-1",
                            {"source": "colab-observe",
                             "terminal_state": "FAILED"}) == "task_failed"
    assert ledger.get(unit_id)["state"] == "pending"
    restarted = L.Ledger(PROJECT)
    assert restarted.get(unit_id)["latest_attempt"] == "att-1"
    restarted.start_attempt(unit_id, "att-2", "sess-2")
    assert restarted.get(unit_id)["latest_attempt"] == "att-2"


def test_reconcile_idempotent_but_not_rewritable(ledger, unit_id):
    ledger.start_attempt(unit_id, "att-1", "sess-1")
    evidence = {"source": "colab-observe", "terminal_state": "FAILED"}
    assert ledger.reconcile(unit_id, "att-1", "sess-1", evidence) == "task_failed"
    assert ledger.reconcile(unit_id, "att-1", "sess-1", evidence) == "task_failed"
    with pytest.raises(L.LedgerError, match="不同终态证据"):
        ledger.reconcile(unit_id, "att-1", "sess-1",
                         {"source": "other-source", "terminal_state": "FAILED"})


def test_success_evidence_forces_commit_before_retry(ledger, unit_id):
    _succeed(ledger, unit_id)
    assert ledger.get(unit_id)["state"] == "pending"
    with pytest.raises(L.LedgerError, match="成果未提交"):
        ledger.start_attempt(unit_id, "att-2", "sess-1")


# ---- 成果提交 -----------------------------------------------------------
def test_commit_requires_resolved_matching_latest_attempt(ledger, unit_id):
    with pytest.raises(L.LedgerError, match="不存在"):
        ledger.commit_result(unit_id, "att-1", "sess-1", **RESULT_KW)
    ledger.start_attempt(unit_id, "att-1", "sess-1")
    with pytest.raises(L.LedgerError, match="task_succeeded"):
        ledger.commit_result(unit_id, "att-1", "sess-1", **RESULT_KW)
    ledger.reconcile(unit_id, "att-1", "sess-1",
                     {"source": "observe", "terminal_state": "FAILED"})
    with pytest.raises(L.LedgerError, match="task_succeeded"):
        ledger.commit_result(unit_id, "att-1", "sess-1", **RESULT_KW)
    ledger.start_attempt(unit_id, "att-2", "sess-2")
    ledger.reconcile(unit_id, "att-2", "sess-2",
                     {"source": "observe", "terminal_state": "SUCCEEDED"})
    with pytest.raises(L.LedgerError, match="非最新|最新"):
        ledger.commit_result(unit_id, "att-1", "sess-1", **RESULT_KW)  # 旧 attempt
    with pytest.raises(L.LedgerError, match="session"):
        ledger.commit_result(unit_id, "att-2", "sess-1", **RESULT_KW)  # 错 session


def test_commit_validates_payload(ledger, unit_id):
    _succeed(ledger, unit_id)
    bad = [
        dict(RESULT_KW, snapshot="c" * 63),
        dict(RESULT_KW, snapshot="latest"),
        dict(RESULT_KW, manifest_sha256="d" * 32),
        dict(RESULT_KW, namespace=""),
        dict(RESULT_KW, validator_result={"kind": "other", "status": "passed"}),
        dict(RESULT_KW, validator_result={"kind": "metric-threshold",
                                          "status": "unverified"}),
        dict(RESULT_KW, validator_result={"kind": "metric-threshold"}),
        dict(RESULT_KW, manifest={"../escape": {}}),
        dict(RESULT_KW, manifest={"outputs/": {"size": float("nan")}}),
        dict(RESULT_KW, manifest="not-a-dict"),
    ]
    for kwargs in bad:
        with pytest.raises(L.LedgerError):
            ledger.commit_result(unit_id, "att-1", "sess-1", **kwargs)
    assert ledger.get(unit_id)["state"] == "pending"


def test_commit_recorded_then_idempotent_then_conflicting(ledger, unit_id, state_dir):
    _succeed(ledger, unit_id)
    assert ledger.commit_result(unit_id, "att-1", "sess-1",
                                manifest={"outputs/ck.pt": {"bytes": 3}},
                                **RESULT_KW) == "recorded"
    kwargs = dict(RESULT_KW, manifest={"outputs/ck.pt": {"bytes": 3}})
    assert ledger.commit_result(unit_id, "att-1", "sess-1",
                                **kwargs) == "duplicate"
    # 不同成果内容冲突拒绝，且已存结果不变
    with pytest.raises(L.LedgerError, match="不同成果"):
        ledger.commit_result(unit_id, "att-1", "sess-1",
                             **dict(kwargs, snapshot="e" * 64))
    view = ledger.get(unit_id)
    assert view["state"] == "completed"  # 提交后重启仍 completed
    assert view["result"]["snapshot"] == "c" * 64
    assert view["result"]["manifest"] == {"outputs/ck.pt": {"bytes": 3}}
    with pytest.raises(L.LedgerError, match="不能再开 attempt"):
        L.Ledger(PROJECT).start_attempt(unit_id, "att-2", "sess-9")
    # 提交后重启，重复成果仍幂等；台账中 unit 数量不变
    assert L.Ledger(PROJECT).commit_result(unit_id, "att-1", "sess-1",
                                           **kwargs) == "duplicate"
    assert len(json.loads(_ledger_path(state_dir).read_text())["units"]) == 1


# ---- 权限 / 原子性 / 损坏处理 -------------------------------------------
def test_permissions_and_snapshot_only_files(state_dir, ledger, unit_id):
    ledger.start_attempt(unit_id, "att-1", "sess-1")
    units_dir = _ledger_path(state_dir).parent
    assert units_dir.stat().st_mode & 0o777 == 0o700
    assert _ledger_path(state_dir).stat().st_mode & 0o777 == 0o600
    assert (_ledger_path(state_dir).parent / (PROJECT + ".lock")).stat().st_mode \
        & 0o777 == 0o600
    # 原子替换不残留临时文件
    names = sorted(p.name for p in units_dir.iterdir())
    assert names == [f"{PROJECT}.json", f"{PROJECT}.lock"]


@pytest.mark.parametrize("payload,detail", [
    (b"not json", "无法解析"),
    (b'{"schema_version": 3, "project": "alpha", "units": {}}', "schema_version"),
    (b'{"schema_version": 2, "project": "other", "units": {}}', "结构"),
    (b'{"schema_version": 1, "project": "alpha"}', "结构"),
])
def test_corrupt_ledger_fails_closed(state_dir, unit_id, payload, detail):
    path = _ledger_path(state_dir)
    path.write_bytes(payload)
    ledger = L.Ledger(PROJECT)
    with pytest.raises(L.CorruptLedgerError, match=detail):
        ledger.get(unit_id)
    with pytest.raises(L.CorruptLedgerError):
        ledger.start_attempt(unit_id, "att-9", "sess-9")
    # 损坏状态绝不覆盖为空台账
    assert path.read_bytes() == payload


def test_tampered_spec_fails_closed_without_overwrite(state_dir, ledger, unit_id):
    _succeed(ledger, unit_id)
    path = _ledger_path(state_dir)
    data = json.loads(path.read_text())
    forged = json.loads(json.dumps(data))
    uid = next(iter(forged["units"]))
    forged["units"][uid]["spec"]["params"] = {"lr": 1e-3, "evil": True}  # spec 与 unit id 不再一致
    text = json.dumps(forged)
    path.write_text(text)
    with pytest.raises(L.CorruptLedgerError, match="不一致"):
        ledger.get(unit_id)
    with pytest.raises(L.CorruptLedgerError, match="不一致"):
        L.Ledger(PROJECT).commit_result(uid, "att-1", "sess-1", **RESULT_KW)
    assert path.read_text() == text  # 未被覆盖


def test_explicit_identity_only(state_dir, ledger, unit_id):
    # 不存在隐式 latest 指代；get 只接受完整 64 位十六进制 id
    with pytest.raises(L.LedgerError, match="不存在"):
        ledger.get("e" * 64)
    with pytest.raises(L.LedgerError, match="不存在"):
        ledger.get("latest")
    _succeed(ledger, unit_id)
    with pytest.raises(L.LedgerError, match="attempt id 非法"):
        ledger.commit_result(unit_id, "att nope", "sess-1", **RESULT_KW)
    with pytest.raises(L.LedgerError, match="attempt[_ ]id"):
        ledger.start_attempt(unit_id, "bad id!", "sess-1")
    with pytest.raises(L.LedgerError, match="session"):
        ledger.start_attempt(unit_id, "att-1", "bad session!")


# ---- 跨进程并发 ---------------------------------------------------------
def test_concurrent_preemption_exactly_one_winner(state_dir):
    ledger = L.Ledger(PROJECT)
    uid = ledger.plan(unit_raw())
    code = f"""
import sys
sys.path.insert(0, {SRC_ROOT!r})
from mlbridge import ledger
led = ledger.Ledger({PROJECT!r})
try:
    led.start_attempt({uid!r}, sys.argv[1], "sess-1")
    print("WIN")
except Exception as exc:
    print("ERR", type(exc).__name__)
"""
    procs = [subprocess.Popen([sys.executable, "-c", code, aid],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, env={**os.environ,
                                              "PYTHONPATH": SRC_ROOT,
                                              "XDG_STATE_HOME": str(state_dir)})
             for aid in ("att-A", "att-B")]
    results = [p.communicate()[0].strip() for p in procs]
    assert len([r for r in results if r == "WIN"]) == 1, results
    assert len([r for r in results if r.startswith("ERR")]) == 1, results
    # 抢占失败方不产生记录，只留下赢家的一次 attempt
    view = L.Ledger(PROJECT).get(uid)
    assert len(view["attempts"]) == 1
    assert list(view["attempts"].values())[0]["phase"] == "start"


def test_concurrent_plan_is_idempotent(state_dir):
    ledger = L.Ledger(PROJECT)
    uid = ledger.plan(unit_raw(params={"lr": 1e-3, "k": 3}))
    code = f"""
import sys
sys.path.insert(0, {SRC_ROOT!r})
from mlbridge import ledger
led = ledger.Ledger({PROJECT!r})
print(led.plan({unit_raw(params={"lr": 1e-3, "k": 3})!r}) == {uid!r})
"""
    out, rc = _child(code, state_dir, [])
    assert rc == 0 and out == "True", out
    units = json.loads(_ledger_path(state_dir).read_text())["units"]
    assert list(units) == [uid]


def test_concurrent_commit_exactly_one_effective_change(state_dir):
    ledger = L.Ledger(PROJECT)
    uid = ledger.plan(unit_raw())
    _succeed(ledger, uid)
    kwargs = dict(RESULT_KW, manifest={"outputs/ck.pt": {"bytes": 3}})
    code = f"""
import sys
sys.path.insert(0, {SRC_ROOT!r})
from mlbridge import ledger
led = ledger.Ledger({PROJECT!r})
try:
    status = led.commit_result({uid!r}, "att-1", "sess-1", **{kwargs!r})
    print("OK", status)
except Exception as exc:
    print("ERR", type(exc).__name__)
"""
    out1, _ = _child(code, state_dir, [])
    out2, _ = _child(code, state_dir, [])
    statuses = sorted(out.split()[-1] for out in (out1, out2))
    assert statuses == ["duplicate", "recorded"], (out1, out2)


# ---- project 归属与 manifest 路径边界 ------------------------------------
def test_plan_rejects_unit_of_other_project(ledger):
    with pytest.raises(L.LedgerError, match="project"):
        ledger.plan(unit_raw(project="other"))


def test_spec_project_mismatch_on_disk_fails_closed(state_dir, ledger, unit_id):
    ledger.plan(unit_raw())
    path = _ledger_path(state_dir)
    data = json.loads(path.read_text())
    uid = next(iter(data["units"]))
    spec_raw = dict(data["units"][uid]["spec"])
    spec_raw["project"] = "other"
    forged = P.WorkUnit.from_dict(spec_raw)  # 身份与定义仍然自洽
    record = data["units"].pop(uid)
    data["units"][forged.identity] = record
    text = json.dumps(data)
    path.write_text(text)
    with pytest.raises(L.CorruptLedgerError, match="spec.project"):
        ledger.get(forged.identity)
    assert path.read_text() == text


def test_commit_rejects_manifest_escapes_and_outside_outputs(ledger, unit_id):
    _succeed(ledger, unit_id)
    bad_keys = ["outputs/../escape", "outputs/.git/config", "/abs/path",
                "..", "", "other/x", "outputs/ck\0.pt", "outputs-extra/x"]
    for key in bad_keys:
        with pytest.raises(L.LedgerError):
            ledger.commit_result(unit_id, "att-1", "sess-1",
                                 manifest={key: {}}, **RESULT_KW)
    assert ledger.get(unit_id)["state"] == "pending"


def test_tampered_manifest_escape_fails_on_reload(state_dir, ledger, unit_id):
    _succeed(ledger, unit_id)
    ledger.commit_result(unit_id, "att-1", "sess-1",
                         manifest={"outputs/ck.pt": {"bytes": 3}}, **RESULT_KW)
    path = _ledger_path(state_dir)
    data = json.loads(path.read_text())
    uid = next(iter(data["units"]))
    for escaped in ("outputs/../escape", "outputs/.git/config"):
        data["units"][uid]["result"]["manifest"] = {escaped: {}}
        text = json.dumps(data)
        path.write_text(text)
        with pytest.raises(L.CorruptLedgerError, match="outputs 树内"):
            L.Ledger(PROJECT).get(uid)
        assert path.read_text() == text


# ---- validator 绑定摘要与成果身份 ----------------------------------------
def test_commit_binds_validator_spec_digest(ledger, unit_id):
    _succeed(ledger, unit_id)
    wrong = dict(RESULT_KW, validator_spec_sha256="e" * 64)
    with pytest.raises(L.LedgerError, match="validator_spec_sha256"):
        ledger.commit_result(unit_id, "att-1", "sess-1", **wrong)
    # 正确摘要 + kind 一致才可提交
    assert ledger.commit_result(unit_id, "att-1", "sess-1", **RESULT_KW) == "recorded"


def test_result_identity_fields_bound_and_survive_reload(ledger, unit_id, state_dir):
    _succeed(ledger, unit_id)
    ledger.commit_result(unit_id, "att-1", "sess-1", **RESULT_KW)
    data = json.loads(_ledger_path(state_dir).read_text())
    uid = next(iter(data["units"]))
    result = data["units"][uid]["result"]
    assert result["unit_id"] == uid
    assert result["attempt_id"] == "att-1"
    assert result["session"] == "sess-1"
    assert result["validator_spec_sha256"] == EXPECTED_VALIDATOR_SPEC_SHA256
    # 成果身份字段纳入幂等比较：绑定字段不一致 → 冲突拒绝
    with pytest.raises(L.LedgerError, match="session"):
        ledger.commit_result(unit_id, "att-1", "sess-other", **RESULT_KW)


# ---- 重启不变量篡改矩阵：全部 fail closed 且不覆盖 -----------------------
@pytest.fixture
def two_attempts_completed(ledger):
    """att-1 失败 → att-2 成功 → 提交成果；用于成果绑定类篡改。"""
    uid = ledger.plan(unit_raw())
    ledger.start_attempt(uid, "att-1", "sess-1")
    ledger.reconcile(uid, "att-1", "sess-1",
                     {"source": "observe", "terminal_state": "FAILED"})
    ledger.start_attempt(uid, "att-2", "sess-2")
    ledger.reconcile(uid, "att-2", "sess-2",
                     {"source": "observe", "terminal_state": "SUCCEEDED"})
    ledger.commit_result(uid, "att-2", "sess-2",
                         manifest={"outputs/ck.pt": {"bytes": 3}}, **RESULT_KW)
    return uid


@pytest.mark.parametrize("label,mutate,detail", [
    ("state=pending 却携带 result",
     lambda r, d: r.update({"state": "pending"}), "却携带 result"),
    ("completed 但 result 缺失",
     lambda r, d: r.update({"result": None}), "result 缺失"),
    ("旧 attempt 必须 task_failed",
     lambda r, d: r["attempts"]["att-1"].update(
         {"resolution": {"outcome": "task_succeeded", "source": "observe",
                         "terminal_state": "SUCCEEDED"}}), "task_failed"),
    ("result 绑定旧 attempt",
     lambda r, d: r["result"].update({"attempt_id": "att-1"}), "不是最新"),
    ("result.session 不符",
     lambda r, d: r["result"].update({"session": "sess-x"}), "session 不符"),
    ("result 绑定错误 unit",
     lambda r, d: r["result"].update({"unit_id": "f" * 64}), "错误 unit"),
    ("validator 摘要不符",
     lambda r, d: r["result"].update({"validator_spec_sha256": "f" * 64}),
     "validator_spec_sha256"),
    ("result 身份字段缺失",
     lambda r, d: [r["result"].pop(k) for k in ("session", "unit_id")],
     "字段不符"),
])
def test_result_binding_tampers(state_dir, ledger, two_attempts_completed,
                                label, mutate, detail):
    path = _ledger_path(state_dir)
    data = json.loads(path.read_text())
    uid = next(iter(data["units"]))
    mutate(data["units"][uid], data)
    text = json.dumps(data)
    path.write_text(text)
    with pytest.raises(L.CorruptLedgerError, match=detail):
        L.Ledger(PROJECT).get(uid)
    # 损坏台账必须让写路径同样 fail closed，且绝不被覆盖/重建
    with pytest.raises(L.CorruptLedgerError):
        L.Ledger(PROJECT).plan(unit_raw(params={"lr": 1e-3, "k": 123}))
    assert path.read_text() == text


def test_attempt_invariant_tampers(state_dir):
    attempt = {"seq": 1, "session": "sess-1", "phase": "start",
               "observation": "ok", "resolution": None}

    def attempts_not_dict(r, d):
        r["attempts"] = []

    def two_unresolved(r, d):
        a2 = dict(attempt, seq=2, session="sess-2", phase="running")
        r["attempts"] = {"att-1": dict(attempt), "att-2": a2}

    def unresolved_not_latest(r, d):
        a2 = dict(attempt, seq=2, session="sess-2", phase="running",
                  resolution={"outcome": "task_failed", "source": "observe",
                              "terminal_state": "FAILED"})
        r["attempts"] = {"att-1": dict(attempt), "att-2": a2}

    def seq_gap(r, d):
        a2 = dict(attempt, seq=3, session="sess-2", phase="running")
        r["attempts"] = {"att-1": dict(attempt), "att-2": a2}

    cases = [
        ("attempts 不是 dict", attempts_not_dict, "dict"),
        ("未解决 attempt 不唯一", two_unresolved, "多个未解决"),
        ("未解决 attempt 不是最新", unresolved_not_latest, "不是最新"),
        ("seq 不连续", seq_gap, "连续"),
    ]
    for label, mutate, detail in cases:
        # 每个用例前重置整个台账目录：上一轮篡改过的损坏文件不能留下
        units_dir = _ledger_path(state_dir).parent
        if units_dir.exists():
            for f in units_dir.iterdir():
                f.unlink()
        ledger = L.Ledger(PROJECT)
        uid = ledger.plan(unit_raw())
        ledger.start_attempt(uid, "att-1", "sess-1")
        path = _ledger_path(state_dir)
        data = json.loads(path.read_text())
        mutate(data["units"][uid], data)
        text = json.dumps(data)
        path.write_text(text)
        with pytest.raises(L.CorruptLedgerError, match=detail):
            L.Ledger(PROJECT).get(uid)
        assert path.read_text() == text


def test_cross_unit_attempt_id_collision_fails_closed(state_dir, ledger):
    uid_a = ledger.plan(unit_raw(params={"lr": 1e-3, "k": 1}))
    uid_b = ledger.plan(unit_raw(params={"lr": 1e-3, "k": 2}))
    ledger.start_attempt(uid_a, "att-1", "sess-1")
    path = _ledger_path(state_dir)
    data = json.loads(path.read_text())
    attempt = dict(data["units"][uid_a]["attempts"]["att-1"],
                   session="sess-2", phase="start")
    data["units"][uid_b]["attempts"] = {"att-1": attempt}
    text = json.dumps(data)
    path.write_text(text)
    with pytest.raises(L.CorruptLedgerError, match="跨 unit 重复"):
        L.Ledger(PROJECT).get(uid_a)
    assert path.read_text() == text


def test_duplicate_json_keys_and_non_finite_numbers_rejected(state_dir, ledger,
                                                            unit_id):
    ledger.plan(unit_raw())
    path = _ledger_path(state_dir)
    good = path.read_bytes()

    path.write_bytes(b'{"schema_version":1,"project":"alpha","units":{},'
                     b'"units":{}}')
    with pytest.raises(L.CorruptLedgerError, match="重复"):
        ledger.get(unit_id)

    path.write_bytes(b'{"schema_version":1,"project":"alpha",'
                     b'"units":{"x":{"phase":NaN}}}')
    with pytest.raises(L.CorruptLedgerError, match="非有限"):
        ledger.get(unit_id)

    path.write_bytes(b'{"schema_version":1,"project":"alpha",'
                     b'"units":{"x":{"phase":1e400}}}')
    with pytest.raises(L.CorruptLedgerError, match="非有限"):
        ledger.get(unit_id)

    path.write_bytes(good)
    assert ledger.get(unit_id)["state"] == "pending"  # 修复文件后恢复正常


def test_symlinked_ledger_and_lock_are_refused(state_dir, ledger, unit_id):
    ledger.plan(unit_raw())
    path = _ledger_path(state_dir)
    good = path.read_text()
    target = state_dir / "escape-target.json"

    path.unlink()
    path.symlink_to(target)
    with pytest.raises(L.LedgerError, match="拒绝追随"):
        ledger.get(unit_id)
    with pytest.raises(L.LedgerError, match="拒绝追随"):
        ledger.plan(unit_raw(params={"lr": 1e-3, "k": 5}))
    assert not target.exists() and path.is_symlink()  # 未写入链接目标

    path.unlink()
    path.write_text(good)
    assert ledger.get(unit_id)["state"] == "pending"

    lock = path.parent / f"{PROJECT}.lock"
    lock_target = state_dir / "lock-target"
    lock.unlink()
    lock.symlink_to(lock_target)
    with pytest.raises(L.LedgerError, match="锁文件"):
        ledger.start_attempt(unit_id, "att-1", "sess-1")
    assert not lock_target.exists()


def test_save_side_validation_refuses_bad_final_state(state_dir, ledger,
                                                      unit_id, monkeypatch):
    """写盘前对最终状态再次完整校验：程序路径写坏数据也无法入盘。"""
    original_save = L.Ledger._save

    def poisoned_save(self, data):
        data["units"][next(iter(data["units"]))]["attempts"] = []  # 模拟程序 bug
        return original_save(self, data)

    ledger.start_attempt(unit_id, "att-1", "sess-1")  # 先留下合法中间态
    monkeypatch.setattr(L.Ledger, "_save", poisoned_save)
    path = _ledger_path(state_dir)
    good = path.read_text()
    with pytest.raises(L.CorruptLedgerError, match="dict"):
        ledger.set_phase(unit_id, "att-1", "running")
    assert path.read_text() == good  # 磁盘上仍是上一个合法状态


def test_plan_idempotent_after_rejection_attempts(ledger, unit_id):
    ledger.plan(unit_raw())
    assert ledger.plan(unit_raw()) == unit_id  # 幂等；unit 数量不变


# ---- 收尾加固：损坏字段矩阵 / 目录边界 / fsync 故障注入 --------------------
CORRUPT_FIELD_CASES = [
    ("observation 是 list", lambda a: a.update({"observation": []})),
    ("observation 是 dict", lambda a: a.update({"observation": {}})),
    ("resolution 是 list", lambda a: a.update({"resolution": []})),
    ("outcome 是 list", lambda a: a.update(
        {"resolution": {"outcome": [], "source": "observe",
                        "terminal_state": "FAILED"}})),
    ("terminal_state 是 dict", lambda a: a.update(
        {"resolution": {"outcome": "task_failed", "source": "observe",
                        "terminal_state": {"k": "v"}}})),
]


@pytest.mark.parametrize("label,mutate", CORRUPT_FIELD_CASES,
                         ids=[c[0] for c in CORRUPT_FIELD_CASES])
def test_corrupt_field_types_fail_closed_read_and_write(
        state_dir, ledger, unit_id, label, mutate):
    """损坏为 list/dict 的字段：读/写路径都 CorruptLedgerError，原文件不变。"""
    ledger.start_attempt(unit_id, "att-1", "sess-1")
    path = _ledger_path(state_dir)
    original = path.read_bytes()
    data = json.loads(path.read_text())
    uid = next(iter(data["units"]))
    mutate(data["units"][uid]["attempts"]["att-1"])
    text = json.dumps(data)
    path.write_text(text)
    assert path.read_bytes() != original  # 测试构造确实改变了文件
    fresh = L.Ledger(PROJECT)
    with pytest.raises(L.CorruptLedgerError):  # 读路径 fail closed
        fresh.get(unit_id)
    with pytest.raises(L.CorruptLedgerError):  # 写路径同样 fail closed
        fresh.plan(unit_raw(params={"lr": 1e-3, "k": 99}))
    assert path.read_bytes() == text.encode()  # 损坏文件绝不被覆盖


def test_non_str_embeddings_in_public_api_raise_ledger_error(ledger, unit_id):
    """公共 API 遇非 str：LedgerError，而不是 re/frozenset 的裸 TypeError。"""
    ledger.start_attempt(unit_id, "att-1", "sess-1")
    with pytest.raises(L.LedgerError, match="observation"):
        ledger.set_observation(unit_id, "att-1", ["ok"])
    with pytest.raises(L.LedgerError, match="attempt[_ ]id"):
        ledger.start_attempt(unit_id, ["att-2"], "sess-2")
    with pytest.raises(L.LedgerError, match="session"):
        ledger.start_attempt(unit_id, "att-2", {"sess": 2})
    with pytest.raises(L.LedgerError, match="attempt[_ ]id"):
        ledger.commit_result(unit_id, 123, "sess-1", **RESULT_KW)
    with pytest.raises(L.LedgerError, match="attempt id 非法"):
        ledger.reconcile(unit_id, b"att-1", "sess-1",
                         {"source": "observe", "terminal_state": "FAILED"})
    # validate_evidence：非 str 键不再让 unknown 的 sorted 冒出裸 TypeError
    with pytest.raises(L.LedgerError, match="字段名"):
        ledger.reconcile(unit_id, "att-1", "sess-1",
                         {"source": "observe", "terminal_state": "FAILED",
                          "auto_retry": True, b"raw": 1})
    assert ledger.get(unit_id)["attempts"]["att-1"]["resolution"] is None


def test_units_directory_symlink_refused_target_untouched(
        state_dir, ledger, unit_id):
    """units 目录是符号链接：拒绝追随；目标目录的权限与内容绝不被修改。"""
    units_dir = _ledger_path(state_dir).parent
    target = state_dir / "mlbridge" / "units-real"
    target.mkdir(parents=True, exist_ok=True)
    (target / "sentinel.txt").write_text("victim")
    target.chmod(0o755)
    shutil.rmtree(units_dir)
    units_dir.symlink_to(target)
    with pytest.raises(L.LedgerError, match="普通目录"):
        ledger.get(unit_id)
    with pytest.raises(L.LedgerError, match="普通目录"):
        ledger.plan(unit_raw(params={"lr": 1e-3, "k": 11}))
    # 链接目标保持原样：权限未被 chmod，哨兵文件未被动过，也没新建台账
    assert target.stat().st_mode & 0o777 == 0o755
    assert (target / "sentinel.txt").read_text() == "victim"
    assert sorted(p.name for p in target.iterdir()) == ["sentinel.txt"]
    assert units_dir.is_symlink()


def test_units_directory_as_plain_file_refused(state_dir, ledger, unit_id):
    units_dir = _ledger_path(state_dir).parent
    shutil.rmtree(units_dir)
    units_dir.write_text("not a directory")
    with pytest.raises(L.LedgerError, match="普通目录"):
        ledger.get(unit_id)
    assert units_dir.read_text() == "not a directory"  # 未被覆盖/重建


def _inject_dir_fsync_failure(monkeypatch):
    real_fsync = os.fsync

    def flaky_fsync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):  # 只对目录 fd 失效
            raise OSError(5, "注入：目录 fsync 失败")
        return real_fsync(fd)
    monkeypatch.setattr(os, "fsync", flaky_fsync)


def test_dir_fsync_failure_after_replace_reports_uncertain(
        state_dir, ledger, unit_id, monkeypatch):
    """目录 fsync 失败但 replace 已成功：报不确定持久性，并重读到新状态。"""
    ledger.start_attempt(unit_id, "att-1", "sess-1")
    _inject_dir_fsync_failure(monkeypatch)
    with pytest.raises(L.LedgerError, match="已原子替换"):
        ledger.set_phase(unit_id, "att-1", "running")
    # 绝不假称“完全未写入”：磁盘重读确认新状态已生效
    view = ledger.get(unit_id)
    assert view["attempts"]["att-1"]["phase"] == "running"
    assert json.loads(_ledger_path(state_dir).read_text())["units"][unit_id][
        "attempts"]["att-1"]["phase"] == "running"


def test_uncertain_report_when_reread_differs(state_dir, ledger, unit_id,
                                              monkeypatch):
    """replace 后磁盘内容漂移：告知"必须人工检查"，不假称新状态已确认。"""
    ledger.start_attempt(unit_id, "att-1", "sess-1")
    _inject_dir_fsync_failure(monkeypatch)
    real_replace = os.replace

    def drifting_replace(src, dst, **kwargs):
        real_replace(src, dst, **kwargs)
        with open(dst, "a", encoding="utf-8") as handle:  # 模拟替换后漂移
            handle.write("<!-- 注入：磁盘内容漂移 -->")
    monkeypatch.setattr(os, "replace", drifting_replace)
    with pytest.raises(L.LedgerError, match="必须人工检查"):
        ledger.set_phase(unit_id, "att-1", "running")
    assert "注入：磁盘内容漂移" in _ledger_path(state_dir).read_text()


def test_uncertain_report_survives_undecodable_reread(
        state_dir, ledger, unit_id, monkeypatch):
    """replace 后重读抛 UnicodeDecodeError：同样归为"必须人工检查"。"""
    ledger.start_attempt(unit_id, "att-1", "sess-1")
    _inject_dir_fsync_failure(monkeypatch)
    real_replace = os.replace

    def corrupting_replace(src, dst, **kwargs):
        real_replace(src, dst, **kwargs)
        with open(dst, "wb") as handle:  # 注入：替换后磁盘内容不可解码
            handle.write(b"\xff\xfe not utf-8")
    monkeypatch.setattr(os, "replace", corrupting_replace)
    with pytest.raises(L.LedgerError, match="必须人工检查"):
        ledger.set_phase(unit_id, "att-1", "running")
    assert _ledger_path(state_dir).read_bytes().startswith(b"\xff\xfe")


def test_file_fsync_failure_keeps_old_ledger_and_cleans_tmp(
        state_dir, ledger, unit_id, monkeypatch):
    """文件 fsync 失败：旧台账原样，tmp 文件清理干净，无半写状态。"""
    ledger.start_attempt(unit_id, "att-1", "sess-1")
    path = _ledger_path(state_dir)
    good = path.read_bytes()
    units_dir = path.parent
    real_fsync = os.fsync

    def flaky_fsync(fd):
        if stat.S_ISREG(os.fstat(fd).st_mode):  # 只对普通文件 fd 失效
            raise OSError(5, "注入：文件 fsync 失败")
        return real_fsync(fd)
    monkeypatch.setattr(os, "fsync", flaky_fsync)
    with pytest.raises(L.LedgerError, match="写入失败"):
        ledger.set_phase(unit_id, "att-1", "running")
    assert path.read_bytes() == good  # 旧台账原封不动
    tmp_leftovers = [p for p in units_dir.iterdir() if p.name.endswith(".tmp")]
    assert tmp_leftovers == []  # 临时文件已清理


def test_same_plan_retry_is_idempotent_across_lifecycle(state_dir, ledger,
                                                        unit_id):
    """同一 unit 定义在各生命阶段重复 plan 都幂等，绝不产生重复登记。"""
    for _ in range(2):
        assert ledger.plan(unit_raw()) == unit_id
    ledger.start_attempt(unit_id, "att-1", "sess-1")
    ledger.reconcile(unit_id, "att-1", "sess-1",
                     {"source": "observe", "terminal_state": "FAILED"})
    assert ledger.plan(unit_raw()) == unit_id
    ledger.start_attempt(unit_id, "att-2", "sess-2")
    ledger.reconcile(unit_id, "att-2", "sess-2",
                     {"source": "observe", "terminal_state": "SUCCEEDED"})
    assert ledger.commit_result(unit_id, "att-2", "sess-2", **RESULT_KW) \
        == "recorded"
    assert ledger.plan(unit_raw()) == unit_id  # completed 后仍幂等
    units = json.loads(_ledger_path(state_dir).read_text())["units"]
    assert list(units) == [unit_id]
