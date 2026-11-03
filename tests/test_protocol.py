"""WorkUnit 协议的聚焦测试：确定性校验、稳定身份与拒绝规则。"""
from __future__ import annotations

import hashlib
import json
import math

import pytest

from mlbridge import protocol as P


def raw(**overrides):
    """一份合法的 unit 定义；overrides 注入变异字段。"""
    base = {
        "schema_version": 2,
        "project": "alpha",
        "repository": "https://github.com/user/repo.git",
        "commit": "a" * 40,
        "inputs": {"train": "b" * 64},
        "environment": {"python": "3.11.8", "cuda": "12.4"},
        "command": ["python", "train.py"],
        "params": {"lr": 1e-3, "epochs": 10},
        "outputs": ["outputs/ckpts"],
        "validator": {"kind": "metric-threshold", "min_accuracy": 0.9},
    }
    base.update(overrides)
    return base


def test_identity_is_sha256_of_canonical_json():
    unit = P.WorkUnit.from_dict(raw())
    payload = {k: v for k, v in unit.to_dict().items()}
    expected = hashlib.sha256(json.dumps(
        payload, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
    assert unit.identity == expected == P.unit_id(raw())


def test_field_order_does_not_change_identity():
    # dict 键插入顺序不同（inputs / validator / params）不影响身份
    r1 = raw(inputs={"train": "b" * 64, "val": "c" * 64})
    r2 = raw(inputs={"val": "c" * 64, "train": "b" * 64})
    r2["validator"] = {"min_accuracy": 0.9, "kind": "metric-threshold"}
    r2["params"] = {"epochs": 10, "lr": 1e-3}
    assert P.unit_id(r1) == P.unit_id(r2)


def test_every_semantic_field_change_changes_identity():
    base_id = P.unit_id(raw())
    variants = [
        lambda r: r.update({"project": "beta"}),
        lambda r: r.update({"repository": "https://github.com/user/other.git"}),
        lambda r: r.update({"commit": "b" * 40}),
        lambda r: r.update({"inputs": {"train": "d" * 64}}),
        lambda r: r["inputs"].update({"extra": "e" * 64}),
        lambda r: r.update({"environment": {"python": "3.12.0", "cuda": "12.4"}}),
        lambda r: r.update({"command": ["python", "train.py", "--full"]}),
        lambda r: r["params"].update({"new_key": 1}),
        lambda r: r["params"].update({"seed": 8}),
        lambda r: r["outputs"].append("outputs/extra"),
        lambda r: r["validator"].update({"min_accuracy": 0.85}),
    ]
    for mutate in variants:
        mutated = raw()
        mutate(mutated)
        assert P.unit_id(mutated) != base_id, mutated


def test_normalization_does_not_change_identity():
    # commit / snapshot 大小写与 .git 后缀不改变语义身份
    assert P.unit_id(raw(commit="A" * 40)) == P.unit_id(raw())
    assert P.unit_id(raw(inputs={"train": "B" * 64})) == P.unit_id(raw())
    assert P.unit_id(raw(repository="https://github.com/user/repo")) == \
        P.unit_id(raw(repository="https://github.com/user/repo.git"))


def test_identity_ignores_session_by_design():
    # session/attempt 之类执行期信息无法进入协议：未知字段直接拒绝
    unit = P.WorkUnit.from_dict(raw())
    with pytest.raises(P.ProtocolError, match="未知字段"):
        P.WorkUnit.from_dict({**raw(), "session": "sess-1", "attempt_id": "a1"})


def test_roundtrip_to_dict_is_normalized():
    unit = P.WorkUnit.from_dict(raw(commit="A" * 40,
                                    repository="https://github.com/user/repo.git/"))
    assert unit.commit == "a" * 40
    assert unit.repository == "https://github.com/user/repo"
    replay = P.WorkUnit.from_dict(unit.to_dict())
    assert replay.identity == unit.identity


def test_unknown_schema_and_missing_fields_rejected():
    for version in (3, 0, "2", True, None):
        with pytest.raises(P.ProtocolError, match="schema_version"):
            P.WorkUnit.from_dict(raw(schema_version=version))
    for field in ("commit", "environment", "validator", "inputs"):
        broken = raw()
        del broken[field]
        with pytest.raises(P.ProtocolError, match="缺少字段"):
            P.WorkUnit.from_dict(broken)
    with pytest.raises(P.ProtocolError, match="必须是"):
        P.WorkUnit.from_dict(["not", "a", "dict"])


def test_unknown_schema_version_rejected():
    """Unknown schema versions are refused; no field-specific hints exist."""
    with pytest.raises(P.ProtocolError, match="未知 schema_version"):
        P.WorkUnit.from_dict(raw(schema_version=1))
    with pytest.raises(P.ProtocolError, match="未知 schema_version"):
        P.WorkUnit.from_dict(raw(schema_version=3))


def test_params_keys_are_ordinary_and_unknown_top_level_fields_rejected():
    """params 内的任意 JSON 键（含 seed）都按普通参数处理；params 之外的
    多余字段一律按未知字段拒绝。"""
    unit = P.WorkUnit.from_dict(raw())
    assert not hasattr(unit, "seed")
    # params.seed 可以是任意 JSON 值，协议不做特殊校验
    for value in ("7", 7.0, None, [1, 2], {"nested": True}):
        weird = raw(params={"seed": value, "lr": 1e-3})
        assert P.WorkUnit.from_dict(weird).params == {"seed": value, "lr": 1e-3}
    # 顶层 seed 是未知字段：照旧拒绝（未知字段拒绝的既有保障不变）
    with pytest.raises(P.ProtocolError, match="未知字段"):
        P.WorkUnit.from_dict({**raw(), "seed": 7})


def test_non_json_values_rejected():
    for bad in [float("nan"), float("inf"), -float("inf")]:
        with pytest.raises(P.ProtocolError, match="NaN"):
            P.WorkUnit.from_dict(raw(params={"x": bad}))
        with pytest.raises(P.ProtocolError, match="NaN"):
            P.WorkUnit.from_dict(raw(validator={"kind": "k", "x": bad}))
    with pytest.raises(P.ProtocolError, match="非 JSON 类型"):
        P.WorkUnit.from_dict(raw(params={"x": (1, 2)}))
    with pytest.raises(P.ProtocolError, match="非 JSON 类型"):
        P.WorkUnit.from_dict(raw(params={"x": {1, 2}}))
    with pytest.raises(P.ProtocolError, match="键必须是字符串"):
        P.WorkUnit.from_dict(raw(params={1: "x"}))
    with pytest.raises(P.ProtocolError, match="键必须是字符串"):
        P.WorkUnit.from_dict(raw(params={"a": {"b": {3: "x"}}}))


def test_illegal_inputs_rejected():
    bad_values = [
        {"project": "bad name!"},
        {"project": ""},
        {"repository": ""},
        {"repository": "//"},
        {"commit": "a" * 39},
        {"commit": "g" * 40},
        {"inputs": {}},
        {"inputs": {"train": "b" * 63}},
        {"inputs": {"train数据": "b" * 64}},
        {"inputs": {"train": "zzz"}},
        {"environment": {}},
        {"environment": {"python": 3.11}},
        {"environment": {"python": None}},
        {"command": []},
        {"command": ["python", ""]},
        {"command": "python train.py"},
        {"outputs": ["../escape"]},
        {"outputs": ["/etc/passwd"]},
        {"outputs": ["a/.git/b"]},
        {"outputs": [""]},
        {"outputs": [".git"]},
        {"outputs": ["a/../../b"]},
        {"validator": {}},
        {"validator": {"kind": ""}},
        {"validator": {"kind": 1}},
        {"validator": "check accuracy"},
    ]
    for override in bad_values:
        with pytest.raises(P.ProtocolError):
            P.WorkUnit.from_dict(raw(**override))


def test_params_list_and_scalars_are_valid_json():
    for ok_params in ([1, 2, {"a": [True, None, 3.5, "s"]}], 42, "text", None):
        unit = P.WorkUnit.from_dict(raw(params=ok_params))
        assert unit.params == ok_params


# ---- 公共构造器与真正不可变表示 ------------------------------------------
def test_public_constructor_marshals_through_same_validation():
    # 公共构造器与 from_dict 必须产出完全一致的合法实例（identity/eq 等值）
    via_dict = P.WorkUnit.from_dict(raw())
    direct = P.WorkUnit(
        project=via_dict.project, repository=via_dict.repository,
        commit=via_dict.commit, inputs=via_dict.inputs,
        environment=via_dict.environment, command=via_dict.command,
        params=via_dict.params, outputs=via_dict.outputs,
        validator=via_dict.validator)
    assert direct == via_dict
    assert direct.identity == via_dict.identity == P.unit_id(raw())
    direct2 = P.WorkUnit(
        project="beta", repository="https://github.com/u/r.git/",
        commit="A" * 40, inputs={"t": "B" * 64}, environment={"py": "3"},
        command=["python"], params=1, outputs=["out/"],
        validator={"kind": "k"})
    assert direct2.commit == "a" * 40
    assert direct2.inputs == {"t": "b" * 64}
    assert direct2.outputs == ["out"]  # checked_relative 去掉尾部斜杠


def test_public_constructor_rejects_illegal_values():
    # 绕过 from_dict 直接构造同样走唯一校验通道；非法值抛 ProtocolError
    bad_kwargs = [
        dict(project="bad name!"), dict(repository=""),
        dict(commit="a" * 39), dict(inputs={}),
        dict(environment={"py": 3.11}), dict(command=[]),
        dict(outputs=["../escape"]),
        dict(validator="not a dict"),
    ]
    base = dict(P.coerce(raw()).to_dict())
    base.pop("schema_version")
    for override in bad_kwargs:
        with pytest.raises(P.ProtocolError):
            P.WorkUnit(**{**base, **override})
    # 非法标量类型也要 ProtocolError，而不是裸 TypeError/AttributeError
    scalar_bad = [
        dict(project=5), dict(project=None), dict(project=True),
        dict(repository=3.14), dict(commit=7),
        dict(inputs="not-a-dict"), dict(environment=None),
        dict(command="python train.py"),
        dict(outputs="/etc/passwd"),
    ]
    for override in scalar_bad:
        with pytest.raises(P.ProtocolError):
            P.WorkUnit(**{**base, **override})


def test_nested_mutation_cannot_reach_back_into_unit():
    unit = P.WorkUnit.from_dict(raw())
    unit_id_before = unit.identity
    # 对返回副本的嵌套修改不会写回：identity / 定义均不变
    hijacked_inputs = unit.inputs
    hijacked_inputs["backdoor"] = "e" * 64
    assert unit.inputs == {"train": "b" * 64}
    hijacked_validator = unit.validator
    hijacked_validator["kind"] = "evil"
    hijacked_validator.clear()
    assert unit.validator == {"kind": "metric-threshold", "min_accuracy": 0.9}
    hijacked_command = unit.command
    hijacked_command.append("--evil")
    hijacked_outputs = unit.outputs
    hijacked_outputs.append("../escape")
    hijacked_environment = unit.environment
    hijacked_environment["CUDA_HOME"] = "/evil"
    hijacked_params = unit.params
    hijacked_params["lr"] = 999
    assert unit.command == ["python", "train.py"]
    assert unit.outputs == ["outputs/ckpts"]
    assert unit.environment == {"python": "3.11.8", "cuda": "12.4"}
    assert unit.params == {"lr": 1e-3, "epochs": 10}
    assert unit.identity == unit_id_before
    # 先读 ID 再修改：读到的身份不会因后续对单元的使用而失配
    identity = unit.identity
    assert identity == P.unit_id(raw())
    P.WorkUnit.from_dict(unit.to_dict())  # 保持可用性


def test_to_dict_returns_independent_copies():
    unit = P.WorkUnit.from_dict(raw())
    identity = unit.identity
    snapshot = unit.to_dict()
    # 修改 to_dict 的结果不影响 unit：再次 to_dict 干净，identity 不变
    snapshot["params"]["lr"] = 0
    snapshot["validator"]["kind"] = "evil"
    snapshot["outputs"][0] = "../escape"
    snapshot["inputs"]["train"] = "f" * 64
    snapshot2 = unit.to_dict()
    assert snapshot2["params"] == {"lr": 1e-3, "epochs": 10}
    assert snapshot2["validator"] == {"kind": "metric-threshold", "min_accuracy": 0.9}
    assert snapshot2["inputs"] == {"train": "b" * 64}
    assert unit.identity == identity
    replay = P.WorkUnit.from_dict(snapshot2)
    assert replay == unit and replay.identity == identity


def test_unit_instance_is_not_mutatable_at_all():
    unit = P.WorkUnit.from_dict(raw())
    with pytest.raises(P.ProtocolError):
        unit.commit = "b" * 40
    with pytest.raises(P.ProtocolError):
        del unit.commit


def test_coerce_and_equality():
    unit = P.WorkUnit.from_dict(raw())
    assert P.coerce(unit) is unit  # 已合法构造的实例原样放行
    assert P.coerce(raw()).identity == unit.identity
    assert P.coerce(raw()) == unit
    assert P.unit_id(raw(params={"lr": 1e-3, "epochs": 10, "seed": 8})) != unit.identity
    with pytest.raises(P.ProtocolError):
        P.coerce({**raw(), "session": "sess-1"})


def test_mixed_unknown_or_nonstring_keys_raise_protocol_error():
    # 混入未知键 / 非 str 键（含混合类型导致 sorted 崩溃的情况）都要
    # ProtocolError 而不是裸 TypeError
    for extra in ("session", "attempt_id", "session_extra"):
        with pytest.raises(P.ProtocolError, match="未知字段"):
            P.WorkUnit.from_dict({**raw(), extra: "x"})
    for mixed in ({5: "x"}, {5: "x", "session": "s"},
                  {None: "x", "zz": 1}, {(1, 2): "x"}):
        with pytest.raises(P.ProtocolError):
            P.WorkUnit.from_dict({**raw(), **mixed})


def test_null_byte_paths_and_argv_rejected():
    with pytest.raises(P.ProtocolError, match="NUL"):
        P.WorkUnit.from_dict(raw(outputs=["outputs/ck\0pt"]))
    with pytest.raises(P.ProtocolError, match="NUL"):
        P.WorkUnit.from_dict(raw(command=["python", "train\0.py"]))
    with pytest.raises(P.ProtocolError):
        P.WorkUnit.from_dict(raw(outputs=["\0"]))
