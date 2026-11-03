"""版本化工作单元协议：确定性校验与稳定 unit 身份计算。

纯 stdlib 数据校验层；schema 显式版本化，严格拒绝未知 schema、缺失/非法
字段与 NaN/非 JSON 值。

真正不可变的内部表示：WorkUnit 实例唯一的权威状态是构造时刻校验并规范
化后的 canonical JSON 字符串。公共构造器与 ``from_dict`` 走同一条校验/
规范化路径，任何实例都保证合法且 ``identity`` 永不失配；所有序列字段
（inputs / environment / command / params / outputs / validator）按访问
返回独立副本，外部对副本的任何修改都不可能写回单元内部，也就不可能让
identity 与 unit 定义失配。

稳定 unit id 通过对影响成果语义的字段（project、repository、固定 commit、
完整 64 位输入 snapshot、environment 指纹、command argv、params、
outputs、validator 描述）的规范化 JSON 计算 SHA256 得到；执行期信息
（session、attempt_id 等）绝不参与身份计算。
其它科学参数一样只是 ``params`` 里的普通键（如需随身份变化，放进
params 即可），协议层不做任何特殊处理。

常用规范化的含义（失败时抛 ProtocolError，而不是静默改写语义）：
- commit / 输入 snapshot 统一小写十六进制（大小写不表示语义差异）；
- repository 仅做 URL 一致化（去尾斜杠与 .git 后缀），与
  ``core.normalize_git_url`` 采用同一规则；
- 所有 dict 的键都必须是 str，禁止 tuple / set 等非 JSON 类型，避免
  "canonical JSON 相同但 Python 对象不等" 导致的状态机误判；
- outputs 与 command 元素拒绝 NUL 字节；outputs 额外拒绝绝对路径、
  ``..`` 越界与 ``.git`` 目录。

environment 指纹必须由调用方显式给定（``dict[str, str]``）；本模块不读
取运行环境，也不声称实现了环境缓存。
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

from .core import BridgeError, checked_relative, ident, normalize_git_url

#: 当前唯一受支持的 schema 版本；未知版本一律拒绝（fail closed）。
SCHEMA_VERSION = 2

_COMMIT = re.compile(r"^[a-f0-9]{40}$")
_SNAPSHOT = re.compile(r"^[a-f0-9]{64}$")

_FIELDS = ("schema_version", "project", "repository", "commit", "inputs",
           "environment", "command", "params", "outputs", "validator")
_IDENTITY_FIELDS = _FIELDS[1:]  # schema_version 单列；其余全部参与身份


class ProtocolError(BridgeError):
    """unit 定义非法：未知 schema、缺失/非法字段、NaN/非 JSON、非法路径等。"""


def _hex(value: Any, label: str, pattern: re.Pattern[str], length: int) -> str:
    if not isinstance(value, str):
        raise ProtocolError(f"{label} 必须是 {length} 位十六进制字符串: {value!r}")
    value = value.lower()
    if not pattern.fullmatch(value):
        raise ProtocolError(f"{label} 必须是完整 {length} 位十六进制: {value!r}")
    return value


def _require_str(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProtocolError(f"{label} 必须是非空字符串: {value!r}")
    return value


def json_guard(value: Any, where: str) -> Any:
    """递归校验 JSON 安全值并返回规范化副本（不改变语义）。

    拒绝 NaN/Infinity、非 str 键与非 JSON 容器类型（tuple/set 等），
    避免 canonical JSON 一致但对象不等导致的状态不一致。
    """
    if value is None or isinstance(value, bool) or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProtocolError(f"{where} 含 NaN/Infinity，非 JSON 合法值")
        return value
    if isinstance(value, list):
        return [json_guard(item, where) for item in value]
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ProtocolError(f"{where} 的键必须是字符串: {key!r}")
            out[key] = json_guard(item, where)
        return out
    raise ProtocolError(f"{where} 含非 JSON 类型 {type(value).__name__}")


def _deep_copy(value: Any) -> Any:
    """复制 JSON 安全值，保证返回给调用方的是独立副本。"""
    if isinstance(value, dict):
        return {key: _deep_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_deep_copy(item) for item in value]
    return value


def _dumps_canonical(payload: dict[str, Any]) -> str:
    try:
        return json.dumps(payload, sort_keys=True, ensure_ascii=False,
                          separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"unit 字段无法转成 canonical JSON: {exc}") from exc


def _normalize_fields(*, project: Any, repository: Any, commit: Any,
                      inputs: Any, environment: Any, command: Any,
                      params: Any, outputs: Any,
                      validator: Any) -> dict[str, Any]:
    """公共的逐字段校验与规范化；这是进入协议的唯一通道。

    任何非法输入（类型错误、越界路径、NUL、非 JSON 值等）都抛
    ProtocolError，绝不静默改写语义。
    """
    if not isinstance(project, str):
        raise ProtocolError(f"project 必须是字符串: {project!r}")
    try:
        project = ident(project)
    except BridgeError as exc:
        raise ProtocolError(f"project 名称不合法: {project!r} ({exc})") from exc
    if not isinstance(repository, str):
        raise ProtocolError(f"repository 必须是字符串: {repository!r}")
    repository = normalize_git_url(repository)
    if not repository:
        raise ProtocolError("repository 不能为空")
    commit = _hex(commit, "commit", _COMMIT, 40)

    if not isinstance(inputs, dict) or not inputs:
        raise ProtocolError("inputs 必须是非空 dict（名称 -> snapshot）")
    normalized_inputs = {}
    for name, snapshot in inputs.items():
        if not isinstance(name, str):
            raise ProtocolError(f"inputs 名称不合法: {name!r}（必须是字符串）")
        try:
            name = ident(name)
        except BridgeError as exc:
            raise ProtocolError(f"inputs 名称不合法: {name!r} ({exc})") from exc
        normalized_inputs[name] = _hex(snapshot, f"inputs[{name!r}] snapshot",
                                       _SNAPSHOT, 64)

    if not isinstance(environment, dict) or not environment:
        raise ProtocolError("environment 必须是非空 dict（显式环境指纹）")
    for key, value in environment.items():
        if not isinstance(key, str) or not key:
            raise ProtocolError(f"environment 键必须是非空字符串: {key!r}")
        if not isinstance(value, str):
            raise ProtocolError(f"environment[{key!r}] 必须是 str 值: {value!r}")

    if not isinstance(command, list) or not command:
        raise ProtocolError("command 必须为非空 argv 数组（元素为非空字符串）")
    for element in command:
        if not isinstance(element, str) or not element:
            raise ProtocolError("command 必须为非空 argv 数组（元素为非空字符串）")
        if "\0" in element:
            raise ProtocolError(f"command 元素含 NUL 字节，拒绝: {element!r}")

    if not isinstance(outputs, list) or any(not isinstance(x, str) for x in outputs):
        raise ProtocolError("outputs 必须是相对路径字符串数组")
    normalized_outputs = []
    for path in outputs:
        if "\0" in path:
            raise ProtocolError(f"outputs 路径含 NUL 字节，拒绝: {path!r}")
        try:
            normalized_outputs.append(checked_relative(path))
        except (BridgeError, ValueError) as exc:
            raise ProtocolError(f"outputs 路径非法 {path!r}: {exc}") from exc

    validator = json_guard(validator, "validator")
    if not isinstance(validator, dict) or not isinstance(validator.get("kind"), str) \
            or not validator["kind"]:
        raise ProtocolError('validator 必须是非空 dict 且含非空 "kind" 字段')

    return {
        "project": project,
        "repository": repository,
        "commit": commit,
        "inputs": dict(sorted(normalized_inputs.items())),
        "environment": dict(environment),
        "command": list(command),
        "params": json_guard(params, "params"),
        "outputs": normalized_outputs,
        "validator": validator,
    }


class WorkUnit:
    """真正不可变的工作单元定义；``identity`` 即稳定 unit id（SHA256 hex）。

    实例唯一的权威状态是 canonical JSON 字符串 ``_payload``（在构造时经过
    ``_normalize_fields`` 完整校验与规范化后固化为不可变 str）。所有公共
    属性按访问返回副本；禁止属性赋值。``__eq__``/``__hash__`` 基于
    canonical JSON，因此公共构造与 from_dict 产物等值。
    """

    __slots__ = ("_fields", "_payload", "_identity")

    def __init__(self, *, project: Any, repository: Any, commit: Any,
                 inputs: Any, environment: Any, command: Any, params: Any,
                 outputs: Any, validator: Any) -> None:
        # 公共构造与 from_dict 走同一条校验/规范化路径：绕过校验构造出
        # 非法或不一致实例是不可能的。
        fields = _normalize_fields(project=project, repository=repository,
                                   commit=commit, inputs=inputs,
                                   environment=environment, command=command,
                                   params=params, outputs=outputs,
                                   validator=validator)
        payload = {"schema_version": SCHEMA_VERSION}
        payload.update(fields)
        canonical = _dumps_canonical(payload)
        # _fields 是规范化的只读工作副本，仅用于属性访问；对外永远给副本。
        object.__setattr__(self, "_fields", fields)
        object.__setattr__(self, "_payload", canonical)
        object.__setattr__(self, "_identity",
                           hashlib.sha256(canonical.encode("utf-8")).hexdigest())

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "WorkUnit":
        """严格解析并校验 unit 定义；显式 schema_version，禁止缺失/未知字段。"""
        if not isinstance(raw, dict):
            raise ProtocolError("unit 定义必须是 JSON 对象/dict")
        unknown = [key for key in raw
                   if not (isinstance(key, str) and key in _FIELDS)]
        if unknown:
            unknown = sorted(unknown, key=repr)
            raise ProtocolError(f"unit 定义含未知字段: {unknown}（执行期信息不属于协议）")
        missing = [field for field in _FIELDS if field not in raw]
        if missing:
            raise ProtocolError(f"unit 定义缺少字段: {missing}")
        version = raw["schema_version"]
        if isinstance(version, bool) or not isinstance(version, int) \
                or version != SCHEMA_VERSION:
            raise ProtocolError(f"未知 schema_version: {version!r}；仅支持 {SCHEMA_VERSION}")
        return cls(project=raw["project"], repository=raw["repository"],
                   commit=raw["commit"], inputs=raw["inputs"],
                   environment=raw["environment"], command=raw["command"],
                   params=raw["params"], outputs=raw["outputs"],
                   validator=raw["validator"])

    def __setattr__(self, name: str, value: Any) -> None:
        raise ProtocolError(f"WorkUnit 实例不可变，禁止赋值属性 {name!r}")

    def __delattr__(self, name: str) -> None:
        raise ProtocolError(f"WorkUnit 实例不可变，禁止删除属性 {name!r}")

    def __eq__(self, other: Any) -> Any:
        if not isinstance(other, WorkUnit):
            return NotImplemented
        return self._payload == other._payload

    def __hash__(self) -> int:
        return hash(self._payload)

    def __repr__(self) -> str:
        return f"WorkUnit(project={self._fields['project']!r}, " \
               f"unit_id={self._identity[:12]}...)"

    # ---- 规范化只读属性（返回副本，外部修改不会写回） ---------------------
    @property
    def project(self) -> str:
        return self._fields["project"]

    @property
    def repository(self) -> str:
        return self._fields["repository"]

    @property
    def commit(self) -> str:
        return self._fields["commit"]

    @property
    def inputs(self) -> dict[str, str]:
        return dict(self._fields["inputs"])

    @property
    def environment(self) -> dict[str, str]:
        return dict(self._fields["environment"])

    @property
    def command(self) -> list[str]:
        return list(self._fields["command"])

    @property
    def outputs(self) -> list[str]:
        return list(self._fields["outputs"])

    @property
    def params(self) -> Any:
        return _deep_copy(self._fields["params"])

    @property
    def validator(self) -> dict[str, Any]:
        return _deep_copy(self._fields["validator"])

    def to_dict(self) -> dict[str, Any]:
        """返回规范化后的原始定义（含 schema_version，全部为副本），可直接再次 from_dict。"""
        out: dict[str, Any] = {"schema_version": SCHEMA_VERSION}
        for field in _IDENTITY_FIELDS:
            out[field] = _deep_copy(self._fields[field])
        return out

    @property
    def identity(self) -> str:
        """稳定 unit id：canonical JSON（排序键、紧凑分隔符）的 SHA256。

        身份在构造时由不可变 canonical 字符串一次算定，之后永不重算，
        因此外部任何操作都不可能让 identity 与 unit 定义失配。
        """
        return self._identity


def canonical_json(unit: WorkUnit) -> str:
    """对参与身份的全部语义字段做确定性序列化（键排序、不转义、禁 NaN）。"""
    if not isinstance(unit, WorkUnit):
        raise ProtocolError("canonical_json 只接受 WorkUnit 实例")
    return unit._payload


def canonical_digest(value: Any) -> str:
    """任意 JSON 安全值的 canonical JSON（排序键、紧凑、禁 NaN）SHA256 hex。"""
    guarded = json_guard(value, "value")
    return hashlib.sha256(_dumps_canonical(guarded).encode("utf-8")).hexdigest()


def coerce(unit: WorkUnit | dict[str, Any]) -> WorkUnit:
    """接受 WorkUnit 或原始 dict；dict 走完整严格校验。

    原样放行 WorkUnit 实例是安全的：实例只可能经由唯一校验通道构造，
    已保证合法且 identity 内在一致。
    """
    if isinstance(unit, WorkUnit):
        return unit
    return WorkUnit.from_dict(unit)


def unit_id(unit: WorkUnit | dict[str, Any]) -> str:
    """便捷入口：任意合法表示下的稳定身份。"""
    return coerce(unit).identity
