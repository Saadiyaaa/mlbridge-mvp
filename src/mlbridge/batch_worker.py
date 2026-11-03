"""远端批（unit_batch）执行器：随 remote_worker 上传的 standalone 模块。

限定与边界（先读齐本文档再扩展接口；详细说明见 docs/UNIT_EXECUTION.md）：
- 纯 stdlib，不 import mlbridge 包、不需要 pip；由 `mlbridge run
  --unit-batch` 的 launch **显式 stage** 才能在远端使用（当前已接入；
  缺失时 remote_worker 会得到清晰的 fail-closed 错误）。
- 批契约 v2（schema_version=2）：plan['unit_batch'] = {batch_id, schema_version, units:[...]}
  （可选 manifest_limits/publish_timeout_seconds）。每个 item 为
  {unit_id, attempt_id, spec, execution:{timeout_seconds}}；spec 是
  WorkUnit.to_dict 的合法 canonical 定义；unit_id 必须等于**本模块重算**
  的 canonical JSON SHA256，绝不信任提交方给出的哈希。
- 所有 schema/身份/绑定/保留路径问题在 clone/restore/setup 任何副作用
  之前 fail closed（validate_batch 是纯校验函数）。session/project/
  repository/commit/restic_repository/setup 由现有 plan 绑定并要求与全部
  unit spec 一致。environment['setup_sha256'] 必须等于 canonical JSON
  SHA256(plan.setup)。全部 unit 的 environment / inputs 必须一致，inputs
  名称排序后的 snapshot 列表必须与 plan.inputs 顺序完全一致（保护快照
  覆盖顺序语义）；unit 输出树互相不得重叠，outputs 非空。
- 本执行器不注入、不改写任何科学参数：params 只是身份字段（
  协议字段，与其它科学参数一样只是 params 里的普通键），command 自行
  负责读取；worker 不声称研究程序真的使用了它们。
- validator v1 仅支持 {'kind':'command','command':[argv],
  'timeout_seconds':正有限数}：以工作区为 cwd 运行、退出码 0 即通过；
  不支持 metric 阈值等未实现 kind，不做 shell 拼接。
- 执行流程：一次 clone/restore/setup → 逐 unit（顺序）：输出预检 →
  command（有超时；超时/正常退出都确认自建组无遗留存活成员）→
  validator（同规则）→ 输出完整性检查（拒绝 symlink/.git/特殊文件/
  越界祖先；遍历按条目（文件+目录）与实际读到的字节累计计预算，
  含目录项遍历与逐块哈希的时间成本）→ 相对路径 SHA256 manifest
  （写前对所有祖先 lstat，拒绝 symlink/特殊文件，防越界写入；创建
  目录与 manifest 文件均 fsync）→ restic backup（outputs+manifest，
  输出流入临时文件不驻内存，错误摘要仅取有界 stderr 尾部；任一
  情况下拿不到确认的 64hex snapshot_id 都记 PUBLISH_UNKNOWN，不重算
  不重传，tags 绑定 batch/unit/attempt 供对账）→ 原子 unit receipt →
  batch 进度更新 → 下一 unit。
- 失败语义：已发布 unit 的回执保持不变；真实 command/validator/输出
  检查失败记 TASK_FAILED；publish 不确定记 PUBLISH_UNKNOWN；receipt
  写入失败绝不吞掉、记 RECEIPT_FAILED（保留 snapshot_id）；单元失败
  后后续单元记 NOT_STARTED，绝不虚构失败。批终态顺序：先写 batch
  状态记录（含逐单元 outcome 的权威对账依据），再写共享 STATE 终态；
  终态已写后 batch 记录持久化失败不改变逐单元事实，只显式标注控制面
  持久化不确定（未持久确认时不得宣称整批可安全停机）。
- 所有权：跨进程互斥 by remote_worker 持有的 BASE 执行锁（全程唯一
  所有权原语；非 owner 绝不写 STATE/删凭据）。本模块不自建锁；claim
  只负责拒绝"同 batch 已有运行/终态记录”（重复 worker 在锁外已被拒
  绝，提倡非阻塞拒绝而非阻塞等待）。不是自动清理旧 clone 重跑。
- 进程组：command/validator/restic 均以 start_new_session 启动，pgid
  捕获于启动时（= leader pid，leader 退出后组仍可寻址）；终止由组活性
  决定（TERM→按组活性有界宽限→KILL→有界确认组无存活非-zombie 成员），
  leader
  退出后组内仍有存活成员也是 unit 失败（防 hash/backup 期间仍在写
  文件）；绝不按名字杀、不做全系统命令行匹配。可信 command 约束：
  command 不得 daemonize —— 手动 setsid 逃离进程组的孙进程超出本执行
  器属地保护，不承诺通用沙箱。
- 回执是 worker 自报，不等于本地独立产物验收；本地台账（ledger）只
  到“身份/摘要契约校验后写入”这一层（opt-in：`mlbridge run
  --unit-batch`，对账/停止规则见 docs/RECOVERY.md），不做本地独立
  产物验收。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import signal
import stat
import subprocess
import sys
import tempfile
import time
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

SCHEMA_VERSION = 2
#: unit RECEIPT format version — deliberately INDEPENDENT of ``SCHEMA_VERSION``.
#: The receipt carries ids/digests only (never the spec), so its shape is
#: versioned independently: receipts already on disk stay readable.
RECEIPT_SCHEMA_VERSION = 1

IDENTITY_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,62}$")
COMMIT_RE = re.compile(r"^[a-f0-9]{40}$")
SNAPSHOT_RE = re.compile(r"^[a-f0-9]{64}$")

# 与 mlbridge.protocol 完全一致的规范字段（本地镜像，跨包一致性由测试保证）。
SPEC_FIELDS = ("schema_version", "project", "repository", "commit", "inputs",
               "environment", "command", "params", "outputs", "validator")
SPEC_FIELD_SET = frozenset(SPEC_FIELDS)

BATCH_FIELDS = frozenset({"batch_id", "schema_version", "units",
                          "manifest_limits", "publish_timeout_seconds"})
ITEM_FIELDS = frozenset({"unit_id", "attempt_id", "spec", "execution"})
EXECUTION_FIELDS = frozenset({"timeout_seconds"})
VALIDATOR_FIELDS = frozenset({"kind", "command", "timeout_seconds"})
LIMIT_FIELDS = frozenset({"max_files", "max_bytes", "timeout_seconds"})

# 工作区内专用运行时元数据目录（备份快照中的相对路径契约见 docs）。
METADATA_DIR = ".mlbridge-metadata"
MANIFEST_NAME = "manifest.json"
RECEIPTS_DIRNAME = "mlbridge_receipts"

DEFAULT_MAX_FILES = 10000
DEFAULT_MAX_BYTES = 1 << 30  # 1 GiB
DEFAULT_MANIFEST_TIMEOUT = 600.0
DEFAULT_PUBLISH_TIMEOUT = 1800.0

# 超时后先 SIGTERM 整个进程组，有界宽限，再 SIGKILL；KILL 后仍有有界
# 确认窗口。判断依据是组活性（非-zombie 存活成员），绝不按名字杀进程。
TERMINATION_GRACE = 10.0
KILL_GRACE = 5.0

# restic 失败摘要最多保留的 stderr 尾部字节（不含 env/凭据内容）。
_PUBLISH_TAIL_BYTES = 2000

OUTCOME_SUCCEEDED = "SUCCEEDED"
OUTCOME_TASK_FAILED = "TASK_FAILED"
OUTCOME_PUBLISH_UNKNOWN = "PUBLISH_UNKNOWN"
OUTCOME_RECEIPT_FAILED = "RECEIPT_FAILED"
OUTCOME_NOT_STARTED = "NOT_STARTED"


class BatchError(RuntimeError):
    """批计划非法、身份不匹配或执行环境不满足契约；永远 fail closed。"""


class BatchClaimError(BatchError):
    """该 batch 在本机已有执行/终态记录，拒绝重复执行。"""


class UnitFailure(Exception):
    """单个 unit 的确定性失败/不确定发布；由 execute_batch 汇报为 outcome。"""

    def __init__(self, unit_id: str, stage: str, outcome: str,
                 detail: str, snapshot_id: str | None = None) -> None:
        super().__init__(detail)
        self.unit_id = unit_id
        self.stage = stage
        self.outcome = outcome
        self.detail = detail
        self.snapshot_id = snapshot_id


# ---------------------------------------------------------------------------
# canonical JSON / 校验原语（protocol 本地镜像）
# ---------------------------------------------------------------------------

def _dumps_canonical(payload: Any) -> str:
    try:
        return json.dumps(payload, sort_keys=True, ensure_ascii=False,
                          separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise BatchError(f"无法转成 canonical JSON: {exc}") from exc


def _json_guard(value: Any, where: str) -> Any:
    """递归校验 JSON 安全值（与协议同规则）：拒绝 NaN/Infinity、非 str 键、
    非 JSON 容器类型；返回独立副本。"""
    if value is None or isinstance(value, bool) or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise BatchError(f"{where} 含 NaN/Infinity，非 JSON 合法值")
        return value
    if isinstance(value, list):
        return [_json_guard(item, where) for item in value]
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise BatchError(f"{where} 的键必须是字符串: {key!r}")
            out[key] = _json_guard(item, where)
        return out
    raise BatchError(f"{where} 含非 JSON 类型 {type(value).__name__}")


def _canon_digest(value: Any) -> str:
    return hashlib.sha256(
        _dumps_canonical(_json_guard(value, "value")).encode("utf-8")).hexdigest()


def _ident_name(value: Any, label: str) -> str:
    if not isinstance(value, str) or not IDENTITY_RE.fullmatch(value):
        raise BatchError(
            f"{label} 仅支持 1–63 位字母、数字、下划线、连字符: {value!r}")
    return value


def _hex64(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SNAPSHOT_RE.fullmatch(value):
        raise BatchError(f"{label} 必须是完整 64 位小写十六进制: {value!r}")
    return value


def _commit_hex(value: Any, label: str) -> str:
    if not isinstance(value, str) or not COMMIT_RE.fullmatch(value):
        raise BatchError(f"{label} 必须是完整 40 位小写十六进制: {value!r}")
    return value


def _positive_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BatchError(f"{label} 必须是正有限数字: {value!r}")
    if not math.isfinite(value) or value <= 0:
        raise BatchError(f"{label} 必须是正有限数字（禁止无限等待）: {value!r}")
    return float(value)


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise BatchError(f"{label} 必须是正整数: {value!r}")
    return value


def _normalize_git_url(url: str) -> str:
    """与 core.normalize_git_url 同规则：去首尾空白、尾斜杠和 .git 后缀。"""
    url = url.strip().rstrip("/")
    if url.endswith(".git"):
        url = url[: -len(".git")]
    return url


def _checked_relative(value: Any, label: str) -> str:
    """与 core.checked_relative 同规则：拒绝绝对/越界/``.git``/NUL/根目录。"""
    if not isinstance(value, str) or "\0" in value:
        raise BatchError(f"{label} 必须是不含 NUL 的字符串: {value!r}")
    p = Path(value)
    if not value or value in (".", "..") or p.is_absolute() or ".." in p.parts:
        raise BatchError(f"拒绝项目根目录/绝对或越界路径: {value!r}")
    if any(part == ".git" for part in p.parts):
        raise BatchError(f"禁止路径包含 .git 目录组件: {value!r}")
    return p.as_posix().removeprefix("./")


def _argv_array(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise BatchError(f"{label} 必须为非空 argv 数组（元素为非空字符串）")
    for element in value:
        if not isinstance(element, str) or not element:
            raise BatchError(f"{label} 必须为非空 argv 数组（元素为非空字符串）")
        if "\0" in element:
            raise BatchError(f"{label} 元素含 NUL 字节，拒绝: {element!r}")
    return list(value)


# ---------------------------------------------------------------------------
# 规范化 spec（WorkUnit 协议本地镜像；identity 由规范化后 canonical JSON 计算）
# ---------------------------------------------------------------------------

def _normalize_spec(raw: Any) -> tuple[dict[str, Any], str]:
    """严格校验并规范化 unit spec，返回 (规范化字段, identity)。

    校验顺序与拒绝规则镜像 mlbridge.protocol.WorkUnit.from_dict；任何差异
    都必须出现为身份不一致（由 tests/test_remote_batch.py 的镜像测试发现）。
    """
    if not isinstance(raw, dict):
        raise BatchError("spec 必须是 JSON 对象/dict")
    unknown = [key for key in raw
               if not (isinstance(key, str) and key in SPEC_FIELD_SET)]
    if unknown:
        unknown = sorted(unknown, key=repr)
        raise BatchError(f"spec 含未知字段: {unknown}（执行期信息不属于协议）")
    missing = [field for field in SPEC_FIELDS if field not in raw]
    if missing:
        raise BatchError(f"spec 缺少字段: {missing}")
    version = raw["schema_version"]
    if isinstance(version, bool) or not isinstance(version, int) \
            or version != SCHEMA_VERSION:
        raise BatchError(f"未知 spec schema_version: {version!r}；仅支持 {SCHEMA_VERSION}")

    project = _ident_name(raw["project"], "spec.project")

    repo = raw["repository"]
    if not isinstance(repo, str):
        raise BatchError(f"spec.repository 必须是字符串: {repo!r}")
    repository = _normalize_git_url(repo)
    if not repository:
        raise BatchError("spec.repository 不能为空")

    commit = _commit_hex(raw["commit"], "spec.commit")

    raw_inputs = raw["inputs"]
    if not isinstance(raw_inputs, dict) or not raw_inputs:
        raise BatchError("spec.inputs 必须是非空 dict（名称 -> snapshot）")
    inputs: dict[str, str] = {}
    for name, snapshot in raw_inputs.items():
        try:
            name = _ident_name(name, f"spec.inputs 名称 {name!r}")
        except BatchError as exc:
            raise BatchError(f"spec.inputs 名称不合法: {name!r}: {exc}") from exc
        inputs[name] = _hex64(snapshot, f"spec.inputs[{name!r}] snapshot")

    environment = raw["environment"]
    if not isinstance(environment, dict) or not environment:
        raise BatchError("spec.environment 必须是非空 dict（显式环境指纹）")
    for key, value in environment.items():
        if not isinstance(key, str) or not key:
            raise BatchError(f"environment 键必须是非空字符串: {key!r}")
        if not isinstance(value, str):
            raise BatchError(f"environment[{key!r}] 必须是 str 值: {value!r}")

    command = _argv_array(raw["command"], "spec.command")

    raw_outputs = raw["outputs"]
    if not isinstance(raw_outputs, list) or any(not isinstance(x, str) for x in raw_outputs):
        raise BatchError("spec.outputs 必须是相对路径字符串数组")
    outputs = [_checked_relative(path, f"spec.outputs[{i!r}]")
               for i, path in enumerate(raw_outputs)]

    validator = _json_guard(raw["validator"], "spec.validator")
    if not isinstance(validator, dict) or not isinstance(validator.get("kind"), str) \
            or not validator["kind"]:
        raise BatchError('validator 必须是非空 dict 且含非空 "kind" 字段')

    fields = {
        "project": project,
        "repository": repository,
        "commit": commit,
        "inputs": dict(sorted(inputs.items())),
        "environment": dict(environment),
        "command": command,
        "params": _json_guard(raw["params"], "spec.params"),
        "outputs": outputs,
        "validator": validator,
    }
    payload = {"schema_version": SCHEMA_VERSION}
    payload.update(fields)
    canonical = _dumps_canonical(payload)
    identity = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return fields, identity


# ---------------------------------------------------------------------------
# 批计划校验（纯函数，无任何副作用）
# ---------------------------------------------------------------------------

def _validate_limits(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {"max_files": DEFAULT_MAX_FILES, "max_bytes": DEFAULT_MAX_BYTES,
                "timeout_seconds": DEFAULT_MANIFEST_TIMEOUT}
    if not isinstance(raw, dict):
        raise BatchError("manifest_limits 必须是 JSON 对象")
    unknown = sorted(set(raw) - LIMIT_FIELDS)
    if unknown:
        raise BatchError(f"manifest_limits 含未知字段: {unknown}")
    limits = {"max_files": DEFAULT_MAX_FILES, "max_bytes": DEFAULT_MAX_BYTES,
              "timeout_seconds": DEFAULT_MANIFEST_TIMEOUT}
    if raw.get("max_files") is not None:
        limits["max_files"] = _positive_int(raw["max_files"], "manifest_limits.max_files")
    if raw.get("max_bytes") is not None:
        limits["max_bytes"] = _positive_int(raw["max_bytes"], "manifest_limits.max_bytes")
    if raw.get("timeout_seconds") is not None:
        limits["timeout_seconds"] = _positive_number(
            raw["timeout_seconds"], "manifest_limits.timeout_seconds")
    return limits


def _validate_plan_binding(plan: dict[str, Any]) -> dict[str, Any]:
    """校验 plan 层绑定字段；session/project/repository/commit/setup 均必须
    显式给出且具备 executor 可执行的形态。"""
    for key in ("session", "project", "job_id", "repository", "commit",
                "restic_repository", "inputs", "setup"):
        if key not in plan:
            raise BatchError(f"plan 缺少批模式必需字段: {key}")
    binding: dict[str, Any] = {
        "session": _ident_name(plan["session"], "plan.session"),
        "project": _ident_name(plan["project"], "plan.project"),
        "job_id": _ident_name(plan["job_id"], "plan.job_id"),
    }
    repo = plan["repository"]
    if not isinstance(repo, str):
        raise BatchError(f"plan.repository 必须是字符串: {repo!r}")
    binding["repository"] = _normalize_git_url(repo)
    if not binding["repository"]:
        raise BatchError("plan.repository 不能为空")
    binding["commit"] = _commit_hex(plan["commit"], "plan.commit")
    restic_repo = plan["restic_repository"]
    if not isinstance(restic_repo, str) or not restic_repo:
        raise BatchError("plan.restic_repository 必须是非空字符串")
    binding["restic_repository"] = restic_repo

    raw_inputs = plan["inputs"]
    if not isinstance(raw_inputs, list) or not raw_inputs \
            or any(not isinstance(x, str) for x in raw_inputs):
        raise BatchError("plan.inputs 必须是非空 snapshot 字符串数组")
    lowered = []
    for snapshot in raw_inputs:
        try:
            lowered.append(_hex64(snapshot, f"plan.inputs snapshot {snapshot!r}"))
        except BatchError as exc:
            raise BatchError(
                "plan.inputs 必须是完整 64 位十六进制快照 id "
                f"（大小写不敏感比较）: {snapshot!r}") from exc
    binding["inputs"] = lowered

    raw_setup = plan["setup"]
    if not isinstance(raw_setup, list):
        raise BatchError("plan.setup 必须是 argv 数组的数组")
    setup = []
    for setup_cmd in raw_setup:
        setup.append(_argv_array(setup_cmd, "plan.setup 元素"))
    binding["setup"] = setup
    # setup 绑定：canonical JSON SHA256，由每个 unit 的 environment 键认领。
    binding["setup_sha256"] = _canon_digest(setup)
    return binding


def _validate_item(item_raw: Any, index: int, binding: dict[str, Any]) -> dict[str, Any]:
    label = f"unit_batch.units[{index}]"
    if not isinstance(item_raw, dict):
        raise BatchError(f"{label} 必须是 JSON 对象")
    unknown = sorted(set(item_raw) - ITEM_FIELDS)
    if unknown:
        raise BatchError(f"{label} 含未知字段: {unknown}")
    missing = sorted(ITEM_FIELDS - set(item_raw))
    if missing:
        raise BatchError(f"{label} 缺少字段: {missing}")

    unit_id = item_raw["unit_id"]
    if not isinstance(unit_id, str) or not SNAPSHOT_RE.fullmatch(unit_id):
        raise BatchError(
            f"{label}.unit_id 必须是完整 64 位小写十六进制（不能信任任意哈希）: {unit_id!r}")
    attempt_id = _ident_name(item_raw["attempt_id"], f"{label}.attempt_id")

    execution = item_raw["execution"]
    if not isinstance(execution, dict):
        raise BatchError(f"{label}.execution 必须是 JSON 对象")
    unknown = sorted(set(execution) - EXECUTION_FIELDS)
    if unknown:
        raise BatchError(f"{label}.execution 含未知字段: {unknown}")
    if "timeout_seconds" not in execution:
        raise BatchError(f"{label}.execution 缺少 timeout_seconds（禁止无限等待）")
    timeout = _positive_number(execution["timeout_seconds"],
                               f"{label}.execution.timeout_seconds")

    fields, identity = _normalize_spec(item_raw["spec"])
    if identity != unit_id:
        raise BatchError(
            f"{label}.unit_id 与 spec canonical JSON SHA256 不一致: "
            f"给定 {unit_id}，本机重算 {identity}")

    if fields["project"] != binding["project"]:
        raise BatchError(
            f"{label} spec.project {fields['project']!r} 与 plan.project "
            f"{binding['project']!r} 不一致")
    if fields["commit"] != binding["commit"]:
        raise BatchError(f"{label} spec.commit 与 plan.commit 不一致")
    if fields["repository"] != binding["repository"]:
        raise BatchError(
            f"{label} spec.repository {fields['repository']!r} 与 plan.repository "
            f"规范化 {binding['repository']!r} 不一致")
    if not fields["outputs"]:
        raise BatchError(f"{label} outputs 不能为空（本批要求所有 unit outputs 非空）")

    environment = fields["environment"]
    digest = environment.get("setup_sha256")
    if not isinstance(digest, str) or digest != binding["setup_sha256"]:
        raise BatchError(
            f"{label} environment['setup_sha256'] 缺失或与 canonical JSON "
            f"SHA256(plan.setup)={binding['setup_sha256']} 不一致")

    # 快照覆盖顺序语义：plan.inputs 的顺序就是 restore 顺序，必须等于
    # unit inputs 名称排序后的 snapshot 列表，否则拒绝而不是猜测。
    spec_input_order = [fields["inputs"][name] for name in sorted(fields["inputs"])]
    if spec_input_order != binding["inputs"]:
        raise BatchError(
            f"{label} spec.inputs 名称排序后的 snapshot 列与 plan.inputs 顺序不一致: "
            f"{spec_input_order} != {binding['inputs']}")

    validator = fields["validator"]
    unknown = sorted(set(validator) - VALIDATOR_FIELDS)
    if unknown:
        raise BatchError(
            f"{label} validator 含未知字段 {unknown}；validator v1 仅支持 kind=command")
    if validator["kind"] != "command":
        raise BatchError(
            f"{label} validator kind {validator['kind']!r} 未实现；"
            "validator v1 仅支持 {'kind':'command','command','timeout_seconds'}")
    if "command" not in validator or "timeout_seconds" not in validator:
        raise BatchError(f"{label} validator 缺少 command/timeout_seconds")
    validator_command = _argv_array(validator["command"], f"{label} validator.command")
    validator_timeout = _positive_number(
        validator["timeout_seconds"], f"{label} validator.timeout_seconds")

    return {
        "index": index,
        "unit_id": unit_id,
        "attempt_id": attempt_id,
        "timeout_seconds": timeout,
        "spec": {"schema_version": SCHEMA_VERSION, **fields},
        "command": fields["command"],
        "outputs": fields["outputs"],
        "environment": environment,
        "inputs": fields["inputs"],
        "validator_command": validator_command,
        "validator_timeout": validator_timeout,
        "validator_spec_sha256": _canon_digest(validator),
        "validator_kind": validator["kind"],
    }


def _outputs_overlap(a: str, b: str) -> bool:
    """路径组件级判断两棵输出树是否重叠（绝不做字符串裸前缀比较）。"""
    pa, pb = Path(a).parts, Path(b).parts
    if pa == pb:
        return True
    shorter, longer = (pa, pb) if len(pa) <= len(pb) else (pb, pa)
    return longer[:len(shorter)] == shorter


def _check_batch_invariants(items: list[dict[str, Any]],
                            _batch_id: str) -> None:
    """跨 unit 不变量：重复 id、environment/inputs 一致性、输出树互斥、
    与批元数据目录互斥。"""
    first = items[0]
    for item in items:
        if item["environment"] != first["environment"]:
            raise BatchError("批内所有 unit 的 environment 必须一致")
        if item["inputs"] != first["inputs"]:
            raise BatchError("批内所有 unit 的 inputs 必须一致")
    seen_units: set[str] = set()
    seen_attempts: set[str] = set()
    for item in items:
        uid = item["unit_id"]
        if uid in seen_units:
            raise BatchError(f"同一 unit 在批内重复出现: {uid}")
        seen_units.add(uid)
        if item["attempt_id"] in seen_attempts:
            raise BatchError(f"attempt_id 必须全批唯一，重复: {item['attempt_id']}")
        seen_attempts.add(item["attempt_id"])
    for i, item in enumerate(items):
        for output in item["outputs"]:
            if any(part == METADATA_DIR for part in Path(output).parts):
                raise BatchError(
                    f"unit {item['unit_id']} 输出路径涉及批元数据目录 "
                    f"{METADATA_DIR!r}（reserved，禁止冲突）: {output!r}")
        for other in items[i + 1:]:
            for a in item["outputs"]:
                for b in other["outputs"]:
                    if _outputs_overlap(a, b):
                        raise BatchError(
                            f"unit {item['unit_id']} 与 unit {other['unit_id']} 的"
                            f"输出树重叠: {a!r} 与 {b!r}") from None


def validate_batch(plan: Any) -> dict[str, Any]:
    """批计划 v2 纯校验；返回 executor 消费的规范化批上下文。

    该函数绝不触碰文件系统/子进程/网络；validate 通过前，调用方不得
    执行 clone/restore/setup 等任何副作用（fail closed 契约）。
    """
    if not isinstance(plan, dict):
        raise BatchError("plan 必须是 JSON 对象")
    raw = plan.get("unit_batch")
    if raw is None:
        raise BatchError(
            "unit_batch 为 null/缺失，批模式必须显式提供对象；"
            "显式 null 在任何 clone/restore/setup 副作用之前被拒绝")
    if not isinstance(raw, dict):
        raise BatchError("unit_batch 必须是 JSON 对象")
    unknown = sorted(set(raw) - BATCH_FIELDS)
    if unknown:
        raise BatchError(f"unit_batch 含未知字段: {unknown}")
    missing = sorted({"batch_id", "schema_version", "units"} - set(raw))
    if missing:
        raise BatchError(f"unit_batch 缺少字段: {missing}")
    version = raw["schema_version"]
    if isinstance(version, bool) or not isinstance(version, int) \
            or version != SCHEMA_VERSION:
        raise BatchError(f"未知 unit_batch schema_version: {version!r}；仅支持 {SCHEMA_VERSION}")
    batch_id = _ident_name(raw["batch_id"], "unit_batch.batch_id")
    units_raw = raw["units"]
    if not isinstance(units_raw, list) or not units_raw:
        raise BatchError("unit_batch.units 必须是非空数组")
    limits = _validate_limits(raw.get("manifest_limits"))
    publish_timeout = DEFAULT_PUBLISH_TIMEOUT
    if raw.get("publish_timeout_seconds") is not None:
        publish_timeout = _positive_number(
            raw["publish_timeout_seconds"], "publish_timeout_seconds")

    binding = _validate_plan_binding(plan)
    items = [_validate_item(item_raw, index, binding)
             for index, item_raw in enumerate(units_raw)]
    _check_batch_invariants(items, batch_id)

    return {
        "batch_id": batch_id,
        "schema_version": SCHEMA_VERSION,
        "session": binding["session"],
        "project": binding["project"],
        "job_id": binding["job_id"],
        "repository": binding["repository"],
        "commit": binding["commit"],
        "restic_repository": binding["restic_repository"],
        "inputs": binding["inputs"],
        "setup": binding["setup"],
        "setup_sha256": binding["setup_sha256"],
        "manifest_limits": limits,
        "publish_timeout_seconds": publish_timeout,
        "units": items,
    }


# ---------------------------------------------------------------------------
# batch 状态记录（同批重复 worker 拒绝 + 逐单元进度）
# ---------------------------------------------------------------------------
# 跨进程互斥由调用方 remote_worker 全程持有的 BASE 执行锁提供；本模块
# 的所有写函数必须在所有权内调用。重复 worker 优先
# 在锁处非阻塞拒绝，claim 只拒绝“同 batch 已有运行/终态记录”。


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, data: dict[str, Any], *, indent: int = 2) -> None:
    """原子替换写入 JSON（0600 临时文件 O_EXCL + fsync + os.replace + 目录 fsync）。

    若 os.replace 已执行但目录 fsync 失败：变更可能已提交，此时以磁盘重读
    为准报告；绝不声称"完全未写入"。
    """
    try:
        text = json.dumps(data, ensure_ascii=False, indent=indent, sort_keys=True,
                          allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise BatchError(f"内容无法序列化为 JSON，拒绝写入: {path}: {exc}") from exc
    tmp_suffix = f".tmp-{os.getpid()}-{time.monotonic_ns()}"
    tmp = path.with_name(path.name + tmp_suffix)
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        with suppress(OSError):
            os.unlink(tmp)
        raise
    try:
        dir_fd = os.open(path.parent, os.O_RDONLY)
    except OSError as exc:
        raise BatchError(
            f"文件已原子替换且磁盘内容已是新状态，但目录无法打开以 fsync，"
            f"掉电后持久性不确定: {path} ({exc})") from exc
    try:
        os.fsync(dir_fd)
    except OSError as exc:
        with suppress(OSError):
            on_disk = path.read_text(encoding="utf-8")
        if on_disk != text:
            raise BatchError(
                f"目录 fsync 失败且重读内容与预期不一致，磁盘状态必须人工检查: "
                f"{path} ({exc})") from exc
    finally:
        os.close(dir_fd)


def _record_batch_id(path: Path) -> str:
    name = path.name
    if not name.startswith('mlbridge_batch_state_') or not name.endswith('.json'):
        raise BatchError(f'batch 状态记录文件名不符合契约: {path}')
    return name[len('mlbridge_batch_state_'):-len('.json')]


def _load_batch_record(path: Path) -> dict[str, Any]:
    """读取并校验本 batch 的状态记录；不属于本 batch 则拒绝改写。"""
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except (ValueError, OSError) as exc:
        raise BatchError(
            f"batch 状态记录无法读取（需人工调查）：{path}: {exc}") from exc
    expected = _record_batch_id(path)
    if not isinstance(data, dict) or data.get('batch_id') != expected:
        raise BatchError(
            f"batch 状态记录不属于本 batch（拒绝覆盖）: {path}")
    if not isinstance(data.get('units'), dict):
        raise BatchError(f"batch 状态记录缺少 units 映射: {path}")
    return data


def batch_state_path(base: Path, batch_id: str) -> Path:
    _ident_name(batch_id, 'batch_id')
    return base / f"mlbridge_batch_state_{batch_id}.json"


def claim_batch(batch: dict[str, Any], base: Path) -> Path:
    """声明本 batch 的执行所有权；已有该 batch 的运行/终态记录则拒绝。

    前提：调用方已持有 BASE 执行锁（remote_worker）。claim 初始把全部
    unit 记为 NOT_STARTED（绝不预标 RUNNING），实际启动再以
    _mark_unit_started 记录当前 unit。拒绝不是自动清理旧 clone 重跑：
    请人工对账 receipt 与快照（tags 里绑定了 batch/unit/attempt）后，
    确认可安全废弃时再手动移除该状态文件；重复执行绝不覆写已有批记录
    或结果。
    """
    path = batch_state_path(base, batch['batch_id'])
    if path.exists():
        raise BatchClaimError(
            f"batch {batch['batch_id']!r} 在本机已有执行记录（{path.name}，运行或终态）；"
            "拒绝重复执行同一 batch。当前 engine 不自动 resume、不自动清理旧 clone："
            "请先人工对账 mlbridge_receipts 与快照 tags（mlbridge-batch-*/"
            "mlbridge-unit-*/mlbridge-attempt-*），确认安全后再手动移除状态文件。")
    record = {
        "schema_version": SCHEMA_VERSION,
        "batch_id": batch["batch_id"],
        "session": batch["session"],
        "status": "running",
        "started": _now(),
        "updated": _now(),
        "current_unit": None,
        "units": {item["unit_id"]: {"attempt_id": item["attempt_id"],
                                    "outcome": "NOT_STARTED"}
                  for item in batch["units"]},
    }
    _atomic_write_json(path, record)
    return path


def _mark_unit_started(path: Path, *, batch_id: str, unit_id: str) -> None:
    """记录“当前单元实际已启动”到持久批进度（控制面事实，绝不吞掉）。"""
    data = _load_batch_record(path)
    unit = data['units'].get(unit_id)
    if unit is None:
        raise BatchError(
            f"batch 记录 units 缺少本 unit（拒绝写入）: {unit_id} @ {path}")
    unit['outcome'] = 'RUNNING'
    data['current_unit'] = unit_id
    data['updated'] = _now()
    _atomic_write_json(path, data)


def _update_batch_state(path: Path, *, status: str | None,
                        outcomes: dict[str, Any], detail: str | None,
                        current_unit: str | None = None) -> None:
    """把逐单元 outcome 与批状态写入持久记录。unit_outcomes 总是补齐全
    量 unit（未启动者标 NOT_STARTED），保证任何阶段的快照自洽。"""
    data = _load_batch_record(path)
    if status is not None:
        data['status'] = status
    data['updated'] = _now()
    combined: dict[str, Any] = dict(outcomes)
    for uid, unit in data['units'].items():
        if uid not in combined:
            combined[uid] = {
                'unit_id': uid,
                'outcome': 'NOT_STARTED',
                'attempt_id': unit.get('attempt_id'),
            }
    data['unit_outcomes'] = combined
    data['current_unit'] = current_unit
    if detail is not None:
        data['detail'] = detail
    _atomic_write_json(path, data)


def receipt_path(base: Path, batch_id: str, unit_id: str) -> Path:
    _ident_name(batch_id, "batch_id")
    _hex64(unit_id, "unit_id")
    return base / RECEIPTS_DIRNAME / batch_id / f"{unit_id}.json"


# ---------------------------------------------------------------------------
# 进程执行原语（进程组自清理；绝不按名字杀进程）
# ---------------------------------------------------------------------------

def _group_alive(pgid: int) -> bool:
    """进程组是否还有任何成员（含 zombie；killpg(sig=0) 探测）。"""
    try:
        os.killpg(pgid, 0)
    except (ProcessLookupError, OSError):
        return False
    return True


def _group_members(pgid: int) -> list[tuple[int, str]]:
    """枚举**本 worker 自建进程组**内尚存成员 (pid, state)。

    仅当 pgid 来源于本执行器 start_new_session 启动的进程时才会被调用；
    扫描范围限于 /proc 数字 pid 的单次定长读取，按 pgrp 字段过滤，绝不
    读命令行、绝不按名字匹配。
    """
    members: list[tuple[int, str]] = []
    try:
        names = os.listdir('/proc')
    except OSError:
        return members
    for name in names:
        if not name.isdigit():
            continue
        try:
            with open(f'/proc/{name}/stat', 'rb') as handle:
                raw = handle.read()
        except OSError:
            continue  # 进程已消失
        close = raw.rfind(b')')
        if close < 0:
            continue
        fields = raw[close + 2:].split()
        # /proc/<pid>/stat: pid (comm) state ppid pgrp ...
        if len(fields) < 3:
            continue
        try:
            pgrp = int(fields[2])
        except ValueError:
            continue
        if pgrp == pgid:
            members.append((int(name), fields[0].decode('ascii', 'replace')))
    return members


def _live_members(pgid: int) -> list[int]:
    """组内存活（非-zombie）成员 pid；zombie 无地址空间且由其父/init
    收尸，不算运行成员。先用组探测短路，仅在存在成员时才枚举分类。"""
    if not _group_alive(pgid):
        return []
    return [pid for pid, state in _group_members(pgid) if state != 'Z']


def _terminate_group(pgid: int) -> bool:
    """有界终止整个自建进程组：SIGTERM → 宽限间持续检查**组活性** →
    SIGKILL → KILL 后仍有界确认组内已无存活（非-zombie）成员。

    pgid 在进程启动时捕获（start_new_session 保证 pgid == leader pid）；
    leader 先退出时组内孙进程仍属同一组、仍可 killpg 寻址，清理继续
    有效。返回组内是否已无活成员；确认失败时调用方必须如实报告，
    绝不声称清理成功。"""
    with suppress(ProcessLookupError, OSError):
        os.killpg(pgid, signal.SIGTERM)
    deadline = time.monotonic() + TERMINATION_GRACE
    while time.monotonic() < deadline:
        if not _live_members(pgid):
            return True
        time.sleep(0.05)
    with suppress(ProcessLookupError, OSError):
        os.killpg(pgid, signal.SIGKILL)
    deadline = time.monotonic() + KILL_GRACE
    while time.monotonic() < deadline:
        if not _live_members(pgid):
            return True
        time.sleep(0.05)
    return False


def _run_inherit(argv: list[str], *, cwd: Path, env: dict[str, str],
                 timeout: float, stage: str, unit_id: str, what: str) -> None:
    """运行 argv（继承 worker 日志），exit 0 = 成功。

    进程组契约：start_new_session 保证新组 leader 是子进程本身，pgid ==
    子进程 pid，启动时捕获；leader 退出后组内遗留孙进程仍可被
    killpg 命中。正常退出也必须确认组内无存活（非-zombie）成员，否则
    终止该组并按失败处理 —— 防止 validator/hash/backup 期间孙进程仍在
    写输出文件。command 绝不 daemonize（手动 setsid 逃离组即超出属地）。"""
    try:
        proc = subprocess.Popen(argv, cwd=str(cwd), env=env,
                                stdin=subprocess.DEVNULL, start_new_session=True)
    except OSError as exc:
        raise UnitFailure(unit_id, stage, OUTCOME_TASK_FAILED,
                          f"无法启动 {what}: {exc}") from exc
    pgid = proc.pid
    try:
        proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        cleaned = _terminate_group(pgid)
        with suppress(subprocess.SubprocessError, OSError):
            proc.wait()
        detail = (
            f"{what} 超时（>{timeout}s），已终止并收尾整个进程组"
            if cleaned else
            f"{what} 超时（>{timeout}s），终止后仍无法确认自建进程组内"
            "无存活成员，需人工检查残留进程")
        raise UnitFailure(unit_id, stage, OUTCOME_TASK_FAILED, detail) from exc
    # leader 正常退出：组内遗留孙进程必须清理并按失败处理，绝不视为成功。
    leftover = _live_members(pgid)
    if leftover:
        cleaned = _terminate_group(pgid)
        detail = (f"{what} 退出码 {proc.returncode} 但自建进程组内仍有存活"
                  f"成员 {leftover[:5]}；已清理进程组并按失败处理（防"
                  "validator/hash/backup 期间继续写文件）")
        if not cleaned:
            detail += "；进程组清理未能最终确认，需人工检查残留进程"
        raise UnitFailure(unit_id, stage, OUTCOME_TASK_FAILED, detail)
    if proc.returncode != 0:
        raise UnitFailure(unit_id, stage, OUTCOME_TASK_FAILED,
                          f"{what} 退出码 {proc.returncode}")


# ---------------------------------------------------------------------------
# 输出完整性检查与 manifest
# ---------------------------------------------------------------------------

class _ManifestBudget:
    """manifest 统计预算（条目/字节/时间）：触碰即拒绝，绝不静默漏项。

    - 条目预算占用文件与目录（目录项遍历、空目录洪泛都计入）；
    - 字节预算按哈希循环**实际读到的字节**累计（防文件并发变大逃逸；
      单文件声明输出同样受限）；
    - 时间预算在目录遍历与每个哈希块读取前检查。

    预算字段就是 manifest_limits 已有的 max_files/max_bytes/timeout_seconds，
    不另设新框架。"""

    def __init__(self, unit_id: str, limits: dict[str, Any]) -> None:
        self._unit_id = unit_id
        self._limit_seconds = limits["timeout_seconds"]
        self._max_files = limits["max_files"]
        self._max_bytes = limits["max_bytes"]
        self._deadline = time.monotonic() + self._limit_seconds
        self.entries = 0
        self.bytes = 0

    def _fail(self, detail: str) -> None:
        raise UnitFailure(self._unit_id, "outputs_check", OUTCOME_TASK_FAILED,
                          detail)

    def check_time(self) -> None:
        if time.monotonic() > self._deadline:
            self._fail("manifest 遍历/哈希超出时间预算 "
                       f"{self._limit_seconds}s；拒绝而非静默漏项")

    def count_entry(self) -> None:
        self.entries += 1
        if self.entries > self._max_files:
            self._fail("输出树条目（文件+目录）超过 manifest_limits.max_files="
                       f"{self._max_files}；拒绝而非静默漏项")

    def add_bytes(self, count: int) -> None:
        self.bytes += count
        if self.bytes > self._max_bytes:
            self._fail("输出字节数（按实际读到）超过 manifest_limits.max_bytes="
                       f"{self._max_bytes}；拒绝而非静默漏项")


def _hash_file_bounded(path: Path, budget: _ManifestBudget) -> str:
    """逐块哈希：每块读取前检查时间预算，按实际读到的字节计预算。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            budget.check_time()
            chunk = handle.read(1 << 16)
            if not chunk:
                break
            digest.update(chunk)
            budget.add_bytes(len(chunk))
    return digest.hexdigest()


def _precheck_outputs(root: Path, unit_id: str, outputs: list[str]) -> None:
    """unit 开始前：声明 outputs 一律不得存在（含悬空 symlink），祖先
    链上不得有 symlink（防止写入越界），已发布 outputs 不复用。"""
    for rel in outputs:
        target = root / Path(rel)
        if os.path.lexists(target):
            raise UnitFailure(
                unit_id, "outputs_precheck", OUTCOME_TASK_FAILED,
                f"unit 输出路径已存在，拒绝复用（防止污染已发布成果）: {rel}")
        parts = Path(rel).parts
        for depth in range(1, len(parts)):
            ancestor = root / Path(*parts[:depth])
            if ancestor.is_symlink():
                raise UnitFailure(
                    unit_id, "outputs_precheck", OUTCOME_TASK_FAILED,
                    f"输出路径的祖先组件是符号链接，拒绝写入: {rel}")
        resolved = target.parent.resolve(strict=False)
        if not resolved.is_relative_to(root.resolve()):
            raise UnitFailure(
                unit_id, "outputs_precheck", OUTCOME_TASK_FAILED,
                f"输出路径解析后越过工作区（symlink 逃逸）: {rel}")


def _metadata_parts(batch_id: str, unit_id: str) -> tuple[str, ...]:
    parts = (METADATA_DIR, batch_id, unit_id)
    for part in parts:
        if part in ("", ".", "..") or "/" in part or "\0" in part:
            raise BatchError(f"批元数据路径组件非法（拒绝写入）: {part!r}")
    return parts


def _check_metadata_chain(root: Path, unit_id: str,
                          parts: tuple[str, ...]) -> None:
    """manifest 写入链每一个**已存在**祖先都必须 lstat 为普通目录
    （拒绝 symlink / 普通文件 / fifo 等特殊文件），且整链解析后仍限工作
    区内 —— command 或 validator 造出的父 symlink 无法把 manifest 写到
    workspace 之外。"""
    root_real = Path(os.path.realpath(root))
    for depth in range(1, len(parts) + 1):
        path = root.joinpath(*parts[:depth])
        if os.path.lexists(path):
            st = os.lstat(path)
            if stat.S_ISLNK(st.st_mode):
                raise UnitFailure(
                    unit_id, "metadata_check", OUTCOME_TASK_FAILED,
                    f"批元数据路径组件是符号链接，拒绝写入: "
                    f"{path.relative_to(root)}")
            if not stat.S_ISDIR(st.st_mode):
                raise UnitFailure(
                    unit_id, "metadata_check", OUTCOME_TASK_FAILED,
                    f"批元数据路径组件不是普通目录: {path.relative_to(root)}")
        resolved = Path(os.path.realpath(path))
        if not resolved.is_relative_to(root_real):
            raise UnitFailure(
                unit_id, "metadata_check", OUTCOME_TASK_FAILED,
                f"批元数据路径解析后越过工作区（symlink 逃逸）: "
                f"{path.relative_to(root)}")


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _ensure_metadata_chain(root: Path, unit_id: str,
                           parts: tuple[str, ...]) -> Path:
    """确保 manifest 目录链完整：已存在组件先重验（防命令期 TOCTOU 后
    symlink 换入），缺失组件 mkdir(0o700) 并对新目录条目做父目录 fsync。"""
    _check_metadata_chain(root, unit_id, parts)
    parents_to_fsync: list[Path] = []
    for depth in range(1, len(parts) + 1):
        path = root.joinpath(*parts[:depth])
        if not os.path.lexists(path):
            path.mkdir(mode=0o700)
            parents_to_fsync.append(path.parent)
    for parent in parents_to_fsync:
        _fsync_dir(parent)
    return root.joinpath(*parts)


def _collect_files(root: Path, unit_id: str, rel: str,
                   budget: _ManifestBudget) -> dict[str, str]:
    """遍历一个声明输出（文件或目录），校验内容合法性并产出
    相对路径 → SHA256 文件清单；预算（条目/字节/时间）由调用方跨全部
    outputs 共享，任何上限被触碰都整批拒绝而绝不静默漏项。"""
    collected: dict[str, str] = {}
    base = root / Path(rel)
    try:
        st = os.lstat(base)
    except OSError as exc:
        raise UnitFailure(unit_id, "outputs_check", OUTCOME_TASK_FAILED,
                          f"声明输出无法访问: {rel} ({exc})") from exc
    if stat.S_ISLNK(st.st_mode):
        raise UnitFailure(unit_id, "outputs_check", OUTCOME_TASK_FAILED,
                          f"声明输出本身是符号链接，拒绝: {rel}")
    if not stat.S_ISDIR(st.st_mode) and not stat.S_ISREG(st.st_mode):
        raise UnitFailure(unit_id, "outputs_check", OUTCOME_TASK_FAILED,
                          f"声明输出既不是普通文件也不是目录（特殊文件）: {rel}")
    if stat.S_ISREG(st.st_mode):
        # 单文件声明输出同样走共享条目/字节/时间预算。
        budget.check_time()
        budget.count_entry()
        collected[Path(rel).as_posix()] = _hash_file_bounded(base, budget)
        return collected
    stack: list[Path] = [base]
    while stack:
        current = stack.pop()
        budget.check_time()  # 目录遍历本身计入时间成本（空目录也会检查）
        try:
            with os.scandir(current) as scan:
                entries = sorted(scan, key=lambda e: e.name)
        except OSError as exc:
            raise UnitFailure(
                unit_id, "outputs_check", OUTCOME_TASK_FAILED,
                f"输出目录遍历失败: {current.relative_to(root)} ({exc})") from exc
        for entry in entries:
            budget.check_time()
            budget.count_entry()  # 文件与目录条目都占用条目预算
            if entry.is_symlink():
                raise UnitFailure(
                    unit_id, "outputs_check", OUTCOME_TASK_FAILED,
                    f"输出树内出现符号链接，拒绝: "
                    f"{Path(entry.path).relative_to(root)}")
            name = entry.name
            if name == ".git":
                raise UnitFailure(
                    unit_id, "outputs_check", OUTCOME_TASK_FAILED,
                    f"输出树内出现 .git，拒绝: {entry.path}")
            child = Path(entry.path)
            mode = entry.stat(follow_symlinks=False)
            if stat.S_ISDIR(mode.st_mode):
                stack.append(child)
                continue
            if not stat.S_ISREG(mode.st_mode):
                raise UnitFailure(
                    unit_id, "outputs_check", OUTCOME_TASK_FAILED,
                    f"输出树内有非常规文件（fifo/socket/device），拒绝: "
                    f"{child.relative_to(root)}")
            collected[child.relative_to(root).as_posix()] = \
                _hash_file_bounded(child, budget)
    return collected


def _write_manifest(batch: dict[str, Any], item: dict[str, Any], root: Path,
                    files: dict[str, str]) -> tuple[str, str]:
    manifest_rel = f"{METADATA_DIR}/{batch['batch_id']}/{item['unit_id']}"
    try:
        # 写前对整链重验（拒绝 command 期换入的 symlink），缺失祖先
        # mkdir(0o700) + 父目录 fsync；manifest 文件写完后对 unit 目录
        # fsync，确保发布前 manifest 的目录条目已持久。
        unit_dir = _ensure_metadata_chain(
            root, item["unit_id"],
            _metadata_parts(batch["batch_id"], item["unit_id"]))
        target = unit_dir / MANIFEST_NAME
        payload = {
            "schema_version": SCHEMA_VERSION,
            "batch_id": batch["batch_id"],
            "unit_id": item["unit_id"],
            "attempt_id": item["attempt_id"],
            "session": batch["session"],
            "files": files,
        }
        raw = _dumps_canonical(payload).encode("utf-8")
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            _fsync_dir(unit_dir)
        except BaseException:
            with suppress(OSError):
                os.unlink(target)
            raise
    except UnitFailure:
        raise
    except OSError as exc:
        raise UnitFailure(
            item["unit_id"], "metadata_check", OUTCOME_TASK_FAILED,
            f"manifest 写入/目录创建失败（发布前拒绝）: {exc}") from exc
    return f"{manifest_rel}/{MANIFEST_NAME}", hashlib.sha256(raw).hexdigest()


# ---------------------------------------------------------------------------
# 发布（restic backup）与回执
# ---------------------------------------------------------------------------

def _stderr_tail(handle, limit: int = _PUBLISH_TAIL_BYTES) -> str:
    """有界读取 stderr 尾部（不把无界输出攒入内存；也不读取任何凭据/env）。"""
    try:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - limit))
        return handle.read()
    except OSError:
        return ""


def _publish(batch: dict[str, Any], item: dict[str, Any], root: Path,
             env: dict[str, str], manifest_rel: str) -> str:
    """备份 outputs + manifest；只在拿到确认的 64hex snapshot_id 时确认发布。

    restic 的 stdout/stderr 都直接流入临时文件（不充满 PIPE，不用
    communicate() 把无界输出攒进内存）；诊断摘要只取有界的 stderr 尾部，
    不含环境变量/凭据内容。任何超时/失败/无有效 snapshot id 的路径都记
    PUBLISH_UNKNOWN：本批不声称已发布，也不会自动重算或重新上传；
    tags 绑定 batch/unit/attempt 供后续人工对账。异常终止也走
    _terminate_group 的组活性确认。"""
    cmd = ["restic", "-r", batch["restic_repository"], "backup", "--json",
           "--tag", f"mlbridge-project-{batch['project']}",
           "--tag", f"mlbridge-artifact-{batch['job_id']}",
           "--tag", f"mlbridge-batch-{batch['batch_id']}",
           "--tag", f"mlbridge-unit-{item['unit_id']}",
           "--tag", f"mlbridge-attempt-{item['attempt_id']}",
           *item["outputs"], manifest_rel]
    stdout_f: Any = tempfile.TemporaryFile("w+", encoding="utf-8", errors="replace")
    stderr_f: Any = tempfile.TemporaryFile("w+", encoding="utf-8", errors="replace")
    try:
        try:
            proc = subprocess.Popen(cmd, cwd=str(root), env=env,
                                    stdin=subprocess.DEVNULL,
                                    start_new_session=True,
                                    stdout=stdout_f, stderr=stderr_f)
        except OSError as exc:
            raise UnitFailure(item["unit_id"], "publish", OUTCOME_PUBLISH_UNKNOWN,
                              f"restic 无法启动，发布结果不声称成功（标签已绑定，供对账）"
                              f"：{exc}") from exc
        # start_new_session ⇒ pgid == leader pid（启动即捕获）。
        pgid = proc.pid
        try:
            proc.wait(timeout=batch["publish_timeout_seconds"])
        except subprocess.TimeoutExpired as exc:
            cleaned = _terminate_group(pgid)
            with suppress(subprocess.SubprocessError, OSError):
                proc.wait()
            detail = (f"restic backup 超时（>"
                      f"{batch['publish_timeout_seconds']}s），发布结果未知；"
                      "已终止自建进程组，不重算/不重传，请用 tags 对账")
            if not cleaned:
                detail += "；进程组清理未能最终确认，需人工检查残留进程"
            raise UnitFailure(item["unit_id"], "publish",
                              OUTCOME_PUBLISH_UNKNOWN, detail) from exc
        leftover = _live_members(pgid)
        if leftover:
            _terminate_group(pgid)
            raise UnitFailure(
                item["unit_id"], "publish", OUTCOME_PUBLISH_UNKNOWN,
                f"restic backup 退出码 {proc.returncode} 但自建进程组内仍有"
                f"存活成员 {leftover[:5]}；已清理；发布结果未知，"
                "不自动重算/重传，请用 tags 对账")
        snapshot = ""
        stdout_f.seek(0)
        for line in stdout_f:  # 流式逐行读，不做整文件读入
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if isinstance(event, dict) and event.get("message_type") == "summary":
                found = event.get("snapshot_id", "")
                if isinstance(found, str):
                    snapshot = found.lower()
        if SNAPSHOT_RE.fullmatch(snapshot):
            return snapshot
        tail = _stderr_tail(stderr_f)
        detail = (
            f"restic backup 退出码 {proc.returncode}，未获得合法 64hex "
            "snapshot_id；不声称已发布、不自动重算/重传；请用 "
            "mlbridge-batch-*/mlbridge-unit-*/mlbridge-attempt-* tags 对账；"
            f"restic stderr 尾部（≤{_PUBLISH_TAIL_BYTES} 字节，"
            f"不含环境变量/凭据内容）：{tail}")
        raise UnitFailure(item["unit_id"], "publish",
                          OUTCOME_PUBLISH_UNKNOWN, detail)
    finally:
        with suppress(OSError):
            stdout_f.close()
        with suppress(OSError):
            stderr_f.close()


def _write_receipt(batch: dict[str, Any], item: dict[str, Any], base: Path,
                   snapshot_id: str, manifest_rel: str, manifest_sha: str) -> Path:
    """单 unit 回执（worker 自报）：BASE 下按 unit 原子保存；失败不吞掉，
    以 RECEIPT_FAILED 上抛并保留 snapshot_id 供对账。"""
    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "batch_id": batch["batch_id"],
        "unit_id": item["unit_id"],
        "attempt_id": item["attempt_id"],
        "session": batch["session"],
        "namespace": batch["restic_repository"],
        "snapshot_id": snapshot_id,
        "manifest_sha256": manifest_sha,
        "manifest_path": manifest_rel,
        "validator_spec_sha256": item["validator_spec_sha256"],
        "validator_result": {"kind": item["validator_kind"], "status": "passed"},
        "phase": "published",
        "finished": _now(),
    }
    path = receipt_path(base, batch["batch_id"], item["unit_id"])
    try:
        if os.path.lexists(path):
            raise FileExistsError(f"unit 回执已存在，拒绝覆盖: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(path, receipt, indent=2)
    except OSError as exc:
        raise UnitFailure(
            item["unit_id"], "receipt", OUTCOME_RECEIPT_FAILED,
            f"回执原子写入失败（绝不吞掉、不声称已提交；已确认 snapshot={snapshot_id} "
            f"仍可用 tags 对账）: {path} ({exc})", snapshot_id=snapshot_id) from exc
    return path


# ---------------------------------------------------------------------------
# 批执行主循环
# ---------------------------------------------------------------------------

def _progress(batch: dict[str, Any], outcomes: dict[str, Any],
              current: str | None) -> dict[str, Any]:
    return {
        "batch_id": batch["batch_id"],
        "session": batch["session"],
        "units_total": len(batch["units"]),
        "current_unit": current,
        "outcomes": dict(outcomes),
    }


def _execute_unit(batch: dict[str, Any], item: dict[str, Any], *, root: Path,
                  env: dict[str, str], base: Path,
                  emit: Callable[..., None]) -> dict[str, Any]:
    unit_id = item["unit_id"]
    _precheck_outputs(root, unit_id, item["outputs"])
    _check_metadata_chain(
        root, unit_id, _metadata_parts(batch["batch_id"], unit_id))

    emit("RUNNING", batch=_progress(batch, {}, unit_id),
         note=f"unit {unit_id[:12]} command")
    _run_inherit(item["command"], cwd=root, env=env,
                 timeout=item["timeout_seconds"], stage="command",
                 unit_id=unit_id, what=f"unit {unit_id[:12]} command")

    emit("RUNNING", batch=_progress(batch, {}, unit_id),
         note=f"unit {unit_id[:12]} validator")
    _run_inherit(item["validator_command"], cwd=root, env=env,
                 timeout=item["validator_timeout"], stage="validator",
                 unit_id=unit_id, what=f"unit {unit_id[:12]} validator")

    files: dict[str, str] = {}
    # 预算跨本 unit 全部声明 outputs 共享（多输出总额也受限）。
    budget = _ManifestBudget(unit_id, batch["manifest_limits"])
    for rel in item["outputs"]:
        files.update(_collect_files(root, unit_id, rel, budget))
    manifest_rel, manifest_sha = _write_manifest(batch, item, root, files)

    emit("PUBLISHING", batch=_progress(batch, {}, unit_id),
         note=f"unit {unit_id[:12]} backup")
    snapshot_id = _publish(batch, item, root, env, manifest_rel)
    receipt = _write_receipt(batch, item, base, snapshot_id, manifest_rel,
                             manifest_sha)
    return {
        "unit_id": unit_id,
        "outcome": OUTCOME_SUCCEEDED,
        "attempt_id": item["attempt_id"],
        "snapshot_id": snapshot_id,
        "manifest_sha256": manifest_sha,
        "manifest_path": manifest_rel,
        "receipt": receipt.relative_to(base).as_posix(),
        "stage": "published",
        "finished": _now(),
    }


def execute_batch(batch: dict[str, Any], *, root: Path, env: dict[str, str],
                  base: Path, emit: Callable[..., None],
                  state_path: Path | None = None) -> dict[str, Any]:
    """一次 clone/restore/setup 后按 unit 顺序执行；失败后剩余单元
    NOT_STARTED。终态（SUCCEEDED/FAILED）在本函数内写入，不依赖调用方。

    终态写顺序固定：① 持久批记录（含逐单元 outcome 的权威对账依据）
    → ② 共享 STATE 终态。任何一方失败都显式上抛并标注控制面持久化
    不确定，绝不吞掉；已写成的终态不被后续失败覆写。前提：调用方已
    持有 BASE 执行锁（remote_worker 全程所有权）。"""
    state = state_path or batch_state_path(base, batch["batch_id"])
    outcomes: dict[str, Any] = {}
    started = time.monotonic()
    finalized = False

    def annotate_not_started(note: str) -> None:
        for item in batch["units"]:
            if item["unit_id"] not in outcomes:
                outcomes[item["unit_id"]] = {
                    "unit_id": item["unit_id"],
                    "outcome": OUTCOME_NOT_STARTED,
                    "attempt_id": item["attempt_id"],
                    "note": note,
                }

    def finalize_terminal(*, succeeded: bool, failure=None) -> None:
        nonlocal finalized
        persist_error: BatchError | None = None
        try:
            _update_batch_state(
                state, status="succeeded" if succeeded else "failed",
                outcomes=outcomes,
                detail=None if succeeded else failure["detail"])
        except BatchError as exc:
            persist_error = exc
        if succeeded:
            elapsed = round(time.monotonic() - started, 1)
            fields: dict[str, Any] = {}
            if persist_error is not None:
                fields["control_warning"] = (
                    "批状态记录最终持久化失败；逐单元回执与快照是该批的"
                    "权威依据，未持久确认前不可宣称整批可安全停机")
            emit("SUCCEEDED", elapsed_seconds=elapsed, batch=_progress(
                batch, outcomes, None), finished=_now(), **fields)
        else:
            fields = {"error": (f"unit {failure['unit_id'][:12]} "
                                f"[{failure['stage']}] {failure['detail']}")}
            if persist_error is not None:
                fields["control_warning"] = (
                    "批状态记录最终持久化失败；逐单元回执与快照是该批的"
                    "权威依据，需人工对账")
            emit("FAILED", batch=_progress(batch, outcomes, None),
                 finished=_now(), **fields)
        finalized = True
        if persist_error is not None:
            # 终态已写成、逐单元事实已明确；绝不吞掉持久化错误，也绝不
            # 用本批中间状态冒充"计算失败"。
            print(f"mlbridge: 批状态记录最终持久化失败（控制面不确定，"
                  f"需人工对账 receipts/snapshots）: {persist_error}",
                  file=sys.stderr)
            raise persist_error

    try:
        for item in batch["units"]:
            unit_id = item["unit_id"]
            # 实际启动才记录当前 unit（控制面事实）；绝不预标 RUNNING。
            _mark_unit_started(state, batch_id=batch["batch_id"], unit_id=unit_id)
            try:
                outcomes[unit_id] = _execute_unit(batch, item, root=root, env=env,
                                                  base=base, emit=emit)
            except UnitFailure as failure:
                outcomes[failure.unit_id if failure.unit_id else unit_id] = {
                    "unit_id": unit_id,
                    "outcome": failure.outcome,
                    "attempt_id": item["attempt_id"],
                    "stage": failure.stage,
                    "detail": failure.detail,
                    **({"snapshot_id": failure.snapshot_id}
                       if failure.snapshot_id else {}),
                    "finished": _now(),
                }
                break
            _update_batch_state(state, status=None, outcomes=outcomes,
                                detail=None, current_unit=None)
        else:
            finalize_terminal(succeeded=True)
            return outcomes
        # 有 unit 失败：保持真实失败现场（不清理 clone），剩余 NOT_STARTED。
        annotate_not_started("前序单元失败；本批未启动该单元")
        finalize_terminal(succeeded=False, failure=_first_failure(outcomes))
        return outcomes
    except BaseException as exc:  # progress/启动记录写失败等：绝不吞掉
        if not finalized:
            annotate_not_started(f"执行器内部错误，未启动（{exc}）")
            with suppress(Exception):
                emit("FAILED", error=f"{type(exc).__name__}: {exc}",
                     batch=_progress(batch, outcomes, None), finished=_now())
            with suppress(BatchError):
                _update_batch_state(state, status="failed", outcomes=outcomes,
                                    detail=f"{type(exc).__name__}: {exc}")
        raise


def _first_failure(outcomes: dict[str, Any]) -> dict[str, Any]:
    """从明确记录的逐单元 outcome 中取第一个非 SUCCEEDED；缺字段视为
    执行器错误，绝不盲目默认。"""
    for outcome in outcomes.values():
        if outcome.get("outcome") != OUTCOME_SUCCEEDED:
            unit_id = outcome.get("unit_id")
            stage = outcome.get("stage")
            detail = outcome.get("detail") or outcome.get("note") or ""
            if not isinstance(unit_id, str) or not isinstance(stage, str):
                raise BatchError(f"失败 outcome 缺少身份字段（执行器 bug）:"
                                 f" {outcome}")
            return {"unit_id": unit_id, "stage": stage, "detail": detail}
    raise BatchError("没有可报告的失败")  # pragma: no cover
