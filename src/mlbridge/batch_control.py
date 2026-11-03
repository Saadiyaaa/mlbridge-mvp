"""批模式本地控制面（opt-in）：`run --unit-batch` 的编排与恢复对账。

阅读顺序：docs/RECOVERY.md（ledger 规则）、docs/UNIT_EXECUTION.md（远端
批契约 v1 与回执结构）。本模块只做【本地】控制面；远端执行器是
src/mlbridge/batch_worker.py（standalone，由 launch 显式 stage）。

边界（全部在 docs 中同步说明）：
- 严格 opt-in：legacy `run` 的参数、staging 上传集合与行为逐字节不变；
  只有 `--unit-batch FILE` 走本模块。`--execution-mode` 只改变交付与
  生命周期（detached 分离 / foreground cell 内同步），不改变台账语义；
  不参与 batch_id 身份（跨模式可继续同一批次的恢复对账）。
- GPU 分配前的 fail-closed 校验链：严格 JSON（拒绝重复键 / NaN /
  Infinity）→ 顶层 schema → 每个单元 `protocol.WorkUnit` 校验（跨
  project / 缺必需字段 / params 非法 / 非法路径全部拒绝）→
  `batch_worker.validate_batch`
  纯校验（绑定/镜像身份/输出树/validator v1）。任何一步失败都发生在
  `colab sessions`/`colab new` 之前，不写台账、不分配实例。
- 批次身份 batch_id 默认由「项目 + 批次定义」的 canonical JSON 摘要
  派生（绝不只看文件名）；显式给出的 batch_id 必须与派生值一致，否则
  拒绝——身份永远锚定定义内容，文件名不参与。
- attempt_id 由本模块按台账内该 unit 的下一次 attempt 序号确定性生成
  （`att-<unit_id 前 12 hex>-<seq>`），不写入批定义文件；同一批内容
  再跑（含 --retry-failed 开新 attempt）会得到新的 attempt_id。
- 启动前恢复对账：对每个 unit 用 Ledger.plan 幂等登记；已 completed →
  跳过；未解决 attempt → 先读该 attempt 对应 session 的远端状态/回执：
  只有**确定终态证据**（携带 attempt/session 身份）才 reconcile；成功
  证据还需回执身份/摘要匹配才 commit_result——这只是 worker 自报，
  不是本地独立产物验收。TASK_FAILED → reconcile task_failed，默认
  **不自动重试**（须 CLI 显式 --retry-failed 才开新 attempt）。
  PUBLISH_UNKNOWN / RECEIPT_FAILED / NOT_STARTED / 观测失联 / 无证据
  → 报告 UNRESOLVED 并停止：不新建 attempt、不把资源失联写成
  task_failed（ledger 无 RESOURCE_LOST 状态，资源缺口不假报）、不自动
  stop 任何实例。唯一例外：存在正面执行终局证据（实例确认停止 /
  foreground 哨兵）时，对“从未进入发布不确定区、且 restic tags 核验
  确认无快照”的 attempt 写 execution_abandoned（诚实放弃；见
  settle_interrupted_execution）——观测失联/传输失败绝不构成该证据。
  批终态（SUCCEEDED/FAILED）只是控制面事实，绝不据此把全部 unit 批
  量写成终态。
- 回执缺失/身份不匹配/匿名（无法携带正确身份）→ 拒绝且绝不改台账。
- 只读 `units` 查询（cli.units）：未决 attempt 与 completed 分明；
  本模块提供报告，不提供任何 force reset。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

from . import batch_worker as BW
from . import protocol as P
from .core import BridgeError, normalize_git_url
from .ledger import CorruptLedgerError, Ledger, LedgerError, _loads_strict

#: 批定义文件顶层字段；未知字段拒绝（fail closed）。
BATCH_FILE_FIELDS = frozenset({"schema_version", "project", "environment",
                               "units", "unit_batch_id", "manifest_limits",
                               "publish_timeout_seconds"})
#: 每个单元条目允许的字段：spec/execution 必填，unit_id 可选（必须等
#: 于定义派生的身份）；attempt_id 不接受——由台账序号确定性生成。
UNIT_ITEM_FIELDS = frozenset({"unit_id", "spec", "execution"})

_BATCH_ID_PREFIX = "batch-"

# 逐单元 disposition（报告用）：
DISP_COMPLETED = "completed"                # 台账已提交成果
DISP_QUEUED = "queued"                      # 本次将执行
DISP_RECONCILED_SUCCESS = "reconciled_succeeded"
DISP_RECONCILED_FAILED = "reconciled_failed"
DISP_ABANDONED = "reconciled_abandoned"    # 执行中断后诚实放弃（需显式重试）
DISP_UNRESOLVED = "unresolved"              # 无确定证据，绝不裁决
DISP_UNCOMMITTED_SUCCESS = "uncommitted_success"  # 成功证据在、成果未提交
DISP_UNTRACKED = "untracked"                # 台账缺少该 attempt 记录
DISP_SKIPPED = "skipped_completed"
DISP_QUEUED_RETRY = "queued_retry"          # --retry-failed 显式重跑入队
DISP_HEALED_COMMIT = "healed_commit"        # 崩溃窗口补提交（reconcile 后
                                            # commit 前中断，本次补全）

# execution_abandoned 证据来源（controller 本地事实，绝不伪造远端终态）。
ABANDON_SOURCE = "controller/execution-ended"


class BatchControlError(BridgeError):
    """批定义非法 / 对账无证据 / 策略拒绝（未给 --retry-failed 等）。"""


# ---------------------------------------------------------------------------
# 严格 JSON 与批定义文件解析（全部发生在任何分配/副作用之前）
# ---------------------------------------------------------------------------

def load_batch_file(path: Path) -> dict[str, Any]:
    """读取批定义文件并严格 JSON 解析（拒绝重复键 / NaN / Infinity）。"""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise BatchControlError(f"无法读取批定义文件 {path}: {exc}") from exc
    try:
        data = _loads_strict(text, path)
    except CorruptLedgerError as exc:
        raise BatchControlError(f"批定义文件不是合法 JSON（拒绝 NaN/Infinity/"
                                f"重复键）: {exc}") from exc
    if not isinstance(data, dict):
        raise BatchControlError("批定义必须是 JSON 对象")
    unknown = sorted(set(data) - BATCH_FILE_FIELDS)
    if unknown:
        raise BatchControlError(f"批定义含未知字段: {unknown}")
    missing = sorted({"schema_version", "project", "environment", "units"}
                     - set(data))
    if missing:
        raise BatchControlError(f"批定义缺少字段: {missing}")
    version = data["schema_version"]
    if isinstance(version, bool) or not isinstance(version, int) \
            or version != P.SCHEMA_VERSION:
        raise BatchControlError(
            f"批定义 schema_version 必须是 {P.SCHEMA_VERSION}: {version!r}")
    if not isinstance(data["units"], list) or not data["units"]:
        raise BatchControlError("批定义 units 必须是非空数组")
    return data


def _validate_limits_field(raw: Any, label: str) -> dict[str, Any]:
    try:
        return BW._validate_limits(raw)
    except BW.BatchError as exc:
        raise BatchControlError(f"{label} 非法: {exc}") from exc


def _validate_positive(value: Any, label: str) -> float:
    try:
        return BW._positive_number(value, label)
    except BW.BatchError as exc:
        raise BatchControlError(f"{label} 非法: {exc}") from exc


def _resolve_environment(raw: Any, setup: Any) -> dict[str, str]:
    """顶部 environment：显式指纹；setup_sha256 缺省由 setup 计算。

    environment 本身必须是非空 dict（显式指纹契约）；若显式携带
    setup_sha256，必须等于 canonical JSON SHA256(setup)，否则拒绝
    （绝不以文件自报哈希静默覆盖计算值）。"""
    try:
        environment = P.json_guard(raw, "environment")
    except P.ProtocolError as exc:
        raise BatchControlError(f"批定义 environment 非法: {exc}") from exc
    if not isinstance(environment, dict) or not environment:
        raise BatchControlError("批定义 environment 必须是非空对象（显式指纹）")
    for key, value in environment.items():
        if not isinstance(key, str) or not key or not isinstance(value, str):
            raise BatchControlError("批定义 environment 必须是 dict[str, str]")
    digest = P.canonical_digest(setup)
    declared = environment.get("setup_sha256")
    resolved = dict(environment)
    if declared is None:
        resolved["setup_sha256"] = digest
    elif not isinstance(declared, str) or declared != digest:
        raise BatchControlError(
            f"批定义 environment['setup_sha256'] 与 canonical JSON "
            f"SHA256(setup)={digest} 不一致: {declared!r}")
    return resolved


def parse_batch_file(data: dict[str, Any], *, project_name: str,
                     repository: str, commit: str, setup: Any,
                     restic_namespace: str) -> dict[str, Any]:
    """把批定义规格化为执行上下文。所有校验纯本地，无副作用。

    - 每个 unit 走 `protocol.WorkUnit`（权威身份/`镜像`校验）；spec 可
      缺省 project/repository/commit/environment，由 CLI 上下文注入；
      显式给出的值必须与上下文一致（跨 project / commit 不符直接拒绝）。
    - unit_id 可选：给出时必须等于协议重算身份（与远端镜像规则一致）。
    - execution.timeout_seconds 必填且为正有限数（缺省/NaN 拒绝）。
    """
    if data["project"] != project_name:
        raise BatchControlError(
            f"批定义 project {data['project']!r} 与 CLI 项目 "
            f"{project_name!r} 不一致（跨 project 批次拒绝）")
    environment = _resolve_environment(data["environment"], setup)
    limits = None
    if data.get("manifest_limits") is not None:
        limits = _validate_limits_field(data["manifest_limits"],
                                        "批定义 manifest_limits")
    publish_timeout = None
    if data.get("publish_timeout_seconds") is not None:
        publish_timeout = _validate_positive(data["publish_timeout_seconds"],
                                             "批定义 publish_timeout_seconds")

    units: list[dict[str, Any]] = []
    for index, item in enumerate(data["units"]):
        label = f"units[{index}]"
        if not isinstance(item, dict):
            raise BatchControlError(f"批定义 {label} 必须是 JSON 对象")
        unknown = sorted(set(item) - UNIT_ITEM_FIELDS)
        if unknown:
            raise BatchControlError(f"批定义 {label} 含未知字段: {unknown}")
        missing = sorted({"spec", "execution"} - set(item))
        if missing:
            raise BatchControlError(f"批定义 {label} 缺少字段: {missing}")
        execution = item["execution"]
        if not isinstance(execution, dict):
            raise BatchControlError(f"批定义 {label}.execution 必须是对象")
        unknown = sorted(set(execution) - {"timeout_seconds"})
        if unknown:
            raise BatchControlError(f"批定义 {label}.execution 含未知字段: "
                                    f"{unknown}")
        if "timeout_seconds" not in execution:
            raise BatchControlError(f"批定义 {label}.execution 缺少 "
                                    "timeout_seconds（禁止无限等待）")
        timeout = _validate_positive(execution["timeout_seconds"],
                                     f"批定义 {label}.execution.timeout_seconds")

        payload = dict(item["spec"])
        if not isinstance(payload, dict):
            raise BatchControlError(f"批定义 {label}.spec 必须是对象")
        payload.setdefault("schema_version", P.SCHEMA_VERSION)  # 仅支持当前版本
        for field, injected in (("project", project_name),
                                ("repository", repository),
                                ("commit", commit),
                                ("environment", environment)):
            payload.setdefault(field, injected)
        try:
            unit = P.WorkUnit.from_dict(payload)
        except P.ProtocolError as exc:
            raise BatchControlError(f"批定义 {label}.spec 非法: {exc}") from exc
        if unit.project != project_name:
            raise BatchControlError(
                f"批定义 {label}.spec 属于 project {unit.project!r}，"
                f"与 CLI 项目 {project_name!r} 不一致（跨 project 拒绝）")
        if unit.repository != normalize_git_url(repository):
            raise BatchControlError(
                f"批定义 {label}.spec.repository {unit.repository!r} 与项目"
                f"仓库不一致")
        if unit.commit != commit:
            raise BatchControlError(
                f"批定义 {label}.spec.commit 与项目工作区当前 commit 不一致")
        if unit.environment != environment:
            raise BatchControlError(
                f"批定义 {label}.spec.environment 与批 environment 指纹不一致")
        if item.get("unit_id") is not None:
            declared = item["unit_id"]
            if not isinstance(declared, str) or declared != unit.identity:
                raise BatchControlError(
                    f"批定义 {label}.unit_id 与 spec 身份不一致: {declared!r}")
        units.append({"unit_id": unit.identity, "spec": unit,
                      "timeout_seconds": timeout, "index": index})

    first = units[0]["spec"]
    for unit in units[1:]:
        if unit["spec"].inputs != first.inputs:
            raise BatchControlError(
                f"批内 unit {unit['unit_id'][:12]} 的 inputs 与首个 unit "
                "不一致（批共享一次 restore，必须一致）")
        if unit["spec"].environment != first.environment:
            raise BatchControlError(
                f"批内 unit {unit['unit_id'][:12]} 的 environment 与首个 "
                "unit 不一致")
    # plan.inputs 的顺序 = 名称排序后的 snapshot 列（restore 顺序，远端强制）。
    inputs = [first.inputs[name] for name in sorted(first.inputs)]

    return {
        "project": project_name,
        "batch_id": derive_batch_id(project_name, repository, commit,
                                    environment, units, limits,
                                    publish_timeout, data),
        "environment": environment,
        "setup_sha256": environment["setup_sha256"],
        "units": units,
        "inputs": inputs,
        "manifest_limits": limits,
        "publish_timeout_seconds": publish_timeout,
        "repository": repository,
        "commit": commit,
        "setup": setup,
        "restic_namespace": restic_namespace,
    }


def derive_batch_id(project_name: str, repository: str, commit: str,
                    environment: dict[str, str],
                    units: list[dict[str, Any]], limits: dict[str, Any] | None,
                    publish_timeout: float | None,
                    raw: dict[str, Any]) -> str:
    """批次身份 = 「项目 + 批次定义」canonical JSON 的 SHA256 前缀。

    参与身份：schema/project/规范化 repository/pinned commit/environment
    （含 setup 指纹）/逐单元 spec+execution/manifest_limits/
    publish_timeout。attempt_id（依赖台账状态）与文件名不参与；同一批
    定义无论如何命名、何时重放都得到同一 batch_id。
    """
    payload: dict[str, Any] = {
        "schema_version": P.SCHEMA_VERSION,
        "project": project_name,
        "repository": normalize_git_url(repository),
        "commit": commit,
        "environment": environment,
        "units": [{"unit_id": unit["unit_id"],
                   "execution": {"timeout_seconds": unit["timeout_seconds"]},
                   "spec": unit["spec"].to_dict()} for unit in units],
    }
    if limits is not None:
        payload["manifest_limits"] = limits
    if publish_timeout is not None:
        payload["publish_timeout_seconds"] = publish_timeout
    digest = P.canonical_digest(payload)
    batch_id = _BATCH_ID_PREFIX + digest[:48]
    declared = raw.get("unit_batch_id")
    if declared is not None and declared != batch_id:
        raise BatchControlError(
            f"批定义显式 unit_batch_id {declared!r} 与定义派生身份 "
            f"{batch_id} 不一致；批次身份必须锚定定义内容（不能只靠"
            "文件名/任意标签）")
    try:
        BW._ident_name(batch_id, "batch_id")
    except BW.BatchError as exc:  # pragma: no cover - 派生值必然合法
        raise BatchControlError(f"派生 batch_id 非法: {exc}") from exc
    return batch_id


# ---------------------------------------------------------------------------
# GPU 分配前的远端计划合成与纯校验（batch_worker.validate_batch 镜像链）
# ---------------------------------------------------------------------------

def build_unit_batch_content(parsed: dict[str, Any],
                             attempt_by_uid: dict[str, str]) -> dict[str, Any]:
    """plan['unit_batch'] 内容：字段名与 batch_worker.validate_batch 一致。

    只包含 attempt_by_uid 中的 unit（即本次真正要跑到远端单元）——不能
    因批中部分完成就把已提交单元重算进去。占位调用（preflight）用
    attempt_by_uid 覆盖全部 parsed units（或空 dict → 全部占位）。"""
    chosen = [u for u in parsed["units"]
              if u["unit_id"] in attempt_by_uid] or (
        parsed["units"] if not attempt_by_uid else [])
    if not chosen:
        raise BatchControlError("无待执行单元")
    content: dict[str, Any] = {
        "batch_id": parsed["batch_id"],
        "schema_version": P.SCHEMA_VERSION,
        "units": [{
            "unit_id": unit["unit_id"],
            "attempt_id": attempt_by_uid[unit["unit_id"]],
            "spec": unit["spec"].to_dict(),
            "execution": {"timeout_seconds": unit["timeout_seconds"]},
        } for unit in chosen],
    }
    if parsed["manifest_limits"] is not None:
        content["manifest_limits"] = parsed["manifest_limits"]
    if parsed["publish_timeout_seconds"] is not None:
        content["publish_timeout_seconds"] = \
            parsed["publish_timeout_seconds"]
    return content


def preflight_validate(parsed: dict[str, Any],
                       attempt_by_uid: dict[str, str]) -> None:
    """远端计划纯校验（validate_batch 镜像），必须在任何 plumbing 之前
    被调用：会话/attempt 用占位身份也足以暴露全部绑定/schema/身份问题
    （fail closed 的第一次防线；后续用真实 attempt_id 再验一次）。"""
    content = build_unit_batch_content(
        parsed, attempt_by_uid or {u["unit_id"]: f"att-preflight-{i}"
                                   for i, u in enumerate(parsed["units"])})
    validate_unit_batch_content(parsed, content)


def validate_unit_batch_content(parsed: dict[str, Any],
                                content: dict[str, Any]) -> None:
    plan = {"session": "batch-preflight", "job_id": "batch-preflight",
            "project": parsed["project"], "repository": parsed["repository"],
            "commit": parsed["commit"],
            "restic_repository": parsed["restic_namespace"],
            "inputs": parsed["inputs"], "setup": parsed["setup"],
            "unit_batch": content}
    try:
        BW.validate_batch(plan)
    except BW.BatchError as exc:
        raise BatchControlError(f"批次远端计划预校验失败（GPU 分配前拒绝）: "
                                f"{exc}") from exc


# ---------------------------------------------------------------------------
# 台账登记与启动前恢复对账
# ---------------------------------------------------------------------------

def next_attempt_seq(view: dict[str, Any]) -> int:
    """unit 在台账中的下一次 attempt 序号（§RECOVERY：显式重开而非自动重试）。"""
    return 1 + max((a["seq"] for a in view["attempts"].values()), default=0)


def attempt_id_for(unit_id: str, seq: int) -> str:
    value = f"att-{unit_id[:12]}-{seq}"
    BW._ident_name(value, "attempt_id")  # 形态守卫（绝不静默放行）
    return value


def plan_units(project_name: str,
               units: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """对每个 unit 幂等 `Ledger.plan`；返回 unit_id → 台账视图。"""
    ledger = Ledger(project_name)
    views: dict[str, dict[str, Any]] = {}
    for unit in units:
        uid = ledger.plan(unit["spec"])
        views[uid] = ledger.get(uid)
    return views


def _state_evidence_identity(session: str, status: dict[str, Any],
                             uid: str, attempt_id: str,
                             batch_id: str) -> dict[str, Any] | None:
    """从远端状态中按身份取逐单元 outcome；身份不符返回 None（绝不裁决）。

    session / batch_id / attempt_id 三者都必须与本地台账 attempt 一致：
    缺 batch_id 比对会让另一批次的同元组 (session, uid, attempt_id)
    状态被错误归因到本批 attempt（跨批状态不得归因）。"""
    batch = status.get("batch")
    if not isinstance(batch, dict) or not isinstance(batch.get("outcomes"), dict):
        return None
    if batch.get("session") != session or batch.get("batch_id") != batch_id:
        return None
    outcome = batch["outcomes"].get(uid)
    if not isinstance(outcome, dict) or outcome.get("attempt_id") != attempt_id:
        return None
    return outcome


def _read_receipt(session: str, batch_id: str, uid: str) -> dict[str, Any]:
    from .colab_runner import fetch_remote_json  # 延迟 import（避免环）
    return fetch_remote_json(
        session, f"/content/mlbridge_receipts/{batch_id}/{uid}.json")


def _validate_receipt(receipt: Any, *, batch_id: str, uid: str,
                      attempt_id: str, session: str, namespace: str,
                      validator_kind: str, validator_spec_sha256: str) -> None:
    """回执只做身份/摘要契约校验（结构见 docs/UNIT_EXECUTION.md）。

    通过都**不**代表本地独立验收了快照内容；任何缺席/不匹配都抛错，
    调用方按 UNRESOLVED 报告，绝不改台账。"""
    fields = {"schema_version", "batch_id", "unit_id", "attempt_id",
              "session", "namespace", "snapshot_id", "manifest_sha256",
              "manifest_path", "validator_spec_sha256", "validator_result",
              "phase", "finished"}
    if not isinstance(receipt, dict) or set(receipt) != fields:
        missing = sorted(fields - set(receipt)) if isinstance(receipt, dict) \
            else []
        raise BatchControlError(
            f"回执缺字段或含未知字段（拒绝匿名回执）: missing={missing}, "
            f"fields={sorted(receipt) if isinstance(receipt, dict) else receipt!r}")

    def _hex64(value: Any, label: str) -> None:
        if not isinstance(value, str) or len(value) != 64 \
                or any(c not in "0123456789abcdef" for c in value):
            raise BatchControlError(f"回执 {label} 不是合法 64 位小写十六进制: "
                                    f"{value!r}")
    for label in ("snapshot_id", "manifest_sha256"):
        _hex64(receipt[label], label)
    _hex64(receipt["validator_spec_sha256"], "validator_spec_sha256")
    if receipt["schema_version"] != BW.RECEIPT_SCHEMA_VERSION \
            or receipt["batch_id"] != batch_id or receipt["unit_id"] != uid \
            or receipt["attempt_id"] != attempt_id \
            or receipt["session"] != session:
        raise BatchControlError("回执身份与台账 attempt/session 不符，拒绝")
    if receipt["namespace"] != namespace:
        raise BatchControlError(f"回执 namespace 与批 restic 仓库不符: "
                                f"{receipt['namespace']!r}")
    if receipt["validator_spec_sha256"] != validator_spec_sha256:
        raise BatchControlError("回执 validator_spec_sha256 与 spec.validator "
                                "摘要不符")
    validator = receipt["validator_result"]
    if not isinstance(validator, dict) \
            or validator.get("kind") != validator_kind \
            or validator.get("status") != "passed":
        raise BatchControlError(f"回执 validator_result 与 spec 不符: "
                                f"{validator!r}")
    if receipt["phase"] != "published" or not isinstance(receipt["finished"], str):
        raise BatchControlError("回执 phase/finished 字段异常")


def reconcile_attempt(
        ledger: Ledger, uid: str, spec: P.WorkUnit, attempt_id: str,
        session: str, *, batch_id: str, restic_namespace: str,
        status: dict[str, Any] | None = None,
        read_receipt: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """尝试用远端状态/回执解决台账中一个 attempt；**只**在拿到确定终态
    证据时才 reconcile，成功还需回执身份/摘要匹配才 commit_result。

    返回: {'disposition': ..., 'detail': str, ...}
    - DISP_RECONCILED_SUCCESS: 已 reconcile(task_succeeded) 并 commit_result
    - DISP_UNCOMMITTED_SUCCESS: 成功证据/终态已解决但成果未提交（回执
      不可用）——需要回执对账，绝不勉强
    - DISP_RECONCILED_FAILED: 已按确定失败证据 reconcile(task_failed)
    - DISP_ABANDONED: 先前已被 execution_abandoned 证据解决（诚实放弃，
      不改写；重跑该 unit 必须显式 --retry-failed）
    - DISP_UNRESOLVED: PUBLISH_UNKNOWN/RECEIPT_FAILED/NOT_STARTED/
      观测失联/无证据/回执缺失或身份不符——不改台账
    - DISP_COMPLETED: 台账已提交成果
    - DISP_UNTRACKED: 台账内没有该 attempt 记录（只报告，不改台账）
    """
    view = ledger.get(uid)
    if view["state"] == "completed":
        return {"disposition": DISP_COMPLETED, "detail": "台账已有已提交成果"}
    attempt = view["attempts"].get(attempt_id)
    if attempt is None:
        return {"disposition": DISP_UNTRACKED,
                "detail": f"台账中不存在该 attempt: {attempt_id}"}
    if attempt["session"] != session:
        return {"disposition": DISP_UNRESOLVED,
                "detail": f"attempt {attempt_id} 绑定 session "
                          f"{attempt['session']}，与对账 session {session} 不符"}
    resolution = attempt["resolution"]
    validator_kind = spec.validator["kind"]
    validator_spec_sha256 = P.canonical_digest(spec.validator)

    if resolution is not None:
        if resolution["outcome"] == "task_failed":
            return {"disposition": DISP_RECONCILED_FAILED,
                    "detail": f"先前证据已解决: {resolution}"}
        if resolution["outcome"] == "execution_abandoned":
            return {"disposition": DISP_ABANDONED,
                    "detail": f"先前执行已被诚实放弃（不改写;需显式重试）: "
                              f"{resolution}"}
        # task_succeeded（未提交成果）：只能靠回执补 commit_result。
        return _commit_from_receipt(
            ledger, uid, spec, attempt_id, session, batch_id,
            restic_namespace, validator_kind, validator_spec_sha256,
            reason="该 attempt 已有 task_succeeded 终态证据但成果未提交",
            read_receipt=read_receipt)

    outcome = _state_evidence_identity(session, status, uid, attempt_id,
                                       batch_id) \
        if status is not None else None
    if outcome is None:
        return {"disposition": DISP_UNRESOLVED,
                "detail": "远端状态缺少该 unit/attempt 的批逐单元 outcome"
                          "（UNREACHABLE/无证据，或 batch.session/batch_id"
                          " 与台账/当前批定义不一致），不裁决不新建 attempt"}
    what = outcome.get("outcome")
    if what == "SUCCEEDED":
        return _commit_from_receipt(
            ledger, uid, spec, attempt_id, session, batch_id,
            restic_namespace, validator_kind, validator_spec_sha256,
            reason="远端状态逐单元 outcome=SUCCEEDED", read_receipt=read_receipt)
    if what == "TASK_FAILED":
        stage = outcome.get("stage")
        detail = outcome.get("detail") or ""
        ledger.reconcile(uid, attempt_id, session, {
            "source": f"colab-state/{session}",
            "terminal_state": "FAILED",
            "note": f"worker 自报远端逐单元 outcome=TASK_FAILED"
                    f"（stage={stage}），自报而非独立验收: "
                    f"{detail[:160]}",
        })
        return {"disposition": DISP_RECONCILED_FAILED,
                "detail": f"stage={stage}: {detail}"}
    if what in ("PUBLISH_UNKNOWN", "RECEIPT_FAILED", "NOT_STARTED"):
        extra = outcome.get("snapshot_id")
        return {
            "disposition": DISP_UNRESOLVED,
            "detail": (f"远端 outcome={what}"
                       + (f"（snapshot_id={extra}，请用 mlbridge-batch-*/"
                          f"mlbridge-unit-*/mlbridge-attempt-* tags 人工对账）"
                          if extra else "")
                       + "：不确定事实，不转失败也不当成功"),
        }
    return {"disposition": DISP_UNRESOLVED,
            "detail": f"远端 outcome={what!r} 无法识别，不裁决"}


def _commit_from_receipt(
        ledger: Ledger, uid: str, spec: P.WorkUnit, attempt_id: str,
        session: str, batch_id: str, restic_namespace: str,
        validator_kind: str, validator_spec_sha256: str, *, reason: str,
        read_receipt: Callable[..., dict[str, Any]] | None) -> dict[str, Any]:
    reader = read_receipt or _read_receipt
    try:
        receipt = reader(session, batch_id, uid)
        _validate_receipt(receipt, batch_id=batch_id, uid=uid,
                          attempt_id=attempt_id, session=session,
                          namespace=restic_namespace,
                          validator_kind=validator_kind,
                          validator_spec_sha256=validator_spec_sha256)
    except (BridgeError, OSError) as exc:
        return {"disposition": DISP_UNRESOLVED,
                "detail": f"{reason}，但回执不可用/不契约（拒绝匿名回执）: {exc}"}
    ledger.reconcile(uid, attempt_id, session, {
        "source": f"colab-receipt/{session}",
        "terminal_state": "SUCCEEDED",
        "note": f"worker 自报回执 published（snapshot={receipt['snapshot_id']}"
                f"）；自报而非本地独立产物验收",
    })
    committed = ledger.commit_result(
        uid, attempt_id, session, snapshot=receipt["snapshot_id"],
        namespace=restic_namespace, manifest_sha256=receipt["manifest_sha256"],
        validator_spec_sha256=validator_spec_sha256,
        validator_result=receipt["validator_result"])
    return {"disposition": DISP_RECONCILED_SUCCESS,
            "detail": f"reconcile+commit_result({committed})；"
                      "仅记录 worker 自报，未经本地独立验收"}


# ---------------------------------------------------------------------------
# 执行中断后的诚实放弃（execution_abandoned）
# ---------------------------------------------------------------------------

def _validate_execution_ended(proof: Any) -> dict[str, str]:
    """校验执行终局证据形态：{'source': str, 'detail': str}，source 需能
    进台账 source 字段（≤128）；非观测猜测（观测失联绝不进入该通道）。"""
    if not isinstance(proof, dict) or set(proof) - {"source", "detail"} \
            or not isinstance(proof.get("source"), str) or not proof["source"] \
            or len(proof["source"]) > 128 \
            or not isinstance(proof.get("detail"), str):
        raise BatchControlError(
            "执行终局证据必须是 {'source','detail'} dict（controller 正面"
            "事实：stop 确认返回或 foreground 哨兵），不是观测猜测")
    return proof


def restic_attempt_snapshots(namespace: str, batch_id: str, uid: str,
                             attempt_id: str) -> list[str]:
    """本地只读 restic tags 查询：该 attempt 是否有任何已发布快照。

    attempt_id 由台账序号确定性生成、全台唯一，attempts 不复用旧
    attempt 的发布，因此 attempt 级 tag 单独足以定位（未来若需要更严
    格的组合查询，签名已预留 batch/uid 参数）。任何 CLI 失败/解析错误
    都抛错，**绝不**返回空列表冒充“缺席已核验”；核验失败时不得放弃。"""
    from .colab_runner import restic_snapshots_with_tags  # 延迟导入避免环
    return [str(s.get("id", ""))
            for s in restic_snapshots_with_tags(
                namespace, f"mlbridge-attempt-{attempt_id}")]


def _batch_identity_status(session: str, batch_id: str, status: Any,
                           record: Any) -> dict[str, Any] | None:
    """返回携带本批身份证据（batch.session+batch_id 匹配）的状态视图。

    优先实时 status；UNREACHABLE/无身份字段时回退本地任务记录的
    last_known（消失前最后确认的远端状态）。都没有 → None：无逐单元
    证据，绝不放弃/裁决。"""
    candidates = [status]
    if isinstance(record, dict) and isinstance(record.get("last_known"), dict):
        candidates.append(record["last_known"])
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        batch = candidate.get("batch")
        if isinstance(batch, dict) and \
                batch.get("session") == session and \
                batch.get("batch_id") == batch_id:
            return candidate
    return None


def _execution_ended_candidate(session: str, evidence_status: Any,
                               uid: str, attempt_id: str,
                               batch_id: str) -> str | None:
    """该 attempt 最后已知控制面证据是否表明它从未进入发布不确定区。

    返回 'RUNNING'/'NOT_STARTED'（放弃候选，未进入或未开始发布）或
    None（有 PUBLISH_UNKNOWN/RECEIPT_FAILED/成功/失败证据、身份不符、
    或根本没有本批身份证据 → 绝不放弃）。发布不确定事实独立成类：
    PUBLISH_UNKNOWN/RECEIPT_FAILED 的 attempt 永远不得转
    execution_abandoned 或 task_failed。"""
    if not isinstance(evidence_status, dict):
        return None
    batch = evidence_status.get("batch")
    if not isinstance(batch, dict) or not isinstance(batch.get("outcomes"), dict):
        return None
    if batch.get("session") != session or batch.get("batch_id") != batch_id:
        return None
    entry = batch["outcomes"].get(uid)
    if entry is None:
        return "NOT_STARTED"
    if not isinstance(entry, dict) or entry.get("attempt_id") != attempt_id:
        return None  # 身份不符：绝不归因到本 attempt
    label = entry.get("outcome")
    if label in ("RUNNING", "NOT_STARTED"):
        return label
    return None


def _try_abandon_interrupted(
        ledger: Ledger, uid: str, attempt_id: str, session: str, *,
        batch_id: str, restic_namespace: str, evidence_status: Any,
        proof: dict[str, str],
        list_attempt_snapshots: Callable[..., list[str]] | None) -> dict[str, Any]:
    """执行终局证据成立时对未决 attempt 的诚实放弃（execution_abandoned）。

    依据链必须全部成立，缺任何一环都返回 DISP_UNRESOLVED 且台账零改写：
    ① proof 是正面执行终局证据（由调用方预校验，观测失联不通过）；
    ② 最后已知控制面证据（本批身份匹配）标明该 attempt 处于
    RUNNING/NOT_STARTED（未进入发布不确定区）；
    ③ restic tags 核验确认该 attempt 无任何快照（因而无 receipt）。
    有 receipt/snapshot 的 attempt、PUBLISH_UNKNOWN/RECEIPT_FAILED 的
    attempt 一律拒绝本函数，不得转 ABANDONED 或 task_failed。"""
    label = _execution_ended_candidate(session, evidence_status, uid,
                                       attempt_id, batch_id)
    if label is None:
        return {"disposition": DISP_UNRESOLVED,
                "detail": "无该 attempt 未进入发布/未发布成功的正面控制面"
                          "证据（或存在 PUBLISH_UNKNOWN/RECEIPT_FAILED 等"
                          "发布不确定事实），不放弃"}
    reader = list_attempt_snapshots or restic_attempt_snapshots
    try:
        snapshots = reader(restic_namespace, batch_id, uid, attempt_id)
    except (LedgerError, BridgeError, OSError) as exc:
        return {"disposition": DISP_UNRESOLVED,
                "detail": f"无法核验该 attempt 无快照（绝不放弃）: {exc}"}
    if not isinstance(snapshots, list):
        return {"disposition": DISP_UNRESOLVED,
                "detail": "快照核验结果非法（拒绝猜测），不放弃"}
    if snapshots:
        shown = ",".join(str(s)[:16] for s in snapshots[:5])
        return {"disposition": DISP_UNRESOLVED,
                "detail": f"该 attempt 存在快照证据（{shown}…），不放弃；"
                          "有 receipt/snapshot 的 attempt 禁止 execution_"
                          "abandoned；请人工对账 tags/回执后处理成果"}
    note = (
        f"执行中断后的诚实放弃（依据链）：①执行终局证据 "
        f"{proof['source']}: {proof['detail']}；②最后已知控制面证据（本批 "
        f"batch.session/batch_id 身份匹配）：该 attempt 为 {label}，"
        "从未进入发布不确定区（无 PUBLISH_UNKNOWN/RECEIPT_FAILED）；"
        f"③已核对 restic tags（mlbridge-attempt-{attempt_id}）：无任何"
        "快照，因此不可能有已确认发布的 receipt/snapshot。观测失联/"
        "传输失败从不构成放弃依据。有 receipt/snapshot 的 attempt 禁止"
        " execution_abandoned；重跑该 unit 必须显式 --retry-failed。")
    ledger.reconcile(uid, attempt_id, session, {
        "source": proof["source"],
        "terminal_state": "ABANDONED",
        "note": note,
    })
    return {"disposition": DISP_ABANDONED,
            "detail": f"execution_abandoned（终局来源={proof['source']}，"
                      f"最后已知={label}，tags 核验无快照）；显式 "
                      "--retry-failed 才能重跑"}


# ---------------------------------------------------------------------------
# 启动前恢复对账（批 run 的第一节）
# ---------------------------------------------------------------------------

def _session_record(session: str) -> dict[str, Any] | None:
    """读取 session 的本地任务记录；不存在/通道失败 → None（尽力而为）。"""
    from .colab_runner import get_local  # 延迟 import（避免环）
    try:
        rec = get_local(session)
    except BridgeError:
        return None
    return rec if isinstance(rec, dict) else None


def _session_ended_proof(session: str) -> dict[str, str] | None:
    """从本地任务记录提取正面执行终局证据（stop 确认 / foreground 哨兵）；
    无则 None。观测失联/传输失败绝不作为执行终局证据。"""
    from .colab_runner import _execution_ended_proof  # 延迟 import（避免环）
    try:
        record = _session_record(session)
    except Exception:  # pragma: no cover - _session_record 已吞错
        return None
    proof = _execution_ended_proof(record)
    return _validate_execution_ended(proof) if proof is not None else None




def recover_batch(project_name: str, parsed: dict[str, Any], *,
                  retry_failed: bool) -> dict[str, Any]:
    """登记并对账全部 unit；返回可执行 subset 与逐单元处置。

    - 已 completed → DISP_COMPLETED（跳过，不重跑）。
    - 未解决 attempt → 用该 attempt 的 session 做状态/回执对账：
      成功→reconcile+commit；失败→reconcile(task_failed)，之后**必须
      显式 --retry-failed** 才允许新 attempt，否则 BatchControlError 停止；
      无证据/不确定→BatchControlError 停止（UNRESOLVED 报告 + session）。
    - 例外（执行终局证据）：该 attempt session 的本地任务记录存在正面
      执行终局证据（实例确认停止 / foreground 哨兵）时，对“最后已知
      证据为 RUNNING/NOT_STARTED（未进入发布不确定区）且 restic 核验
      确认无快照”的未决 attempt 写 execution_abandoned（诚实放弃），
      随后同样受 --retry-failed 显式门控；其它情况不动台账。
    - ledger 无 RESOURCE_LOST 语义：观测失联/无证据永远不写成失败。
    - 崩溃窗口自愈（F3）：台账已有 task_succeeded 终态但成果未提交
      （上次运行可能在 reconcile 后、commit_result 前本地中断）→ 先按
      同一 session/batch_id 做回执补提交，全部身份/摘要契约校验照旧
      绝不放宽；补提交成功则跳过该 unit，失败才按上述策略停止。
    """
    ledger = Ledger(project_name)
    views = plan_units(project_name, parsed["units"])
    queued: list[dict[str, Any]] = []
    dispositions: dict[str, dict[str, Any]] = {}
    statuses: dict[str, dict[str, Any]] = {}
    for unit in parsed["units"]:
        uid = unit["unit_id"]
        view = views[uid]
        if view["state"] == "completed":
            dispositions[uid] = {"disposition": DISP_SKIPPED,
                                 "detail": "台账已提交成果，跳过"}
            continue
        latest = view["latest_attempt"]
        if latest is not None and not view["latest_resolved"]:
            session = view["attempts"][latest]["session"]
            # 每个被 probe 的 session 只下载一次远端状态（get_status
            # 失败也返回 UNREACHABLE dict，由身份检查归为未决）。
            if session not in statuses:
                from .colab_runner import get_status  # 延迟 import（避免环）
                statuses[session] = get_status(session)
            probe = reconcile_attempt(ledger, uid, unit["spec"], latest,
                                      session, batch_id=parsed["batch_id"],
                                      restic_namespace=parsed["restic_namespace"],
                                      status=statuses[session])
            ended = None
            if probe["disposition"] == DISP_UNRESOLVED:
                # 执行终局证据存在时的诚实放弃窗口；不存在或不满足
                # 全部环片时保持未决（零改写）。
                ended = _session_ended_proof(session)
            if ended is not None:
                try:
                    record = _session_record(session)
                except (LedgerError, BridgeError):
                    record = None
                evidence = _batch_identity_status(
                    session, parsed["batch_id"], statuses[session], record)
                proof = _validate_execution_ended(ended)
                probe = _try_abandon_interrupted(
                    ledger, uid, latest, session,
                    batch_id=parsed["batch_id"],
                    restic_namespace=parsed["restic_namespace"],
                    evidence_status=evidence, proof=proof,
                    list_attempt_snapshots=None)
            if probe["disposition"] == DISP_UNRESOLVED:
                raise BatchControlError(
                    f"unit {uid[:12]} 存在未决 attempt {latest}（session "
                    f"{session}），证据不可判定: {probe['detail']}\n"
                    f"  → 不新建 attempt、不分配实例；请先 "
                    f"`mlbridge status {session}` / `mlbridge watch {session}` "
                    f"确认远端事实；若实例确已不存在，该缺口（ledger 无 "
                    f"RESOURCE_LOST）需人工检查台账，本工具不提供 force reset。")
            dispositions[uid] = probe
            if probe["disposition"] == DISP_UNCOMMITTED_SUCCESS:
                raise BatchControlError(
                    f"unit {uid[:12]} 已有成功终态证据但成果未提交且回执"
                    f"不可用（session {session}）: {probe['detail']}\n"
                    f"  → 请人工对账 tags（mlbridge-unit-{uid}）后处理；"
                    "本工具不提供 force 通道。")
        view = ledger.get(uid)  # 对账后重读：可能已被 reconcile/commit 变更
        if view["state"] == "completed":
            dispositions.setdefault(uid, {"disposition": DISP_SKIPPED,
                                          "detail": "台账已提交成果，跳过"})
            continue
        if view["latest_resolved"]:
            resolved = view["attempts"][view["latest_attempt"]]["resolution"]
            if resolved["outcome"] == "task_succeeded":
                # 崩溃窗口自愈：task_succeeded 已解决但成果未提交（上次
                # 运行可能在 reconcile 之后、commit_result 之前本地中断）。
                # 先按同一 session/batch_id 做回执补提交；全部身份/摘要
                # 契约校验照旧，绝不放宽（与 settle_terminal 同一条路径）。
                attempt_id = view["latest_attempt"]
                attempt = view["attempts"][attempt_id]
                spec_from_view = P.WorkUnit.from_dict(view["spec"])
                try:
                    heal = _commit_from_receipt(
                        ledger, uid, spec_from_view, attempt_id,
                        attempt["session"], batch_id=parsed["batch_id"],
                        restic_namespace=parsed["restic_namespace"],
                        validator_kind=spec_from_view.validator["kind"],
                        validator_spec_sha256=P.canonical_digest(
                            spec_from_view.validator),
                        reason="该 attempt 已有 task_succeeded 终态但成果未提交"
                               "（上次运行可能在 reconcile 后 commit 前中断）",
                        read_receipt=None)
                except (LedgerError, BridgeError, OSError) as exc:
                    raise BatchControlError(
                        f"unit {uid[:12]} 已有成功终态证据但成果未提交"
                        f"（attempt {attempt_id}），崩溃窗口补提交失败: "
                        f"{exc}\n"
                        f"  → 请人工对账 tags（mlbridge-unit-{uid}）后处理；"
                        "本工具不提供 force 通道。") from exc
                view = ledger.get(uid)  # 补提交后重读
                if view["state"] == "completed":
                    dispositions[uid] = {
                        "disposition": DISP_HEALED_COMMIT,
                        "detail": f"崩溃窗口补提交: {heal['detail']}"}
                    continue
                raise BatchControlError(
                    f"unit {uid[:12]} 已有成功终态证据但成果未提交"
                    f"（attempt {attempt_id}，崩溃窗口补提交失败）: "
                    f"{heal['detail']}\n"
                    f"  → 请人工对账 tags（mlbridge-unit-{uid}）后处理；"
                    "本工具不提供 force 通道。")
            # 无论失败/放弃是本次刚对账得出还是此前遗留，都同样受
            # 默认的不自动重试策略约束：必须显式 --retry-failed 才开
            # 新 attempt（task_failed 与 execution_abandoned 同门槛；
            # 台账层无 force 通道）。
            if not retry_failed:
                raise BatchControlError(
                    f"unit {uid[:12]} 已确定终止（最新 attempt {latest} 的"
                    f"resolution={resolved.get('outcome')}，"
                    "task_failed 或 execution_abandoned）。默认策略不自动"
                    "重试；确认后用 --retry-failed 显式开启新 attempt。")
            # 重跑入队的 unit 用独立 disposition（不再显示先前结论：
            # 失败/放弃证据是上一 attempt 的结论，本 unit 本次已新鲜
            # 入队）。
            if dispositions.get(uid, {}).get("disposition") in (
                    DISP_RECONCILED_FAILED, DISP_ABANDONED):
                dispositions[uid] = {"disposition": DISP_QUEUED_RETRY,
                                     "detail": f"已按失败证据开新 attempt: "
                                               f"{resolved}"}
            else:
                dispositions.setdefault(uid, {"disposition": DISP_QUEUED_RETRY,
                                              "detail": f"先前证据: {resolved}"})
        queued.append(unit)
    return {"views": views, "dispositions": dispositions, "queued": queued}


# ---------------------------------------------------------------------------
# run --unit-batch 编排（GPU 分配前全部完成）
# ---------------------------------------------------------------------------

def start_batch_attempts(plan: dict[str, Any], attempt_by_uid: dict[str, str],
                         session: str) -> list[str]:
    """launch 的对齐窗口：`colab new` 成功后、任何 staging 前调用。

    为每个待执行 unit 显式 start_attempt（attempt_id/身份已在计划中向
    远端公开）；写入即与远端 plan 对齐，之后达到 staging 的计划必然对应
    同一批 attempt。已完成的 unit 不在此映射内、不会被登记。这里的失败
    （并发 attempt_id 冲突、台账 IO 写入失败等）属于**对齐失败**：不
    进入 staging，由 launch 尽力停止刚分配的实例；仅当 stop 确认成功
    后写 controller 本地事实证据并要求后续 --retry-failed（见
    record_alignment_abort 与 docs/RECOVERY.md）。
    """
    ledger = Ledger(plan["project"])
    started: list[str] = []
    for uid in sorted(attempt_by_uid):
        ledger.start_attempt(uid, attempt_by_uid[uid], session)
        started.append(attempt_by_uid[uid])
    return started


def record_alignment_abort(project_name: str, attempt_by_uid: dict[str, str],
                           session: str) -> dict[str, Any]:
    """对齐窗口中止后的控制器本地事实证据（**只在**实例已确认停止后调用）。

    为该窗口内已写入且仍未决的 attempt 写
    `reconcile(task_failed, source='controller/alignment-abort')`：note
    明确 staging 未开始、刚分配的实例已停止。这是控制器本地事实：
    远端从未执行本批任务，因此绝不声称远端任务失败、绝不伪造远端终态。
    - 未写入该 session 名下的 attempt（start_attempt 未到达）一律不碰；
    - 已有终态证据的 attempt（绝不改写）一律不碰；
    - 调用方若未能确认实例已停止（instance_stopped=false），**绝不能**
      调用本函数：实例状态未知时不得写任何证据。

    证据解开后该 attempt 即确定失败（task_failed）：重跑该 unit 必须
    显式 `--retry-failed`（新 attempt、新 seq），默认策略不自动重试。
    """
    evidence = {
        "source": "controller/alignment-abort",
        "terminal_state": "FAILED",
        "note": "对齐窗口中止（controller 本地事实）：staging 未开始"
                "（无 worker、无凭据上传），远端从未执行本批任务，因此"
                "绝不声称远端任务失败、绝不伪造远端终态；刚分配的实例"
                "已停止（stop 已确认）。开新 attempt 必须显式 --retry-failed。",
    }
    ledger = Ledger(project_name)
    recorded: list[str] = []
    for uid, attempt_id in sorted(attempt_by_uid.items()):
        try:
            view = ledger.get(uid)
        except (LedgerError, BridgeError, OSError) as exc:
            print(f'[mlbridge] 对齐中止证据：读取 unit {uid[:12]} 失败，'
                  f'跳过: {exc}', file=sys.stderr, flush=True)
            continue
        attempt = view["attempts"].get(attempt_id)
        if attempt is None or attempt["session"] != session:
            continue  # 窗口内未写入（start_attempt 未到达该 unit）
        if attempt["resolution"] is not None:
            continue  # 已被终态证据解决，绝不改写
        try:
            ledger.reconcile(uid, attempt_id, session, evidence)
        except (LedgerError, BridgeError, OSError) as exc:
            print(f'[mlbridge] 对齐中止证据：写入 unit {uid[:12]} 失败，'
                  f'跳过: {exc}', file=sys.stderr, flush=True)
            continue
        recorded.append(attempt_id)
    return {"session": session, "attempted": sorted(attempt_by_uid.values()),
            "attempts_recorded": recorded}


def run_batch(*, project_name: str, batch_file: Path, commit: str,
              repository: str, setup: Any, restic_namespace: str,
              retry_failed: bool, on_launch: Callable[[dict[str, Any], dict[str, str]], Any],
              execution_mode: str = 'detached') -> dict[str, Any]:
    """批模式编排；on_launch(parsed, attempt_by_uid) 由 CLI 注入真正
    launch/watch 流程，签名约定见 cli 批分支。

    execution_mode 只改变交付/生命周期（detached 分离启动 = 缺省旧行为；
    foreground 仅批模式、cell 内同步运行），**不参与 batch_id 身份**，
    不改变台账语义：跨模式重入同一批次的恢复对账一致。

    全部校验与对账都在 on_launch（含任何 GPU 分配）之前完成。返回
    {'session', 'batch_id', 'report', 'result'}。
    """
    if execution_mode not in ('detached', 'foreground'):
        raise BatchControlError(
            f"--execution-mode 只支持 detached/foreground: {execution_mode!r}")
    parsed = parse_batch_file(load_batch_file(batch_file),
                              project_name=project_name, repository=repository,
                              commit=commit, setup=setup,
                              restic_namespace=restic_namespace)
    # 第一道：与台账状态无关的纯校验（尝试_id 用占位身份，暴露一切
    # 绑定/身份/schema 问题）。
    preflight_validate(parsed, {})
    recovered = recover_batch(project_name, parsed, retry_failed=retry_failed)
    if not recovered["queued"]:
        report = {
            "batch_id": parsed["batch_id"],
            "allocated": False,
            "execution_mode": execution_mode,
            "units": {uid: disp for uid, disp in
                      recovered["dispositions"].items()},
        }
        return {"session": None, "batch_id": parsed["batch_id"],
                "report": report, "result": None}
    attempt_by_uid: dict[str, str] = {}
    ledger = Ledger(project_name)
    for unit in recovered["queued"]:
        uid = unit["unit_id"]
        view = ledger.get(uid)
        attempt_by_uid[uid] = attempt_id_for(uid, next_attempt_seq(view))
    content = build_unit_batch_content(parsed, attempt_by_uid)
    # 第二道：真实 attempt_id 的完整计划纯校验（仍在 GPU 分配前）。
    validate_unit_batch_content(parsed, content)
    plan = {"project": project_name, "repository": repository,
            "commit": commit, "setup": setup, "inputs": parsed["inputs"],
            "unit_batch": content, "restic_namespace": parsed["restic_namespace"],
            "batch_id": parsed["batch_id"]}
    if execution_mode == "foreground":
        # detached（缺省）不写入该键，与旧计划/载荷保持一致。
        plan["execution_mode"] = "foreground"
    session, result = on_launch(plan, attempt_by_uid)
    report = {
        "batch_id": parsed["batch_id"],
        "allocated": True,
        "execution_mode": execution_mode,
        "units": {unit["unit_id"]: recovered["dispositions"].get(
            unit["unit_id"], {"disposition": DISP_QUEUED})
            for unit in parsed["units"]},
    }
    return {"session": session, "batch_id": parsed["batch_id"],
            "plan": plan, "report": report, "result": result}


# ---------------------------------------------------------------------------
# watch / 批终态对账（由 colab_runner.watch 在终态时调用）
# ---------------------------------------------------------------------------

def settle_terminal(session: str, plan: dict[str, Any],
                    status: dict[str, Any], *, execution_ended: dict[str, str] | None = None,
                    list_attempt_snapshots: Callable[..., list[str]] | None = None
                    ) -> dict[str, Any]:
    """批终态逐单元对账：读状态证据与回执，把每个 unit 映射到台账处置。

    - SUCCEEDED + 身份/摘要匹配回执 → reconcile(task_succeeded) +
      commit_result（worker 自报，不宣称独立验收）
    - TASK_FAILED（真实 command/validator 失败）→ reconcile(task_failed)
      （不自动重试）
    - PUBLISH_UNKNOWN / RECEIPT_FAILED / NOT_STARTED / 无证据 → 未决，
      不转失败也不当成功，保留实例（不 stop）
    - 批整体 phase 只是控制面事实；绝不据此批量写终态。

    execution_ended（中断对账：仅传入正面的执行终局证据，见
    `_validate_execution_ended`）不为 None 时，对常规对账后仍未决的
    unit 尝试诚实放弃（`_try_abandon_interrupted`）：仅“最后已知证据
    标为 RUNNING/NOT_STARTED 且 restic 核验确认无快照”的 attempt 才写
    execution_abandoned；PUBLISH_UNKNOWN/RECEIPT_FAILED、查到快照、
    无法核验、身份不符等一律保持未决，绝不动台账。"""
    ledger = Ledger(plan["project"])
    namespace = plan.get("restic_namespace") or plan.get("restic_repository")
    batch_id = plan["unit_batch"]["batch_id"]
    dispositions: dict[str, dict[str, Any]] = {}
    evidence_status = None
    if execution_ended is not None:
        proof = _validate_execution_ended(execution_ended)
        evidence_status = _batch_identity_status(session, batch_id, status,
                                                 _session_record(session))
    for item in plan["unit_batch"]["units"]:
        uid = item["unit_id"]
        attempt_id = item["attempt_id"]
        spec = P.WorkUnit.from_dict(item["spec"])
        try:
            info = reconcile_attempt(
                ledger, uid, spec, attempt_id, session,
                batch_id=plan["unit_batch"]["batch_id"],
                restic_namespace=namespace, status=status)
        except (LedgerError, BridgeError, OSError) as exc:
            info = {"disposition": DISP_UNRESOLVED,
                    "detail": f"对账通道失败（不改台账）: {exc}"}
        if info["disposition"] == DISP_UNRESOLVED and execution_ended is not None:
            try:
                info = _try_abandon_interrupted(
                    ledger, uid, attempt_id, session, batch_id=batch_id,
                    restic_namespace=namespace,
                    evidence_status=evidence_status, proof=proof,
                    list_attempt_snapshots=list_attempt_snapshots)
            except (LedgerError, BridgeError, OSError) as exc:
                info = {"disposition": DISP_UNRESOLVED,
                        "detail": f"放弃通道失败（不改台账）: {exc}"}
        dispositions[uid] = {"unit_id": uid, "attempt_id": attempt_id, **info}
    achieved = all(d["disposition"] in (DISP_RECONCILED_SUCCESS, DISP_COMPLETED)
                   for d in dispositions.values())
    unresolved = any(d["disposition"] in (DISP_UNRESOLVED, DISP_UNCOMMITTED_SUCCESS,
                                          DISP_UNTRACKED)
                     for d in dispositions.values())
    return {"session": session, "batch_id": plan["unit_batch"]["batch_id"],
            "units": dispositions, "all_resolved_succeeded": achieved,
            "any_unresolved": unresolved}


def settle_interrupted_execution(
        session: str, plan: dict[str, Any], status: dict[str, Any], *,
        proof: dict[str, str],
        list_attempt_snapshots: Callable[..., list[str]] | None = None
) -> dict[str, Any]:
    """执行中断对账入口；proof 必须是正面的执行终局证据（非观测猜测）。

    内部两条腿：①常规终态对账照旧（成功/真实失败证据照常处理）；
    ②对仍未决的 unit 尝试诚实放弃（`_try_abandon_interrupted`：最后
    已知为 RUNNING/NOT_STARTED + restic 核验无快照 →
    execution_abandoned；其余一切保持未决）。绝不因观测失联/传输失败
    动台账。"""
    return settle_terminal(session, plan, status,
                           execution_ended=_validate_execution_ended(proof),
                           list_attempt_snapshots=list_attempt_snapshots)


# ---------------------------------------------------------------------------
# 只读查询（cli `units`）
# ---------------------------------------------------------------------------

def units_report(project_name: str) -> dict[str, Any]:
    """只读台账查询：每 unit 状态 / attempts / result 摘要。"""
    ledger = Ledger(project_name)
    units = ledger.list_units()
    return {"project": project_name, "units": units}


def format_units_report(report: dict[str, Any]) -> str:
    """人类可读且稳定的排序输出；未决 attempt 与 completed 分明。"""
    lines = [f"project {report['project']}: units={len(report['units'])}"]
    for uid in sorted(report["units"]):
        view = report["units"][uid]
        lines.append(f"  {uid[:12]} state={view['state']}")
        for attempt_id in sorted(view["attempts"]):
            attempt = view["attempts"][attempt_id]
            resolution = attempt["resolution"]
            if resolution is None:
                lines.append(
                    f"    attempt {attempt_id} seq={attempt['seq']} "
                    f"session={attempt['session']} phase={attempt['phase']} "
                    f"observation={attempt['observation']} "
                    f"resolution=UNRESOLVED（未决；无确定终态证据，不裁决）")
            else:
                lines.append(
                    f"    attempt {attempt_id} seq={attempt['seq']} "
                    f"session={attempt['session']} phase={attempt['phase']} "
                    f"observation={attempt['observation']} "
                    f"resolution={resolution['outcome']} "
                    f"terminal={resolution['terminal_state']} "
                    f"source={resolution['source']}")
        result = view["result"]
        if result is not None:
            lines.append(
                f"    result snapshot={result['snapshot'][:12]}... "
                f"namespace={result['namespace']} "
                f"validator={result['validator_result'].get('status')}")
    return "\n".join(lines)


def to_json(report: dict[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
