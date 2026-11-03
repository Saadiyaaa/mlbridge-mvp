"""本地恢复台账：project/unit 命名空间下的持久化执行台账。

职责与边界（连同 docs/RECOVERY.md 一起阅读）：
- 按 project（每个 project 一个 JSON 文件）命名空间，unit 以协议稳定 id 为
  键；分别记录稳定 unit 定义、唯一 attempt_id、执行 session、当前阶段、
  终态解析来源与已提交成果（成果显式绑定 unit_id/attempt_id/session 与
  validator 定义摘要）；重开 Ledger 即从磁盘恢复。
- 目录 0700 / 文件 0600；每次变更在跨进程 flock 内读-改-写，先写同目录
  mkstemp 临时文件（0600、O_EXCL、fsync）再原子 os.replace 并 fsync 目录，
  不留半写状态。os.replace 之后若目录 fsync 失败，变更可能已提交：此时以
  磁盘重读为准报告实际状态，绝不声称"完全未写入"。
- 台账文件与锁文件若是符号链接或非普通文件，拒绝追随（fail closed）。
- 每次读入与每次写盘前，都完整校验顶层结构、schema、每个 unit 记录与
  attempt 状态机不变量（attempts 为 dict、attempt_id 跨 unit 唯一、
  seq 从 1 连续、completed iff result 存在、result 绑定最新成功 attempt、
  旧 attempt 必须已 task_failed 或 execution_abandoned、未决 attempt 至
  多一个且必须是最新）；
  损坏一律 CorruptLedgerError，绝不覆盖或重建台账文件。
- JSON 解析拒绝重复键与非有限数字（NaN/Infinity/超大浮点）。

明确不是：本模块只记录"当前控制器接受"的事实。reconcile 仅验证外部终态
证据自身的契约（必须携带 attempt/session 身份与来源声明），不假装本库独立
验证了远程平台；commit_result 是"调用者已在外部核验"的记录 API：
namespace 是显式给定字符串；manifest_sha256 / snapshot 是外部给出的摘要
（例如外部仓库中文件的 SHA256），本库不读取也不核验它们指向的数据，
外部仓库与快照内容的核验责任在调用方。没有 force retry：任务失败后的
重跑必须由调用方显式再次 start_attempt。不承诺计算 exactly-once，只保证
本地台账记录不重复提交、跨进程并发安全。
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import math
import os
import stat
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any, Callable

from . import protocol as P
from .core import BridgeError, checked_relative, ident, state_dir
from .protocol import (SCHEMA_VERSION, WorkUnit, canonical_digest, canonical_json,
                       coerce, json_guard)

# 终态三分类：SUCCEEDED（有成果）/ FAILED（真实任务失败）/
# ABANDONED（执行身份已确认终止、无任何已发布成果、且无证据表明任务本
# 身失败——即 诚实放弃"。ABANDONED 必须带 source 与 note，且禁止用于
# 有 receipt/snapshot 的 attempt（后者由调用方在放弃前核验，见
# docs/RECOVERY.md）；观测失联/传输失败绝不是 ABANDONED。
TERMINAL_STATES = frozenset({"SUCCEEDED", "FAILED", "ABANDONED"})
_TERMINAL_OUTCOME = {"SUCCEEDED": "task_succeeded", "FAILED": "task_failed",
                     "ABANDONED": "execution_abandoned"}
_OUTCOMES = frozenset(_TERMINAL_OUTCOME.values())
_OBSERVATION_STATES = frozenset({"ok", "lost"})
_UNIT_FIELDS = frozenset({"spec", "state", "attempts", "result"})
_ATTEMPT_FIELDS = frozenset({"seq", "session", "phase", "observation", "resolution"})
_RESOLUTION_FIELDS = frozenset({"outcome", "source", "terminal_state", "note"})
_RESULT_FIELDS = frozenset({"unit_id", "attempt_id", "session", "snapshot",
                            "namespace", "manifest_sha256",
                            "validator_spec_sha256", "validator_result",
                            "manifest"})


class LedgerError(BridgeError):
    """台账状态机的非法转移或身份不匹配。"""


class CorruptLedgerError(LedgerError):
    """台账文件损坏、schema 不识别或存量记录自相矛盾；fail closed。"""


def _hex(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 \
            or any(c not in "0123456789abcdefABCDEF" for c in value):
        raise LedgerError(f"{label} 必须是完整 64 位十六进制: {value!r}")
    return value.lower()


def _require_str(value: Any, label: str, *, limit: int = 256) -> str:
    if not isinstance(value, str) or not value or len(value) > limit:
        raise LedgerError(f"{label} 必须是非空字符串（≤{limit} 字符）: {value!r}")
    return value


def validate_evidence(evidence: Any) -> dict[str, Any]:
    """验证外部终态证据自身的契约；不验证其来源平台是否真实。

    ABANDONED（execution_abandoned）的额外硬约束：**必须**携带非空
    note（放弃的依据必须写明：执行身份如何确认终止、为何确信没有
    receipt/snapshot），且绝不允许纯观测失联（那只是 lost observation，
    不是终态）。"""
    if not isinstance(evidence, dict):
        raise LedgerError("证据必须是携带来源与终态的 dict")
    # 先拒绝非 str 键，后面的 sorted 才不会混入 bytes/int 等键抛裸 TypeError。
    if any(not isinstance(key, str) for key in evidence):
        raise LedgerError(
            f"证据字段名必须是字符串: {sorted(map(repr, evidence))}")
    unknown = sorted(set(evidence) - _RESOLUTION_FIELDS)
    if unknown or not {"source", "terminal_state"} <= set(evidence):
        raise LedgerError(
            f"证据字段不符（需 source/terminal_state，可选 note）: {sorted(evidence)}")
    source = _require_str(evidence["source"], "证据 source", limit=128)
    terminal = evidence["terminal_state"]
    if not isinstance(terminal, str) or terminal not in TERMINAL_STATES:
        raise LedgerError(
            f"证据 terminal_state 必须是确定性终态 {'/'.join(sorted(TERMINAL_STATES))}: "
            f"{terminal!r}")
    note = evidence.get("note")
    if note is not None and not isinstance(note, str):
        raise LedgerError("证据 note 必须是字符串")
    if terminal == "ABANDONED" and (not isinstance(note, str) or not note.strip()):
        raise LedgerError(
            "execution_abandoned 必须携带非空 note（放弃依据：执行身份如何"
            "确认终止、为何确信无 receipt/snapshot；观测失联不是依据）")
    return {"source": source, "terminal_state": terminal,
            **({"note": note} if note is not None else {})}


def _check_plain_file(path: Path, label: str) -> None:
    """路径若已存在则必须是普通文件；符号链接等一律拒绝追随。"""
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise LedgerError(f"{label} 状态不可读: {path} ({exc})") from exc
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
        raise LedgerError(f"{label} 不是普通文件，拒绝追随: {path}")


def _loads_strict(text: str, path: Path) -> Any:
    """严格 JSON 解析：拒绝重复键与非有限数字（NaN/Infinity/超大浮点）。"""
    def no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        keys = [key for key, _ in pairs]
        if len(set(keys)) != len(keys):
            raise CorruptLedgerError(f"台账含重复 JSON 键: {path}: {keys}")
        return dict(pairs)

    def finite_float(token: str) -> float:
        value = float(token)
        if not math.isfinite(value):
            raise CorruptLedgerError(f"台账含非有限数字: {path}: {token}")
        return value

    def no_constant(token: str) -> float:
        raise CorruptLedgerError(f"台账含非有限数字: {path}: {token}")

    try:
        return json.loads(text, object_pairs_hook=no_duplicate_keys,
                          parse_float=finite_float, parse_constant=no_constant)
    except CorruptLedgerError:
        raise
    except (ValueError, TypeError) as exc:
        raise CorruptLedgerError(f"台账无法解析: {path} ({exc})") from exc


class Ledger:
    """单个 project 的恢复台账；每个操作独立加锁，跨进程安全。"""

    def __init__(self, project: str) -> None:
        self.project = ident(project)

    # ---- 文件与锁 -------------------------------------------------------
    @property
    def path(self) -> Path:
        return state_dir() / "units" / f"{self.project}.json"

    @property
    def lock_path(self) -> Path:
        return state_dir() / "units" / f"{self.project}.lock"

    @contextlib.contextmanager
    def _locked(self) -> Any:
        # XDG 之下的父级路径视为用户可信配置（详见 docs/RECOVERY.md）；
        # 这里只对最后一级 units 目录做严格的符号链接/类型检查，不回溯祖先。
        directory = state_dir() / "units"
        try:
            st = os.lstat(directory)
        except FileNotFoundError:
            with suppress(OSError):
                directory.mkdir(parents=True)
            try:
                st = os.lstat(directory)
            except FileNotFoundError as exc:
                raise LedgerError(f"units 目录无法创建: {directory} ({exc})") from exc
            except OSError as exc:
                raise LedgerError(f"units 目录状态不可读: {directory} ({exc})") from exc
        except OSError as exc:
            raise LedgerError(f"units 目录状态不可读: {directory} ({exc})") from exc
        if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
            raise LedgerError(f"units 目录不是普通目录，拒绝追随: {directory}")
        # 只有在 lstat 确认真实目录后才 chmod；绝不修改符号链接目标的权限。
        if st.st_mode & 0o777 != 0o700:
            directory.chmod(0o700)
        lock_path = self.lock_path
        _check_plain_file(lock_path, "锁文件")
        try:
            fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        except OSError as exc:
            raise LedgerError(f"锁文件无法安全打开（拒绝追随）: {lock_path} ({exc})") from exc
        try:
            os.fchmod(fd, 0o600)
            fcntl.flock(fd, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    # ---- 读取与校验 -----------------------------------------------------
    def _load(self) -> dict[str, Any]:
        path = self.path
        _check_plain_file(path, "台账")
        if not path.exists():
            return {"schema_version": SCHEMA_VERSION, "project": self.project,
                    "units": {}}
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise CorruptLedgerError(f"台账不是合法 UTF-8 文本: {path} ({exc})") from exc
        except OSError as exc:
            raise CorruptLedgerError(f"台账无法读取: {path} ({exc})") from exc
        data = _loads_strict(text, path)
        self._validate_records(data, path)
        return data

    def _validate_records(self, data: Any, path: Path) -> None:
        """完整校验顶层结构、schema 与全部 unit 记录；读入/写盘前都调用。"""
        if not isinstance(data, dict) or set(data) != {"schema_version", "project", "units"}:
            raise CorruptLedgerError(f"台账顶层结构不符: {path}")
        version = data["schema_version"]
        if isinstance(version, bool) or not isinstance(version, int) \
                or version != SCHEMA_VERSION:
            raise CorruptLedgerError(f"台账 schema_version 不识别: {path}")
        if data["project"] != self.project or not isinstance(data["units"], dict):
            raise CorruptLedgerError(f"台账 project/units 结构不符: {path}")
        seen_attempts: set[str] = set()
        for uid, record in data["units"].items():
            self._validate_unit(uid, record, path)
            shared = seen_attempts & set(record["attempts"])
            if shared:
                raise CorruptLedgerError(
                    f"attempt id 跨 unit 重复: {sorted(shared)}: {path}")
            seen_attempts.update(record["attempts"])

    def _validate_unit(self, uid: Any, record: Any, path: Path) -> None:
        def fail(detail: str) -> CorruptLedgerError:
            return CorruptLedgerError(f"台账 unit {uid} 记录损坏: {path}: {detail}")

        if not isinstance(uid, str) or set(uid) - set("0123456789abcdef") \
                or len(uid) != 64:
            raise fail("unit id 不是 64 位小写十六进制")
        if not isinstance(record, dict) or set(record) != _UNIT_FIELDS:
            raise fail("unit 记录字段不符")
        try:
            spec = WorkUnit.from_dict(record["spec"])
        except BridgeError as exc:
            raise fail(f"spec 无法解析 ({exc})") from exc
        if spec.identity != uid:
            raise fail("spec 与 unit id 不一致")
        if spec.project != self.project:
            raise fail(f"spec.project 与台账 project 不一致: {spec.project!r}")
        state = record["state"]
        if state not in ("pending", "completed"):
            raise fail(f"未知 state {state!r}")

        attempts = record["attempts"]
        if not isinstance(attempts, dict):
            raise fail("attempts 必须是 dict")
        seqs: set[int] = set()
        unresolved: list[str] = []
        for attempt_id, attempt in attempts.items():
            if not isinstance(attempt_id, str):
                raise fail(f"attempt id 非法: {attempt_id!r}（必须是字符串）")
            try:
                ident(attempt_id)
            except BridgeError as exc:
                raise fail(f"attempt id 非法 ({exc})") from exc
            if not isinstance(attempt, dict) or set(attempt) != _ATTEMPT_FIELDS:
                raise fail(f"attempt {attempt_id} 字段不符")
            seq = attempt["seq"]
            if isinstance(seq, bool) or not isinstance(seq, int) or seq < 1 \
                    or seq in seqs:
                raise fail(f"attempt {attempt_id} 的 seq 非法或重复")
            seqs.add(seq)
            if not _plain(attempt["session"], 63) or not _valid_ident(attempt["session"]):
                raise fail(f"attempt {attempt_id} session 非法")
            if not _plain(attempt["phase"], 64):
                raise fail(f"attempt {attempt_id} phase 非法")
            observation = attempt["observation"]
            # frozenset 成员判断会对 list/dict 抛裸 TypeError；先做 str 检查，
            # 任何损坏类型都必须落到 CorruptLedgerError。
            if not isinstance(observation, str) \
                    or observation not in _OBSERVATION_STATES:
                raise fail(f"attempt {attempt_id} observation 非法")
            resolution = attempt["resolution"]
            if resolution is not None:
                if not isinstance(resolution, dict) \
                        or set(resolution) - _RESOLUTION_FIELDS \
                        or not {"outcome", "source", "terminal_state"} <= set(resolution):
                    raise fail(f"attempt {attempt_id} resolution 结构不符")
                outcome = resolution["outcome"]
                terminal = resolution["terminal_state"]
                if not isinstance(outcome, str) or outcome not in _OUTCOMES \
                        or not isinstance(terminal, str) \
                        or terminal not in TERMINAL_STATES \
                        or _TERMINAL_OUTCOME[terminal] != outcome \
                        or not _plain(resolution["source"], 128):
                    raise fail(f"attempt {attempt_id} resolution 内容不符")
                note = resolution.get("note")
                if note is not None and not isinstance(note, str):
                    raise fail(f"attempt {attempt_id} resolution.note 非法")
            else:
                unresolved.append(attempt_id)

        # ---- 状态机不变量（读入与写盘前都必须成立） -----------------------
        if seqs != set(range(1, len(attempts) + 1)):
            raise fail(f"attempt seq 必须从 1 连续编号，实际 {sorted(seqs)}")
        latest_id = max(attempts, key=lambda k: attempts[k]["seq"]) if attempts else None
        if len(unresolved) > 1:
            raise fail(f"存在多个未解决 attempt: {sorted(unresolved)}")
        if len(unresolved) == 1 and unresolved[0] != latest_id:
            raise fail(f"未解决 attempt 不是最新: {unresolved[0]} != {latest_id}")
        for attempt_id, attempt in attempts.items():
            resolution = attempt["resolution"]
            if attempt_id != latest_id and (resolution is None
                                            or resolution["outcome"]
                                            not in ("task_failed",
                                                    "execution_abandoned")):
                raise fail(f"旧 attempt {attempt_id} 必须已 task_failed 或 "
                           f"execution_abandoned，实际 resolution={resolution}")
        if state == "completed" and record["result"] is None:
            raise fail("state=completed 但 result 缺失")
        if state == "pending" and record["result"] is not None:
            raise fail("state=pending 却携带 result")
        if record["result"] is not None:
            self._validate_result(uid, record["result"], spec, attempts,
                                  latest_id, fail)

    def _validate_result(self, uid: str, result: Any, spec: WorkUnit,
                         attempts: dict[str, Any], latest_id: str | None,
                         fail: Callable[[str], CorruptLedgerError]) -> None:
        if not isinstance(result, dict) or set(result) != _RESULT_FIELDS:
            raise fail("result 字段不符")
        _result_hex(result["snapshot"], fail, "snapshot")
        _result_hex(result["manifest_sha256"], fail, "manifest_sha256")
        _result_hex(result["validator_spec_sha256"], fail, "validator_spec_sha256")
        if not _plain(result["namespace"], 256):
            raise fail("result.namespace 非法")
        # 成果身份绑定：必须对应该 unit 最新一次被 task_succeeded 解决的 attempt
        for field in ("unit_id", "attempt_id", "session"):
            if not isinstance(result[field], str):
                raise fail(f"result.{field} 必须是字符串")
        if result["unit_id"] != uid:
            raise fail(f"result 绑定到错误 unit: {result['unit_id']!r}")
        if result["attempt_id"] != latest_id:
            raise fail(f"result 绑定的不是最新 attempt: "
                       f"{result['attempt_id']!r} != {latest_id!r}")
        latest = attempts[latest_id]
        if latest["session"] != result["session"]:
            raise fail("result.session 与最新 attempt 的 session 不符")
        if latest["resolution"] is None \
                or latest["resolution"]["outcome"] != "task_succeeded":
            raise fail("result 绑定的 attempt 未被 task_succeeded 终态证据解决")
        validator = result["validator_result"]
        if not isinstance(validator, dict) or not isinstance(validator.get("kind"), str) \
                or not isinstance(validator.get("status"), str) \
                or validator["status"] != "passed" \
                or validator["kind"] != spec.validator["kind"]:
            raise fail("result.validator_result 与 spec.validator 不符")
        if result["validator_spec_sha256"] != canonical_digest(spec.validator):
            raise fail("result.validator_spec_sha256 与 spec.validator 的"
                       " canonical JSON SHA256 不符")
        manifest = result["manifest"]
        if manifest is not None:
            if not isinstance(manifest, dict) \
                    or any(not isinstance(key, str) for key in manifest):
                raise fail("result.manifest 必须是键为字符串的 dict 或 None")
            for key in manifest:
                try:
                    inside = _within_outputs(key, spec.outputs)
                except LedgerError as exc:
                    raise fail(f"result.manifest 键必须落在 spec.outputs 树内 ({exc})") from exc
                if not inside:
                    raise fail(f"result.manifest 键必须落在 spec.outputs 树内: {key!r}")

    # ---- 变更框架 -------------------------------------------------------
    def _transact(self, change: Callable[[dict[str, Any]], Any]) -> Any:
        with self._locked():
            data = self._load()
            result = change(data)
            self._save(data)
            return result

    def _save(self, data: dict[str, Any]) -> None:
        # 写盘前对最终状态做完整校验：任何程序路径错误都无法把非法状态
        # 写进整个 project 的台账。
        self._validate_records(data, self.path)
        path = self.path
        _check_plain_file(path, "台账")
        try:
            text = json.dumps(data, ensure_ascii=False, indent=2,
                              sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as exc:  # 理论上已被协议校验兜住
            raise CorruptLedgerError(f"台账序列化失败（拒绝写入）: {path}: {exc}") from exc
        fd, tmp_name = tempfile.mkstemp(dir=path.parent,
                                        prefix=f".{path.name}.", suffix=".tmp")
        tmp = Path(tmp_name)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:
            with suppress(OSError):
                os.close(fd)
            tmp.unlink(missing_ok=True)
            raise LedgerError(f"台账写入失败: {path} ({exc})") from exc
        try:
            os.replace(tmp, path)
        except OSError as exc:
            tmp.unlink(missing_ok=True)
            raise LedgerError(f"台账写入失败: {path} ({exc})") from exc
        # 目录 fsync：失败时 os.replace 可能已生效，此时以磁盘重读为准
        # 报告实际状态，绝不声称"完全未写入"。
        try:
            dir_fd = os.open(path.parent, os.O_RDONLY)
        except OSError as exc:
            self._report_uncertain(path, text, exc, "目录无法打开")
        try:
            os.fsync(dir_fd)
        except OSError as exc:
            self._report_uncertain(path, text, exc, "目录 fsync 失败")
        finally:
            os.close(dir_fd)

    def _report_uncertain(self, path: Path, text: str, exc: OSError,
                          detail: str) -> None:
        """os.replace 已执行后的持久化失败：重读磁盘判定实际状态再报告。

        重读本身也可能失败（例如磁盘内容已不可解码的 UnicodeDecodeError）；
        那同样归入 "重读内容与预期不一致，磁盘状态必须人工检查"，
        绝不假称新状态已确认。"""
        on_disk: str | None = None
        with suppress(OSError, UnicodeDecodeError):
            on_disk = path.read_text(encoding="utf-8")
        if on_disk == text:
            raise LedgerError(
                f"台账已原子替换且重读确认磁盘内容已是新状态；但{detail}，"
                f"掉电后的持久性不确定: {path} ({exc})") from exc
        raise LedgerError(
            f"台账{detail}，且重读内容与预期不一致，磁盘状态必须人工检查: "
            f"{path} ({exc})") from exc

    # ---- 查询 -----------------------------------------------------------
    def get(self, unit_id: str) -> dict[str, Any]:
        """返回 unit 的独立快照：state / spec / attempts / result / 最新 attempt。

        快照是从磁盘重读新建的对象，不是台账内部状态的"只读视图"；对返回
        的 dict/list 的任何修改都不影响持久化，也不会被后续 Ledger 操作
        看到。因此同一快照内无需多余深拷贝，attempts 原样共享磁盘读出的
        对象即可。
        """
        with self._locked():
            record = self._unit(self._load(), unit_id)
            attempts: dict[str, dict[str, Any]] = {}
            latest_id, latest_seq = None, 0
            for attempt_id, attempt in record["attempts"].items():
                attempts[attempt_id] = attempt  # 整个返回值都是独立快照
                if attempt["seq"] > latest_seq:
                    latest_id, latest_seq = attempt_id, attempt["seq"]
            latest = attempts.get(latest_id or "", None)
            return {
                "unit_id": unit_id,
                "state": record["state"],
                "spec": record["spec"],
                "attempts": attempts,
                "latest_attempt": latest_id,
                "latest_resolved": latest is not None
                and latest["resolution"] is not None,
                "result": record["result"],
            }

    # ---- 注册与执行 -----------------------------------------------------
    def list_units(self) -> dict[str, dict[str, Any]]:
        """只读全量快照：每个条目与 ``get(unit_id)`` 同构。

        在单次锁内读出并构造全部快照，比逐个 ``get`` 更一致（不会被并发
        写入拆散成不同代的视图）；但只读对账仍应 discern 每个 unit 自己
        的 ``latest_attempt``/``latest_resolved``，因为快照之间互不共享
        状态。损坏台账由 ``_load`` 的全量校验 fail closed。"""
        with self._locked():
            data = self._load()
            out: dict[str, dict[str, Any]] = {}
            for uid, record in sorted(data["units"].items()):
                attempts: dict[str, dict[str, Any]] = dict(record["attempts"])
                latest_id, latest_seq = None, 0
                for attempt_id, attempt in attempts.items():
                    if attempt["seq"] > latest_seq:
                        latest_id, latest_seq = attempt_id, attempt["seq"]
                latest = attempts.get(latest_id or "", None)
                out[uid] = {
                    "unit_id": uid,
                    "state": record["state"],
                    "spec": record["spec"],
                    "attempts": attempts,
                    "latest_attempt": latest_id,
                    "latest_resolved": latest is not None
                    and latest["resolution"] is not None,
                    "result": record["result"],
                }
            return out

    def plan(self, unit: WorkUnit | dict[str, Any]) -> str:
        """登记一个有意执行的工作单元（幂等），返回稳定 unit id。

        unit 的 project 必须与台账 project 一致；已登记且定义的 canonical
        JSON 一致 → 幂等返回；unit id 相同但定义序列化不一致（例如 1 与
        1.0 差异）→ 冲突拒绝，绝不覆盖。
        """
        spec = coerce(unit)
        if spec.project != self.project:
            raise LedgerError(
                f"unit 属于 project {spec.project!r}，与台账 {self.project!r} 不一致")
        uid = spec.identity

        def change(data: dict[str, Any]) -> str:
            record = data["units"].get(uid)
            if record is not None:
                stored = json.dumps(record["spec"], ensure_ascii=False,
                                    sort_keys=True, separators=(",", ":"))
                if stored != canonical_json(spec):
                    raise LedgerError(
                        f"unit id 相同但定义不同，拒绝覆盖登记: {uid}")
                return uid
            data["units"][uid] = {"spec": spec.to_dict(), "state": "pending",
                                  "attempts": {}, "result": None}
            return uid
        return self._transact(change)

    def start_attempt(self, unit_id: str, attempt_id: str, session: str) -> None:
        """为 pending unit 显式开启一次执行（attempt_id 全局唯一）。

        拒绝：unit 已提交成果；最新 attempt 仍未解决；最新 attempt 已被
        task_succeeded 解决但成果未提交；attempt_id 与台账内任何已有
        attempt 重复。task_failed **或 execution_abandoned** 之后都
        允许显式开新 attempt（默认策略是不自动重试，重跑必须由调用者
        再次调用本方法；本方法就是那个显式调用点，无任何 force 参数）。
        """
        attempt_id = _must_ident(attempt_id, "attempt_id")
        session = _must_ident(session, "session")

        def change(data: dict[str, Any]) -> None:
            record = self._unit(data, unit_id)
            if record["state"] == "completed":
                raise LedgerError(f"unit 已提交成果，不能再开 attempt: {unit_id}")
            for other in data["units"].values():
                if attempt_id in other["attempts"]:
                    raise LedgerError(f"attempt_id 必须在台账内唯一，已存在: {attempt_id}")
            attempts = record["attempts"]
            if attempts:
                latest = max(attempts.values(), key=lambda a: a["seq"])
                if latest["resolution"] is None:
                    latest_id = max(attempts, key=lambda k: attempts[k]["seq"])
                    raise LedgerError(f"已有未解决 attempt {latest_id}；"
                                      "请先用 reconcile 提交终态证据")
                if latest["resolution"]["outcome"] == "task_succeeded":
                    raise LedgerError(
                        "上次执行已有成功终态证据但成果未提交；请先 commit_result，"
                        "不要静默重复执行同一 unit")
            seq = 1 + max((a["seq"] for a in attempts.values()), default=0)
            attempts[attempt_id] = {"seq": seq, "session": session,
                                    "phase": "start", "observation": "ok",
                                    "resolution": None}
        self._transact(change)

    def set_phase(self, unit_id: str, attempt_id: str, phase: str) -> None:
        """更新未决 attempt 的执行阶段；resolved 之后阶段冻结。"""
        self._mutate_attempt(unit_id, attempt_id,
                             lambda attempt: attempt.update(
                                 {"phase": _require_str(phase, "phase", limit=64)}))

    def set_observation(self, unit_id: str, attempt_id: str,
                        observation: str) -> None:
        """仅把观测状态改为 ok/lost；不改变阶段，不转失败，不授权重跑。"""
        if not isinstance(observation, str) or observation not in _OBSERVATION_STATES:
            raise LedgerError(f"observation 只能是 {'/'.join(sorted(_OBSERVATION_STATES))}: "
                              f"{observation!r}")
        self._mutate_attempt(unit_id, attempt_id,
                             lambda attempt: attempt.update({"observation": observation}))

    def _mutate_attempt(self, unit_id: str, attempt_id: str,
                        mutate: Callable[[dict[str, Any]], None]) -> None:
        def change(data: dict[str, Any]) -> None:
            attempt = self._attempt(self._unit(data, unit_id), attempt_id)
            if attempt["resolution"] is not None:
                raise LedgerError(f"attempt 已被终态证据解决，不再变更: {attempt_id}")
            mutate(attempt)
        self._transact(change)

    def reconcile(self, unit_id: str, attempt_id: str, session: str,
                  evidence: dict[str, Any]) -> str:
        """用携带 attempt/session 身份与来源的外部终态证据解除未决执行。

        返回 outcome（task_succeeded / task_failed / execution_abandoned）；
        重复提交完全一致的证据幂等成功，不同证据 → 冲突拒绝。绝不把已
        有终态的 attempt 改写（包括把 task_failed 改写成
        execution_abandoned，或反之）。
        """
        evidence = validate_evidence(evidence)
        outcome = _TERMINAL_OUTCOME[evidence["terminal_state"]]

        def change(data: dict[str, Any]) -> str:
            attempt = self._attempt(self._unit(data, unit_id), attempt_id)
            if attempt["session"] != _must_ident(session, "session"):
                raise LedgerError(f"证据 session 与 attempt 身份不符: {session!r}")
            if attempt["resolution"] is not None:
                if attempt["resolution"] == {"outcome": outcome, **evidence}:
                    return outcome
                raise LedgerError(
                    f"attempt 已被不同终态证据解决，拒绝改写: {attempt_id}")
            attempt["resolution"] = {"outcome": outcome, **evidence}
            return outcome
        return self._transact(change)

    def commit_result(self, unit_id: str, attempt_id: str, session: str, *,
                      snapshot: str, namespace: str, manifest_sha256: str,
                      validator_spec_sha256: str,
                      validator_result: dict[str, Any],
                      manifest: dict[str, Any] | None = None) -> str:
        """记录调用者已在外部核验成功的成果；校验身份一致与 validator 定义摘要。

        要求：attempt 是该 unit 最新一次执行、已被 task_succeeded 终态证据
        解决、session 匹配；snapshot / manifest_sha256 /
        validator_spec_sha256 为完整 64 位 hex，其中
        validator_spec_sha256 必须等于 unit.validator（canonical JSON）的
        SHA256；validator_result.kind 必须与 unit.validator.kind 一致且
        status == "passed"；manifest（可选）键必须落在 unit.outputs 子树内。
        成果显式绑定 unit_id / attempt_id / session，并纳入幂等比较：完全
        一致的重复提交幂等返回 "duplicate"，首次返回 "recorded"，任何不同
        内容（含摘要差异）→ 冲突拒绝并保持已存结果不变。

        本方法不核验远端内容：namespace 是显式给定字符串，manifest_sha256
        可以是外部仓库中文件的字节摘要，本库不读取也不核实它们指向的
        数据；外部仓库与快照内容的核验责任在调用方（详见 docs/RECOVERY.md）。
        也没有任何绕过校验的 force 通道。
        """
        attempt_id = _must_ident(attempt_id, "attempt id")
        session = _must_ident(session, "session")
        if manifest is not None and not isinstance(manifest, dict):
            raise LedgerError("manifest 必须是 dict 或 None")
        try:
            validator_result = json_guard(validator_result, "validator_result")
            guarded_manifest = json_guard(manifest, "manifest") \
                if manifest is not None else None
        except P.ProtocolError as exc:
            raise LedgerError(f"成果载荷非法: {exc}") from exc
        if not isinstance(validator_result, dict):
            raise LedgerError("validator_result 必须是 dict")
        for field in ("kind", "status"):
            if not isinstance(validator_result.get(field), str):
                raise LedgerError(f"validator_result 缺少 str 字段 {field}")
        result = {
            "unit_id": unit_id,
            "attempt_id": attempt_id,
            "session": session,
            "snapshot": _hex(snapshot, "snapshot"),
            "namespace": _require_str(namespace, "namespace"),
            "manifest_sha256": _hex(manifest_sha256, "manifest_sha256"),
            "validator_spec_sha256": _hex(validator_spec_sha256,
                                          "validator_spec_sha256"),
            "validator_result": validator_result,
            "manifest": guarded_manifest,
        }

        def change(data: dict[str, Any]) -> str:
            record = self._unit(data, unit_id)
            attempt = self._attempt(record, attempt_id)  # 验 id 合法且存在
            attempts = record["attempts"]
            latest_id = max(attempts, key=lambda k: attempts[k]["seq"])
            if attempt_id != latest_id:
                raise LedgerError(
                    f"拒绝非最新 attempt 的成果提交: {attempt_id} != {latest_id}")
            if attempt["session"] != session:
                raise LedgerError(f"成果 session 与 attempt 身份不符: {session!r}")
            resolution = attempt["resolution"]
            if resolution is None or resolution["outcome"] != "task_succeeded":
                raise LedgerError(
                    "attempt 未被 task_succeeded 终态证据解除，不能提交成果；"
                    "未检验路径或未绑定回执不构成成功")
            spec = WorkUnit.from_dict(record["spec"])
            if result["manifest"] is not None:
                for key in result["manifest"]:
                    if not _within_outputs(key, spec.outputs):
                        raise LedgerError(
                            f"manifest 键必须落在 unit.outputs 树内: {key!r}")
            if result["validator_spec_sha256"] != canonical_digest(spec.validator):
                raise LedgerError(
                    "validator_spec_sha256 与 unit.validator 的 canonical JSON "
                    f"SHA256 不符: {result['validator_spec_sha256']!r}")
            if validator_result["kind"] != spec.validator["kind"]:
                raise LedgerError(
                    f"validator_result.kind 与 unit.validator.kind 不一致: "
                    f"{validator_result['kind']!r} != {spec.validator['kind']!r}")
            if validator_result["status"] != "passed":
                raise LedgerError(
                    f"validator_result.status 必须是 'passed': {validator_result['status']!r}")
            existing = record["result"]
            if record["state"] == "completed":
                if existing != result:
                    raise LedgerError("unit 已提交不同成果，拒绝改写")
                return "duplicate"
            record["result"] = result
            record["state"] = "completed"
            return "recorded"
        return self._transact(change)

    def _unit(self, data: dict[str, Any], unit_id: str) -> dict[str, Any]:
        record = data["units"].get(unit_id)
        if record is None:
            raise LedgerError(f"台账中不存在该 unit: {self.project}/{unit_id}")
        return record

    def _attempt(self, record: dict[str, Any], attempt_id: str) -> dict[str, Any]:
        if not _valid_ident(attempt_id):
            raise LedgerError(f"attempt id 非法: {attempt_id!r}")
        attempt = record["attempts"].get(attempt_id)
        if attempt is None:
            raise LedgerError(f"该 unit 不存在此 attempt: {attempt_id}")
        return attempt


def _plain(value: Any, limit: int) -> bool:
    return isinstance(value, str) and bool(value) and len(value) <= limit


def _valid_ident(value: str) -> bool:
    if not isinstance(value, str):
        return False
    try:
        ident(value)
    except BridgeError:
        return False
    return True


def _within_outputs(key: str, outputs: list[str]) -> bool:
    """manifest 键必须落在某个声明输出路径（或其子树）之内；非法即抛。

    先做严格相对路径校验（拒绝 ``..``、绝对路径、``.git``、NUL 字节、
    空路径），再做路径组件级包含判断，不做字符串裸前缀比较。
    """
    if not isinstance(key, str) or "\0" in key:
        raise LedgerError(f"manifest 键不是合法字符串路径: {key!r}")
    try:
        rel = checked_relative(key)
    except (BridgeError, ValueError) as exc:
        raise LedgerError(
            f"manifest 键不是合法相对输出路径: {key!r} ({exc})") from exc
    rel = rel.rstrip("/")
    for output in outputs:
        base = output.rstrip("/")
        if rel == base or rel.startswith(base + "/"):
            return True
    return False


def _must_ident(value: str, label: str) -> str:
    # 非 str 输入（例如调用方传了 int/list）必须变成 LedgerError，
    # 而不是从 re.fullmatch 冒出裸 TypeError。
    if not isinstance(value, str):
        raise LedgerError(f"{label} 必须是字符串: {value!r}")
    try:
        return ident(value)
    except BridgeError as exc:
        raise LedgerError(f"{label} 非法: {value!r} ({exc})") from exc


def _result_hex(value: Any, fail: Callable[[str], CorruptLedgerError],
                label: str) -> None:
    if not isinstance(value, str) or len(value) != 64 \
            or any(c not in "0123456789abcdef" for c in value):
        raise fail(f"result.{label} 必须是完整 64 位小写十六进制")
