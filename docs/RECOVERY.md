# 可恢复执行协议与本地恢复台账（当前状态）

本文档描述第一阶段已落地的两块能力：`mlbridge.protocol`（版本化工作单
元协议）与 `mlbridge.ledger`（持久化本地恢复台账）。范围边界务必先读：
这里只实现"控制器本地可恢复的记录与仲裁"，**不**宣称远端验证、不承诺
exactly-once。

第二阶段的远端执行切片（worker 的 `unit_batch` 顺序执行/发布/回执）
见 `docs/UNIT_EXECUTION.md`；其远端执行器由 `mlbridge run --unit-batch`
**opt-in** 接入（launch 显式 stage `batch_worker.py`，legacy 路径零改
变），详见本文末尾“批模式（opt-in）”一节。

## mlbridge.protocol —— 工作单元协议

- 显式 `schema_version`（当前 = `SCHEMA_VERSION = 2`）；未知 schema、
  缺失/未知字段、NaN/Infinity、非 JSON 容器（tuple/set）、非 str 键一律
  抛 `ProtocolError`（注释字段说明见源码）。
- 单元定义字段（全部必填）：`project / repository / commit / inputs /
  environment / command / params / outputs / validator`。
- **稳定 unit id**：对参与成果语义的上述全部字段做 canonical JSON
  （排序键、紧凑分隔符、`allow_nan=False`）后取 SHA256
  （`WorkUnit.identity` / `unit_id(spec)`）。规范化动作仅限语义无损项：
  commit/输入 snapshot 统一小写 40/64 位十六进制，repository 用
  `core.normalize_git_url` 去尾斜杠与 `.git`；键序不影响 id。
- **真正不可变的内部表示**：WorkUnit 实例唯一权威状态是构造时校验/
  规范化后固化的 canonical JSON 字符串；`identity` 由它一次算定，永不
  重算。公共构造器 `WorkUnit(**字段)` 与 `from_dict` 走同一条校验路径，
  不可能绕过校验构造出非法实例（非法值一律 `ProtocolError`，包括非法
  标量类型与混合未知/非 str 键）。所有序列字段按访问返回独立副本：
  对 `inputs`/`params`/`validator`/`to_dict()` 等的外部修改不会写回，
  identity 与定义永不失配；属性赋值/删除被拒绝。`__eq__`/`__hash__`
  基于 canonical JSON，公开构造与 `from_dict` 产物等值。
- **session / attempt_id 等执行期信息不属于协议**：作为未知字段直接被
  拒绝，不可能混入身份计算。
- 规则性拒绝：outputs 相对路径非法（绝对/`..`/`.git`/空）与 NUL 字节；
  command 非 argv 数组 / 非空字符串 / NUL；environment 非
  `dict[str, str]` 等（params 键同规则，
  不做整数校验等任何特殊处理）。
- environment 指纹必须由调用方**显式给出**；本模块不读取运行环境，
  目前没有已实现的环境自动采集/缓存——接入方不得默认它存在。

## mlbridge.ledger —— 本地恢复台账

存储：`core.state_dir() / units / <project>.json`（0600），目录 0700，
并发锁 `<project>.lock`（跨进程 `flock`，`O_NOFOLLOW` 打开）。台账/锁
文件若是符号链接或非普通文件一律拒绝追随；`units` 目录在创建/lstat
之后必须是普通目录（拒绝符号链接/文件型路径）且权限 0700，并且 chmod
只在 lstat 确认真实目录之后执行——永不修改符号链接目标的权限。XDG 之
下的父级路径视为用户可信配置，本层不做祖先目录回溯。每次修改
在锁内先完整读-改-写：同目录 `mkstemp` 临时文件（0600，随机名 O_EXCL）
fsync → `os.replace` 原子替换 → 目录 fsync。`os.replace` 之后若目录
fsync 失败，变更可能已提交：此时重读磁盘确认实际状态再报告，绝不声称
"完全未写入"（重读自身损坏，包括不可解码的 UnicodeDecodeError，同样
归入"磁盘状态必须人工检查"）。重新 `Ledger(project)` 即从磁盘恢复。
解析失败、schema 不识别、重复 JSON 键、非有限数字（NaN/Infinity/超大
浮点）一律 `CorruptLedgerError` fail closed，**从不覆盖或重建**台账。

每次读入与每次写盘前完整校验全部不变量（写入前的全量校验使程序路径
无法把非法状态写进整个 project）：attempts 为 dict、attempt_id 跨 unit
唯一、spec 归属台账 project 且与 unit id 一致、attempt seq 从 1 连续，
损坏为 list/dict 等非法类型的字段（如 `observation`、`resolution` 的
`outcome`/`terminal_state`）也一律 `CorruptLedgerError`，而非裸 TypeError：
`completed` 当且仅当 `result` 存在、成果显式绑定 `unit_id/attempt_id/
session` 且必须对应最新一次被 `task_succeeded` 解决的 attempt、旧
attempt 必须已 `task_failed` 或 `execution_abandoned`、未决 attempt
至多一个且必须是最新。

状态机（每个 unit 独立）：

```
plan(unit) ──▶ pending ──▶ commit_result() 成功 ──▶ completed(终态)
                │  ▲ attempt（唯一活跃；flock 内先到先得）
   start_attempt│  │ reconcile(终态证据: 携带 attempt/session/来源；
                ▼  │   FAILED→task_failed，ABANDONED→execution_abandoned)
             未解决 attempt（FAILED 或 ABANDONED 后允许显式开新 attempt）
```

- `Ledger.plan(unit)`：登记有意执行的 unit；unit 的 project 必须与台账
  project 一致；重复登记同 id 且序列化一致 → 幂等；不一致 → 拒绝覆盖。
- `start_attempt(unit_id, attempt_id, session)`：attempt_id 全台唯一；
  已提交 unit、存在未解决 attempt、或"成功证据已到但成果未提交"时拒绝。
  任务失败（task_failed）**或诚实放弃（execution_abandoned）之后都不
  会自动重试**；重跑必须显式再调用本方法（也无任何 force 参数）。
- `set_phase` / `set_observation(att, "ok"|"lost")`：只记录执行阶段与
  观测状态；observation lost 不改变 phase、不生成失败、不解锁新
  attempt，也**绝不构成** execution_abandoned 的证据。
- `reconcile(unit_id, attempt_id, session, evidence)`：解除未决执行的
  唯一途径。`evidence` 必须是 `{"source": str, "terminal_state":
  "SUCCEEDED"|"FAILED"|"ABANDONED", "note"?: str}`；session 必须与
  attempt 一致，否则拒绝。重复提交一致证据幂等；不一致拒绝（绝不改写：
  task_failed 不得改成 execution_abandoned，或反之）。LEDGER 只验证
  该契约，**不**验证来源平台是否真实。`ABANDONED`（outcome =
  `execution_abandoned`，诚实放弃）另有硬约束：**必须**携带非空
  note（放弃依据：执行身份如何确认终止、为何确信无 receipt/
  snapshot）；观测失联/传输失败不是放弃依据；有 receipt/snapshot 的
  attempt 禁止放弃（调用方在放弃前核验，见下“中断对账”）
- `commit_result(unit_id, attempt_id, session, *, snapshot, namespace,
  manifest_sha256, validator_spec_sha256, validator_result, manifest=None)`：
  记录"调用者已在外部核验成功"的成果。要求 attempt 为最新、已被
  task_succeeded 证据解除、session 匹配；`snapshot`/`manifest_sha256`/
  `validator_spec_sha256` 为完整 64 位 hex；`validator_spec_sha256` 必须
  等于 unit.validator（canonical JSON）的 SHA256；
  `validator_result.kind` 必须等于 unit 的 `validator.kind` 且
  `status == "passed"`；manifest 键先过严格相对路径校验（拒
  `..`/绝对路径/`.git`/NUL、统一规范化）再按路径组件判断是否落在
  unit.outputs 子树内。成果显式绑定 `unit_id/attempt_id/session`，这些
  身份字段纳入幂等比较。首次返回 `"recorded"`，完全一致重放返回
  `"duplicate"`，不同内容拒绝且已存值不变。提交后 unit completed，不得
  再开 attempt。
  本库不验证远端内容：`namespace` 是显式给定字符串（本层不约束其形状）；
  `manifest_sha256` 可以是外部仓库中文件的字节摘要，本层不读取文件核验
  它；外部仓库与快照内容的核验责任在调用方。
- `get(unit_id)`：从磁盘重读的**独立快照**（state / spec / attempts /
  result / latest_attempt），不是台账内部状态的只读视图；对快照的
  dict/list 修改不影响持久化，也不会被后续操作看到。`list_units()`
  是在单次锁内读出的全量只读快照（条目与 `get` 同构），供 `mlbridge
  units` 查询只用，不提供任何变更通道。

## 批模式（opt-in）：`mlbridge run --unit-batch FILE`

这是把 ledger 接入执行/启动/恢复链路的唯一入口；**未给出
`--unit-batch` 时 legacy `run` 的参数、staging 上传集合与行为逐字节
不变**。实现：`mlbridge.batch_control`（本地控制面）+
`colab_runner.launch(..., unit_batch=...) / watch(..., batch_plan=...)`
（分配与对账接入），远端执行器 `mlbridge.batch_worker` 由 launch
**显式 stage**（`/content/batch_worker.py`；缺文件在任何分配前 fail
closed）。

批定义文件（版本化 JSON，严格解析：拒绝重复键 / NaN / Infinity）：

```jsonc
{
  "schema_version": 1,
  "project": "demo",                      // 必须 = CLI 项目名
  "environment": {"python": "3.12"},      // 显式指纹；可选
                                          // "setup_sha256"（必须与
                                          // canonical JSON SHA256(job.setup)
                                          // 一致，缺省自动计算注入）
  "units": [                              // 有序
    {
      "spec": { ...WorkUnit 定义... },    // 可缺省 project/repository/
                                          // commit/environment（由 CLI 上下文
                                          // 注入；显式给出则必须一致，跨
                                          // project/不符 commit 直接拒绝）
      "execution": {"timeout_seconds": 600},  // 必填；正有限数
      "unit_id": "…64hex…"              // 可选；给出时必须等于派生身份
    }
  ],
  "unit_batch_id": "batch-…",             // 可选；必须与派生身份一致
  "manifest_limits": {...},               // 可选（透传 batch_worker 契约）
  "publish_timeout_seconds": 900          // 可选（同上）
}
```

- **批次身份 batch_id**：默认由「项目 + 批次定义」（含 environment
  指纹、逐单元 spec+execution、manifest_limits、publish_timeout）的
  canonical JSON SHA256 派生（`batch-<摘要前 48 hex>`）。文件名不参与；
  显式 `unit_batch_id` 必须等于派生值，否则拒绝（身份永远锚定定义
  内容，绝不轻信任意标签）。
- **attempt_id 由本工具生成**（`att-<unit_id 前12 hex>-<台账序号>`），
  不写入批文件；同一批内容用 `--retry-failed` 重跑会得到新 attempt_id。
- **setup 来源**：来自 CLI `--job`（同 legacy）的 `job.setup`；批文件
  的 environment 指纹因此与该 setup 绑定。`--input` 与批模式互斥
  （输入快照来自 unit spec，plan.inputs = 名称排序后快照）。

### 启动前（GPU 分配前）的校验链与恢复对账

顺序固定，任何一步失败都在 `colab sessions`/`colab new` 之前，无任何
副作用：① 严格 JSON 解析 → ② 顶层 schema/未知字段拒绝 → ③ 每个
unit 走 `protocol.WorkUnit`（缺必需字段 / params 非法 / 非法路径 /
跨 project 全部拒绝）
→ ④ `batch_worker.validate_batch`（远端计划的镜像纯校验，先以占位
身份、后以真实 attempt_id 再验一次）。校验通过后对**每个** unit 幂等
`Ledger.plan` 登记，然后逐 unit 对账：

- 已 `completed` → 跳过（不重算；绝不因批中部分完成就重跑全部）。
- 存在未决 attempt → **不分配新实例**，先读该 attempt 所绑定 session
  的远端状态（`mlbridge_state.json` 的 `batch.outcomes`，要求
  batch.session 与 unit/attempt 身份一致）及回执：
  - `SUCCEEDED` + 契约回执（身份/摘要全匹配）→
    `reconcile(task_succeeded)` + `commit_result`。
  - `TASK_FAILED` → `reconcile(task_failed)`；默认**不自动重试**：
    必须显式 `--retry-failed` 才允许新 attempt，否则报告并停止。
  - `PUBLISH_UNKNOWN` / `RECEIPT_FAILED` / `NOT_STARTED` / 回执缺失或
    身份不符 / 观测失联 / 任何无证据 → 报告 **UNRESOLVED**（附
    session）并停止：不新建 attempt、不把资源失联写成 `task_failed`、
    不自动 stop 任何实例。
- 所有 unit completed → 零实例一次性报告（不查询 sessions）。

台账没有 `RESOURCE_LOST` 状态：资源失联/无证据是**未决**而非失败，
本工具绝不代裁决，也不提供 force reset——确认需要放弃某个未决
attempt 时只能人工检查台账文件（层外操作），损坏台账一律 fail
closed。

### 分配与台账对齐窗口

`launch` 在 `colab new` **成功返回之后、任何 staging（含凭据上传）
之前**调用 CLI 传入的对账回调，为每个待执行 unit 显式
`Ledger.start_attempt`——写入即与远端 plan 对齐，此后到达 staging 的
计划必然对应同一批 attempt。校验类拒绝（上面四步链）都在分配**前**
被排除，不会在分配后重现；但台账层在分配后的对齐窗口内仍可能拒绝：
并发进程占用同一 attempt_id、台账 IO 写入失败等——这属于**对齐失败**
（不是校验缺陷，无法在分配前预知）。若该回调失败：不进入 staging
（无 worker、无凭据副本）、尽力停止刚分配的实例并明确报告；若失败
发生在台账原子替换之后（可能已提交），以重读/`mlbridge units` 为准
人工对账，本层绝不静默回滚。分配失败发生在 attempt 写入**之前**，
因此不会留下假 attempt 占用。staging 上传集合 = legacy 集合 + 显式
`batch_worker.py`（插在 job.json 之后、凭据之前）；批 JSON 即
`plan['unit_batch']`（与 `batch_worker.validate_batch` 契约一致，绑定
session/project/commit/restic_repository），legacy 不 stage、不改。

**对齐窗口自愈（controller 本地事实，绝不伪造远端终态）**：对齐回调
因**任何**异常（含 KeyboardInterrupt / OSError，而不只是 BridgeError）
失败时，launch 都先尽力停止刚分配的实例再重新抛出原始异常；只有 stop
**确认成功**后才为窗口内已写入且未决的 attempt 写控制器本地事实证据
`reconcile(task_failed, source='controller/alignment-abort', note=…)`，
note 明确：staging 未开始（无 worker、无凭据上传）、远端从未执行本批
任务、刚分配的实例已停止——这是控制器本地事实，**不是**远端任务终态，
绝不声称远端任务失败。stop 未能确认（instance_stopped=false）时**不写**
任何证据（实例状态未知），session 名保留在本地任务记录中并提示手动
`mlbridge stop`。该证据解开后 unit 即为确定失败：现有台账状态机下，
重跑必须显式 `--retry-failed`（新 attempt、新 seq），默认绝不自动重试。

### 执行中断与诚实放弃（execution_abandoned）

新台账终态：`ABANDONED` → resolution.outcome = `execution_abandoned`
（源码：`ledger.TERMINAL_STATES`）。语义严格限定为：**执行身份确认已
终止、未产生任何已发布成果、且无证据表明任务本身失败**。它不是失败
（不得假报 task_failed），也不是未决（不占用重跑解锁）——唯一作用是
诚实记录"这次尝试到此为止、可以显式重来"。

诚实放弃必须同时成立三条（`batch_control._try_abandon_interrupted`，
缺任何一条都不动台账、保持未决）：

1. **正面执行终局证据**（`{'source','detail'}`，controller 本地事实）：
   仅认两类——实例已确认停止（`colab stop` 成功返回后写入的 stopped
   标记，VM 已释放）；foreground 哨兵（cell 内同步执行已确认返回，
   worker 已退出）。观测失联（UNREACHABLE/UNKNOWN）、传输失败、
   exec 超时本身都**绝不**构成该证据。
2. **本批身份匹配的最后已知控制面证据**标明该 attempt 处于
   `RUNNING` / `NOT_STARTED`（含 batch.outcomes 缺席=未启动）：
   永未进入发布不确定区。`PUBLISH_UNKNOWN` / `RECEIPT_FAILED` 的
   attempt **永远不得**转 ABANDONED 或 task_failed。
3. **restic tags 核验确认无快照**（`restic snapshots --tag
   mlbridge-attempt-<attempt_id>`，只读本地命令）：无快照 ⇒ 不可能有
   已确认发布的 receipt/snapshot。核验失败/查到任何快照 → 拒绝放弃。

冲突与代价如下表：

| 场景 | 处置 |
| --- | --- |
| 有 receipt / 有 snapshot / tags 命中 | 拒绝放弃；人工对账 tags+回执后处理成果 |
| PUBLISH_UNKNOWN / RECEIPT_FAILED | 拒绝放弃（保持未决，不转失败） |
| 仅有观测失联 / exec 超时 / 无终局证据 | 不放弃（保持未决） |
| 终局证据 + RUNNING/NOT_STARTED + 无快照 | `execution_abandoned`，需显式 `--retry-failed` 重跑 |
| 想放弃执行但 VM 还在 | 先 `mlbridge stop`（拿到终局证据）再对账；绝不代停 |

入口：`settle_interrupted_execution(session, plan, status, proof=…)`
（内部先跑常规 `settle_terminal` 逐单元对账：成功/真实失败照常
reconcile/commit，再对仍未决单元尝试放弃）。接入点：

- `watch`（批模式）：本地任务记录存在终局证据且当前状态非终态 →
  自动中断对账（`batch_reconcile` 汇总；`all_resolved_succeeded` 为
  false 时 CLI 返回非零、不自动 stop）。
- `run --unit-batch` 恢复对账：未决 attempt 的 session 本地记录存在
  终局证据时同样走放弃窗口；放弃后与其他确定终态一样受
  `--retry-failed` 显式门控。
- `execution_abandoned` 幂等：完全一致的证据重放不动台账；
  `mlbridge units` 以 `execution_abandoned terminal=ABANDONED
  source=…` 与 `task_failed` / `completed` / UNRESOLVED 分明。

明确边界：放弃不承诺找回任何数据；放弃后有 receipt/snapshot 的一律
拒绝（需人工对账）；放弃也不会免费——restic tags 查询会联网访问对象
存储（只读）。放弃**不消除** PUBLISH_UNKNOWN 的对账责任：该状态永远
保持未决直到人工处理。

### 交付模式（opt-in）：`--execution-mode detached|foreground`

只影响交付与生命周期，不影响批契约、批次身份（不进 batch_id）与台账
语义。缺省 `detached`：与本节上述所有行为逐字节一致。`foreground`
（仅 `--unit-batch`）：worker 在 Colab cell 内同步运行到批结束，平台的
观测画面显示真实持续执行；代价是**本地 exec 连接断开/超时可能随 cell
终止执行**。本地 exec 带显式长超时（按远端各阶段硬超时之和 + 固定余量
推导，`colab_runner.foreground_exec_timeout`），超时/断开都只作为
**观测中断**处理：记 `observation_lost`，不判远端失败，交 watch/对账。
恢复对账与 ABANDONED 规则见上节。细节与限制见 docs/UNIT_EXECUTION.md。

### 对齐失败 / 未决 attempt：人工对账与解锁（层外操作）

台账是控制器的本地事实记录；本工具从不提供 force reset。对齐失败或
未决 attempt 的人工解锁步骤（按顺序）：

1. **确认实例事实**：`mlbridge status <session>` / `colab sessions`。
   UNREACHABLE 只表示 VM/传输不可达，绝不等于任务失败（status 会一并
   显示消失前最后的 `last_known`）。实例仍活跃且想让任务继续 →
   `mlbridge watch <session>`（批模式自动从本地任务记录读取批计划逐
   单元对账）；确认任务未跑或已死 → `mlbridge stop <session>` 手动
   释放，避免持续计费。
2. **读台账**：`mlbridge units <project> --json`，区分三类：
   - `state=completed`：无需处理。
   - attempt `resolution.source == controller/alignment-abort`：控制器
     本地事实（该 attempt 的 staging 从未开始、实例已停止），**信任
     台账**，无需人工编辑；重跑该 unit 必须显式 `--retry-failed`（新
     attempt、新 seq）。它不属于“远端失败”，不要据此怀疑数据。
   - `resolution` 缺失（UNRESOLVED，含 stop 未确认、无证据、
     PUBLISH_UNKNOWN 等）→ 第 3 步。
3. **未决 attempt 的人工对账**：用 `mlbridge status <session>` 与
   restic tags（`restic snapshots --tag mlbridge-unit-<uid>` 等：
   mlbridge-project-*/mlbridge-batch-*/mlbridge-unit-*/mlbridge-attempt-*）
   人工核对远端真实发生过什么。仅在**确认**后：
   - 任务确未执行/已失败且不打算寻找成果 → 人工编辑台账文件
     `$XDG_STATE_HOME/mlbridge/units/<project>.json`（0600；先备份）。
     把该未决 attempt 的 `resolution` 写成诚实的人工证据，例如任务
     失败用 `{"outcome": "task_failed", "source":
     "controller/manual-abandon", "terminal_state": "FAILED",
     "note": "人工确认…（依据）"}`；仅"放弃"（执行已终止、无发布、
     无失败证据）用 `{"outcome": "execution_abandoned",
     "source": "…", "terminal_state": "ABANDONED", "note":
     "放弃依据…"}`（必须非空 note；禁止给有 receipt/snapshot 的
     attempt 写 abandoned）。两者效果一致：需显式 `--retry-failed`
     才能开新 attempt。编辑必须保持全部状态机不变量（attempt_id 跨
     unit 唯一、seq 从 1 连续、旧 attempt 全部 task_failed 或
     execution_abandoned、未决至多一个且最新）；非法编辑会在下次读入
     时被 `CorruptLedgerError` fail-closed 拒绝。随后
     `mlbridge units` 重新校验，再用 `--retry-failed` 重跑。
   - 确认成果已发布 → 人工核对回执各字段后按 `commit_result` 契约补
     全该 attempt 的 `result`（snapshot/manifest_sha256 等必须真实
     来自该 attempt 的发布产物）；不确定时保持未决，绝不编造。
4. **controller/alignment-abort 证据本身有误时**（例如 stop 回执丢失
   导致误报停止）：人工核对本地任务记录、`colab sessions` 与步骤 1
   的事实，确认后同样按第 3 步人工改写为准确事实；未确认前保持原样。

任何步骤都不得伪造远端终态或把“资源失联”写成 task_failed；无法
判定时宁可保持未决。

## 明确不承诺的行为

- 不承诺计算 **exactly-once**；本库独立验证远程平台/快照内容的责任
  归属不变（调用方做外部回执核对、快照内容校验、外部仓库核验）。
- 不提供不受约束的 force retry / force 解锁；损坏台账必须人工调查。
- 批终态对账与 stop 规则（`watch`，含独立 `mlbridge watch` 自动从
  本地任务记录读取批计划）：批整体终态（SUCCEEDED/FAILED）只是控制面
  事实，绝不据此把全部 unit 批量写成终态；逐 unit 对账规则与启动前
  相同（首先读取本机已写下的终态状态，再读取契约回执）。只有批内
  **全部** unit 已为确定成功且 auto_stop 时才释放实例；任何未决
  （PUBLISH_UNKNOWN / RECEIPT_FAILED / NOT_STARTED / 无证据）都保留
  实例、不自动 stop，绝不谎报成功或把资源丢失写成任务失败。回执
  缺字段/身份不匹配/匿名回执（download 不到、非契约形态）→ 拒绝
  且绝不改台账。
- **worker 自报 ≠ 独立验收**：本切片的 reconcile/commit 全部来自
  远端 worker 的状态与回执（台账 evidence.note 与 source 显式记录
  这一点），不宣称本地独立验收了快照内容、manifest 与 restic tags ；
  独立验收是下一步工作。
- **PUBLISH_UNKNOWN 需人工对账**：响应丢失≠必然无快照；请用 restic
  tags（mlbridge-project-*/mlbridge-batch-*/mlbridge-unit-*/
  mlbridge-attempt-*）人工核对后处理；本工具不自动重传/重算。
- 不承诺计算 **exactly-once**：批模式同样只保证本地台账记录不重复
  提交与跨进程并发安全（flock + 原子替换）。
- 未接入（不承诺）：PUBLISH_UNKNOWN 的自动对账/重传；批内容变更后
  的自动重放策略（新定义即新 batch_id，由用户显式提交）；本地独立
  产物验收（快照内容/manifest/tags 核验）；环境采集缓存；远端身份
  核验；`--detach` 后自动定时对账（需用户手动 `mlbridge watch`）；
  放弃自动化的其他证据源（远端 worker 进程退出观察等）；
  PUBLISH_UNKNOWN/RECEIPT_FAILED 存在时永不自动放弃，其解开仍需
  人工 tags 对账。
- 只读查询 `mlbridge units <project> [--json]`：每 unit 状态/
  attempt/阶段/observation/成果摘要；未决 attempt 显式标注
  UNRESOLVED；不做任何变更，不提供 force reset。
