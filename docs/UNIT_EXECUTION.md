# 远端工作单元顺序执行（unit_batch v2）——当前接口、限制与验收责任

本文档描述远端 worker 的 `unit_batch` 批模式当前能力：所有权、顺序执行、
逐单元验证/发布/回执与终态写法。先读边界：

- **交付模式（execution_mode v1，opt-in）**：`run --unit-batch` 可加
  `--execution-mode detached|foreground`（缺省 detached）。二者只改
  "worker 如何被启动/与本地连接的生命周期关系"，不改变批契约、逐单元
  执行/发布/回执与台账语义；**不参与 batch_id 身份**（跨模式可继续
  同一批次的恢复对账）。详见下方"交付模式与生命周期成本"。
- **本地接入（opt-in）**：`mlbridge run --unit-batch FILE` 会经由
  launch 显式 stage 本文件（作为 `/content/batch_worker.py`，与
  `mlbridge_job.json`、`mlbridge_worker.py` 同目录，且在凭据文件之前）
  以及 `plan['unit_batch']`（随 job.json 一起上传，契约与
  `validate_batch` 一致）。校验/对账规则见 `docs/RECOVERY.md`
  “批模式（opt-in）”一节。legacy `run`（无 `--unit-batch`）不 stage
  本文件、不改上传集合；未被 stage 时任何含 `unit_batch` 键的 plan
  到达远端 worker 仍会 fail closed（清晰报错、无任何副作用）。

## 文件与角色

- `src/mlbridge/batch_worker.py`：**standalone 远端执行器**（纯 stdlib，
  不 import mlbridge 包、不需要 pip）；由 `remote_worker.py` 的 opt-in
  分支按需加载 sibling。含批契约校验（`validate_batch`，纯函数）、
  同批重复执行拒绝（claim）、逐 unit 执行与发布回执。跨进程互斥不在此
  模块内实现（见所有权）。
- `src/mlbridge/remote_worker.py`：入口与**唯一所有权原语**。plan 中
  `unit_batch` **键存在**（含显式 null）即走批模式校验——显式 null 是
  非法批计划，在任何 clone/restore/setup 副作用之前被拒绝；无该键的
  legacy plan 使用单任务执行路径。
- `src/mlbridge/protocol.py`：unit 身份的权威定义。batch_worker 在远端
  **本地镜像**同一条规范化/校验规则并独立重算 canonical SHA256，绝不
  信任提交方给出的哈希；镜像一致性由
  `test_batch_identity_matches_protocol` 保证。

## 所有权（BASE 执行锁）

- `/content/mlbridge_execution.lock`：唯一跨进程互斥原语，legacy 与批
  模式共用。`main()` 先**非阻塞**独占获取；拿不到立刻拒绝（竞争 worker
  未写任何 STATE、未删任何凭据、未 clone），绝不阻塞等待造成旧
  cleanup 与新任务的竞态。
- 锁全程持有：心跳启动/停止/join、读计划、批校验与 claim、凭据
  chmod、clone/restore/setup、执行与 STATE 写入、异常记录、凭据清理都
  在所有权内；执行锁在 finally 中**最后释放**（owner 才清理凭据）。
- 非 owner 不得写共享 STATE、不得删凭据。进程内 `_WRITE_LOCK` 只串行
  化同进程多线程（主线程终态 vs 心跳），跨进程安全由执行锁提供。
- 心跳线程可停：`stop_event` 置位后有界 join，main 正常或异常退出后
  都不残留 daemon 心跳在锁外继续写共享 STATE。

## 批契约 v2（plan['unit_batch']）


```jsonc
{
  "unit_batch": {
    "schema_version": 2,
    "batch_id": "bt1",                       // ident：1–63 位 [A-Za-z0-9_-]
    // 可选：manifest_limits{max_files,max_bytes,timeout_seconds}
    // 可选：publish_timeout_seconds
    "units": [                               // 有序；按此顺序逐单元执行
      {
        "unit_id": "<64hex>",                // 必须等于 spec canonical JSON SHA256
        "attempt_id": "att-one",             // ident；全批唯一
        "execution": { "timeout_seconds": 600 }, // 必填；正有限数；禁止无限等待
        "spec": { ... WorkUnit.to_dict 的合法 canonical 定义 ... }
      }
      // 每个 unit 的 spec：session/attempt 等执行期信息不属于协议，
      // 未知字段直接拒绝；params 是身份字段（      // 普通键，非协议字段、不做任何特殊处理）
    ]
  }
}
```

plan 层绑定（沿用现有 plan 字段，全部必填且被严格校验）：
`session`、`project`、`job_id`、`repository`、`commit`、
`restic_repository`、`inputs`、`setup`。绑定约束：

- 每个 unit 的 `spec.project/repository(规范化)/commit` 必须与 plan 一致。
- 每个 unit 的 `spec.environment['setup_sha256']` 必须等于 canonical
  JSON SHA256(plan.setup)。environment 其余键属于身份，自担其义。
- 全部 unit 的 `environment` / `inputs` 必须一致；`inputs` 名称排序后的
  snapshot 列表必须与 `plan.inputs` 顺序**完全一致**（plan.inputs 的顺序
  就是 restore 顺序，禁止快照覆盖顺序隐含改变语义）。
- 所有 unit 的 outputs 非空；unit 间输出树互不重叠（路径组件级判断）；
  任何输出路径不得落入/包含批元数据目录 `.mlbridge-metadata`。
- **参数不被注入**：本 worker 从不改写 command argv、不写入科学
  参数或种子。`spec.params` 只是身份字段（与其它
  科学参数一样只是 `params` 里的普通键）；研究程序如何读取它们由
  command 自己负责。worker 不声称研究程序真的使用了这些参数。

`validate_batch` 是纯校验：任何 schema/身份/绑定/保留路径问题（含
`unit_batch` 显式 null）在任何副作用（clone/restore/setup/凭据 chmod）
之前 fail closed。

## validator v1

只支持 `{'kind':'command','command':[非空 argv...],
'timeout_seconds':正有限数}`：以工作区为 cwd 运行，退出码 0 即通过；
非 0 或超时 → 单元失败，**不 backup**。metric 阈值等其他 kind 未实现、
直接拒绝（fail closed）。不做 shell 拼接。

## 远端执行流程

1. `validate_batch(plan)`（纯校验，副作用前；校验发生在执行锁内）。
2. `claim_batch`：若该 batch 已有运行/终态记录
   （`/content/mlbridge_batch_state_<batch_id>.json`），**拒绝重复执行**
   ——不是自动清理旧 clone 重跑，也不是自动 resume；请人工对账后确认
   安全再手动移除状态文件。claim 初始把全部 unit 记为
   `NOT_STARTED`，**绝不预标 RUNNING**；实际启动某 unit 时才把该 unit
   记为 RUNNING 并写入 `current_unit`。
3. 一次 clone/restore/setup（legacy 与批模式共用同一段唯一执行，
   无第二批 clone）。
4. 逐 unit（顺序）：
   `_precheck_outputs`（outputs 不得已存在——防污染已发布成果；祖先链
   不得有 symlink；解析后不得越界）→ **command**（真实子进程，
   `execution.timeout_seconds` 超时：SIGTERM 自建进程组→按**组活性**
   有界宽限→SIGKILL→KILL 后仍有界确认组内无存活非-zombie 成员；绝不
   按名字杀进程）→ **validator**（同规则）→ **输出完整性检查**（输出
   树内拒绝 symlink/.git/fifo/socket/device；预算按条目（文件+目录，
   含目录项遍历与空目录洪泛）、**实际读到的字节**（哈希逐块累计）与
   时间上限（每个目录遍历、每个哈希块读取前检查）计，缺省 10000 /
   1 GiB / 600s；
   单文件声明输出与多 outputs 共享同一套预算，超限拒绝而非静默漏项）
   → 相对路径 → SHA256 **manifest**（canonical JSON，写入
   `<ws>/.mlbridge-metadata/<batch_id>/<unit_id>/manifest.json`；写入前
   对整链 lstat 重验，祖先存在则必须是普通目录——拒绝 symlink/普通
   文件/特殊文件，整链解析后仍限工作区内；缺失祖先 mkdir(0o700) 并
   对父目录 fsync；manifest 文件 fsync 后再对 unit 目录 fsync）
   → **restic backup**（一次调用：unit outputs + manifest，全部相对
   工作区路径；tags：`mlbridge-project-<project>`、
   `mlbridge-artifact-<job_id>`、`mlbridge-batch-<batch_id>`、
   `mlbridge-unit-<unit_id>`、`mlbridge-attempt-<attempt_id>`；restic
   stdout/stderr 全部流入临时文件，不驻内存，不充满 PIPE；错误诊断
   只取有界的 stderr 尾部（≤2000 字节），不含环境变量/凭据内容）
   → 校验 **64hex snapshot_id** → 原子写入 **unit receipt** → batch
   进度补全（未启动 unit 保持 NOT_STARTED）→ 下一 unit。

进程组契约：`start_new_session` 保证新组 leader 是被启动命令本身，
pgid == 子进程 pid，启动时捕获；leader 先退出后组内孙进程仍属同一
组、仍可 `killpg` 寻址，收尾继续有效。**正常退出也必须确认组内无
存活（非-zombie）成员**，否则终止该组并按失败处理（防止
validator/hash/backup 期间遗留进程继续写输出文件）。可行约束：可信
command 不得 daemonize——手动 `setsid` 逃离进程组的孙进程超出本执行
器属地保护，不承诺通用沙箱。

快照相对路径契约：快照内是（a）各 unit outputs 以其工作区相对路径，
（b）`.mlbridge-metadata/<batch_id>/<unit_id>/manifest.json`。该元数据
目录是 Colab 临时 clone 内的运行时目录，不与输入/outputs 冲突（批校验
强制互斥）、不含凭据或 .git；本地研究项目不受影响。manifest 本身随之
归档；最终 snapshot_id 回执在 BASE（见下）。

## 失败语义（不把批 reconcile 成单一失败）

逐单元 outcome（state 文件与心跳状态里的 `batch.outcomes`）：

- `SUCCEEDED`：command+validator 通过、发布拿到确认 64hex snapshot、
  receipt 写入成功。已发布单元的回执/现场**保持不变**。
- `TASK_FAILED`：真实失败（command/validator/输出检查/元数据链/
  precheck/组内遗留残留），附 stage 与 detail。validator 失败不
  publish。
- `PUBLISH_UNKNOWN`：restic backup 超时、无法启动、泄留进程被强制
  收尾或拿不到合法 64hex snapshot_id。**不自动重算/不重新上传**；tags
  已绑定 batch/unit/attempt 供对账。响应丢失≠必然无快照：一律不声称
  已发布，也不会当作失败单元重新重跑。
- `RECEIPT_FAILED`：发布已确认但回执写入失败。绝不吞掉、绝不此场景下
  声称已提交；outcome 中保留 snapshot_id。
- `NOT_STARTED`：单元失败后剩余单元**不启动、不虚构失败**。批终态可写
  `FAILED`，但必须附带明确逐单元 outcome。

终态写顺序（`execute_batch` 固定）：① 持久批记录
`mlbridge_batch_state_<batch_id>.json` 落终态（含逐单元 outcome 的权威
对账依据）→ ② 共享 STATE（`mlbridge_state.json`）落终态。两处失败都
显式上抛并在 STATE 标注 `control_warning`（控制面持久化不确定；逐单元
receipt 与快照是该批权威依据；**未持久确认前不宣称整批可安全停机**），
绝不吞掉、也绝不把已写成的 SUCCEEDED 改写成 FAILED 或反之。

兜底终态（remote_worker `_write_failed_if_open`）受**身份门控**：

- 身份已确认（plan 合法 job_id）且 STATE 已有终态属于同一 job/batch
  ⇒ 保留（批的逐单元 outcome 不被兜底覆盖）；
- 身份无法确认（计划不可读 / job_id 非法形态）⇒ 绝不改写已有 STATE：
  那份终态属于别的 job，不可冒充本 job 的结果（失败由异常与日志报
  告）；仅当 STATE 不存在时写入不含身份的最小启动失败记录；
- 身份已确认但残留终态属于其他 job ⇒ 用 fresh 覆盖为本次失败，不继
  承残留 `snapshot_id` 等字段。

回执 `receipt`（`/content/mlbridge_receipts/<batch_id>/<unit_id>.json`，
0600、原子替换 + fsync）字段：`schema_version / batch_id / unit_id /
attempt_id / session / namespace(=restic_repository) / snapshot_id /
manifest_sha256(manifest 文件字节的 SHA256) / manifest_path(快照内相对路径) /
validator_spec_sha256(= canonical JSON SHA256(spec.validator)) /
validator_result{kind,status:"passed"} / phase:"published" / finished`。
manifest 载荷：`schema_version / batch_id / unit_id / attempt_id /
session / files{相对路径: SHA256}`。

明确不是：
- 回执是 **worker 自报**，不等于本地独立产物验收；快照内容、
  manifest 与 tags 的独立核验（本地控制端）尚未实现，当前只有控制面
  的契约/身份校验（`docs/RECOVERY.md` “批模式（opt-in)”）。
- 当前 engine 只做首次执行：无自动重试，无盲目 resume，无
  PUBLISH_UNKNOWN 自动对账/重传。

## 状态/进度

`/content/mlbridge_state.json` 是本地 watch 的视图；`phase` 为
PREPARING/RESTORING/RUNNING/PUBLISHING/终态，`batch` 字段为：
`batch_id / session / units_total / current_unit / outcomes`。
`/content/mlbridge_batch_state_<batch_id>.json` 是该批的持久进度记录
（claim 即创建；逐单元 outcome 补齐全量 unit、未启动者保持
NOT_STARTED；crash 后仍保留，供同批 dup 拒绝与人工对账）。
心跳只刷新 `heartbeat` 时间戳字段，绝不修改 phase 与终态。

## 交付模式与生命周期成本（execution_mode v1）

`plan['execution_mode']`（plan 级键，可选）：缺省/`'detached'` = 分离执行；
`'foreground'` = cell 内同步执行（**仅批模式允许**；任何 legacy plan 带
该值、或非法值，都会在任何副作用（含心跳）之前被
`remote_worker.validate_execution_mode` 拒绝）。detached 时该键不写入
job 载荷。

- **detached（缺省）**：starter 在 cell 内 `Popen(start_new_session)`
  分离启动 worker 后 cell 立即结束；exec 秒级返回，worker 独立于
  SSH/exec 生命周期。代价：Notebook 空闲、平台可能**回收实例**（本
  工具不承诺不回收）。
- **foreground（仅批模式）**：starter 在 cell 内 `subprocess.run`
  同步启动 worker，该次 exec 持续到批结束（或 cell 被终止），平台
  观测到真实持续执行。代价：**本地连接断开/超时时 cell 进程终止可能
  随之终止 worker**（worker 不脱离 cell 进程组；未实现自举逃逸，也
  不建议用 `setsid` 逃离属地保护）。本地 exec 的显式长超时由
  `colab_runner.foreground_exec_timeout` 推导：逐 unit 的
  execution/validator 硬超时 + manifest 预算超时 + publish_timeout
  之和 + 固定余量（clone/apt/restore/setup 无远端硬超时）。**超时只
  是观测中断**：本地记录 `observation_lost`，绝不判远端失败、绝不
  stop、不删凭据；是否真中断由 watch/对账依据远端事实（state/回执/
  restic tags）判定。

两种模式都：不承诺平台不回收、不承诺执行成功、不承诺 exactly-once；
任何"保证不中断/不会被回收/平台不会回收"的宣称都超出本实现。

输出可见性（foreground）：worker 的 stdout/stderr 全部写入
`/content/mlbridge_run.log`（`mlbridge logs` 拉取有界尾部），status/
receipt 周期性落盘供本地 watch 独立读取；cell 本身只打印两行（开始
一行 + `MLBRIDGE_FOREGROUND_DONE worker_exit=… elapsed=…` 哨兵行）。
哨兵出现是**正面证据**（cell 内同步执行已返回 ⇒ worker 已退出），由
launch 记入本地任务记录 `foreground_exit`，供后续中断对账使用；哨兵
缺失则什么都证明不了。不把整批 stdout 塞进内存/cell 输出，也不用空
循环/伪造活动维持连接或规避配额。

## 限制 / 集成状态

- **已接入（opt-in）**：`mlbridge run --unit-batch FILE`（控制面校验/
  恢复对账/`units` 查询见 `docs/RECOVERY.md`）；launch 显式 stage
  本文件；`watch`（含独立 `mlbridge watch`，读本地任务记录中的
  `unit_batch_plan`）在批终态读逐单元 outcome 与回执并 reconcile/
  commit_result 到本地 ledger——但回执是 worker 自报，**不构成本地
  独立产物验收**；stop 只在批内全部单元确定成功且 auto_stop 时发生，
  未决一律不自动 stop。执行中断（实例确认停止 / foreground 哨兵 +
  RUNNING/NOT_STARTED + restic tags 无快照）时按 `execution_abandoned`
  （诚实放弃）处理，规则与证据链见 `docs/RECOVERY.md`“执行中断与
  诚实放弃”；`--execution-mode foreground` 与观测中断语义见本文
  “交付模式与生命周期成本”。
- `params` 不注入（见上）；"研究程序用了这些参数"由本地验收方
  自行核对——worker 不提供该保证。
- 未实现（不承诺）：本地独立产物验收（快照内容/manifest/tags 核验
  只由调用方人工完成）；环境采集缓存；PUBLISH_UNKNOWN 自动对账/重传
  （需人工 tags 对账）；中断放弃的其他证据源（远端 worker 进程退出
  观察、本地独立产物核验后自动放弃等）；PUBLISH_UNKNOWN/RECEIPT_
  FAILED 存在时永不自动放弃或转失败；claim
  后、准备前失败的批会在 BASE 留下 `running` 状态记录，同批重跑会被
  dup 拒绝，需人工移除状态文件。batch 重放语义（内容变更 → 新
  batch_id）由用户显式提交新批定义。
- 输出完整性检查与 manifest 写入存在 TOCTOU 窗口（检查与使用之间的
  并发修改不在本层防护内，链在写入前会重验一次）；restic 自身的完整
  性与快照内容核验由验收责任方承担。
- restic 错误摘要只取有界 stderr 尾部，不做语义解析，也不包含环境
  变量或凭据文件内容；凭据只存在于 0600 文件，绝不出现在 argv/日志。
- 运行元数据目录名 `.mlbridge-metadata` 为保留前缀；未来如引入其他
  保留路径需同步扩展互斥校验。
- 心跳间隔（300s）与 legacy 相同；批模式下 phase 仍是
  RUNNING/PUBLISHING，watch 轮询能正常观察到终态。
