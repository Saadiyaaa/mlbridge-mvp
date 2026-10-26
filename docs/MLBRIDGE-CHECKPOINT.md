# MLBridge per-key checkpoint protocol v1（研究 driver 接口）

目的：五人同会话场景下，每个人完成时在**已运行的** Colab worker 环境里以
自己的 checkpoint key 独立触发一次可恢复发布（一次 restic backup + 原子
response），失败或中断永远留证、不伪装成功、不自动重试、不停止 VM。
本协议只提供身份与存储完整性；科学验收（pins 恢复比对）不在 MLBridge 范围。

## 环境契约

- worker 在 VM 上的固定路径：`/content/mlbridge_worker.py`
  （由 `colab_runner.launch` 上传，stdlib-only）。
- worker `main()` 在远端执行 `plan.command` 前显式导出：
  - `MLBRIDGE_CHECKPOINT_PROTOCOL=v1`
  - `MLBRIDGE_CHECKPOINT_WORKER=/content/mlbridge_worker.py`（`Path(__file__).resolve()`）
- checkpoint 接口只允许在该 command 环境里被 driver 调用；环境缺失或不匹配
  fail-closed（exit 2）。`MLBRIDGE_WORKER_BASE` 仅离线测试用，真实 VM 不设置。
- restic/rclone 凭据由 command 环境继承（`RESTIC_PASSWORD_FILE` /
  `RCLONE_CONFIG`），key、请求与日志中绝不出现凭据或 repo 私密细节；
  不重复上传任何数据 inputs。

## driver CLI（VM 内）

```
python /content/mlbridge_worker.py checkpoint --request REQUEST.json [--timeout SECONDS]
```

`--timeout`：1..43200 秒，默认 1800（单次 backup 的有界时限）。

### Request schema（v1，JSON object，未知字段拒绝）

```json
{
  "protocol": "v1",
  "project": "mypure",
  "session": "<plan 的 session，必须与 mlbridge_job.json 一致>",
  "commit": "<40 位十六进制，必须与 plan.commit 一致>",
  "key": "<研究 rowkey：1..256 字符，原样保留，无 NUL/换行/首尾空白>",
  "roots": ["<授权 plan.outputs 完整根或其子路径>"],   // 研究流程 = 3 个已 postverify 的人物输出根
  "metadata": "<授权 session root 下的不可变 metadata-intent 文件（普通文件，≤4MiB，非空）>",
  "response": "<原子 response 的相对路径（尚不存在）>",
  "provenance": {"backend": "...", "host": "...", "gpu": "..."}  // caller 显式提供；MLBridge 原样透传，不写死 GPU 型号
}
```

校验（任何失败 = exit 2，发生在任何 backup 之前）：
- project/session/commit 与 plan 严格一致；
- 所有路径为工作区内相对路径：拒绝绝对、`..`、`.git`、软链接组件（逐级检查，
  拒绝路径别名）、`./` 与 `//` 归一化；
- `roots` 与 `metadata`、`response` 都必须是授权 `plan.outputs` 的完整根或其
  子路径；不许任意备份 inputs/秘密/整个 workspace；
- `metadata` 必须是已存在的普通文件（非软链），内容 SHA-256 计为指纹；
- `roots` 之间不得互相包含；`response` 不得与任一 root 包含或重叠，也不得与 metadata 相同。
- `response` 不得已存在（已有记录即拒绝重发，不代表产物已验收）。

### 语义（一个 key 只有一次机会）

1. 同 key 已有 marker 或已有 response ⇒ exit 3 显式拒绝，**绝不重复 backup**。
   未知结果先按 tag 重发现并核实，不能换 key 绕过防重；新执行尝试需中央明确授权。
2. 以 `O_CREAT|O_EXCL` 原子占位 PENDING marker（`/content/mlbridge_checkpoint/<key_sha256>.json`）；同 key 并发请求只有一个能开始 backup。中断留下不完整 marker 也保持阻塞，不自动删除重试。
3. **恰好一次** restic backup（工作区内相对路径调用）：
   ```
   restic -r <repo> backup --json
     --tag mlbridge-project-<project>
     --tag mlbridge-session-checkpoint-<session>
     --tag mlbridge-checkpoint-v1
     --tag mlbridge-checkpoint-key-<sha256(key)>
     --tag mlbridge-checkpoint-meta-<sha256(metadata bytes)>
     <roots...> <metadata>
   ```
   （backup 的多个 --tag 只是给同一新快照打多枚标签，与 snapshots 的
   过滤语义不同，可分写。）
4. 从 JSON 摘要**严格**取 `message_type=="summary"` 的 `snapshot_id`；
   必须是 full 64 hex 字符串（类型/格式非法一律不猜）。
5. backup 之后原子写 response（tmp + fsync + `os.replace`，mode 0644，
   相对工作区 `<response>` 路径），因此 response 里可以有自引用 snapshot id。

### 退出码

| code | 含义 | response |
|---|---|---|
| 0 | SUCCEEDED（backup 成功且 id 合法） | `{'phase':'SUCCEEDED','snapshot_id':<full64>,...}` |
| 1 | 未分类操作系统错误；不能推断 backup 是否发生 | response 可能缺失；保留 marker，先查日志和 tags，禁止盲重发 |
| 2 | 拒绝（协议环境/请求校验，无 backup 发生） | 无 |
| 3 | 同 key 已有记录，拒绝重复 backup | 无 |
| 4 | backup 无法启动，或硬错误且无已建快照的摘要证据 | `{'phase':'FAILED','error','restic_log_excerpt'}`（有界、已脱敏；仍不自动重试） |
| 5 | AMBIGUOUS：restic 退出码 3（部分来源读取失败但可能已建快照）、其他非零码伴合法快照摘要、超时、摘要缺失/冲突或 id 非法、response 落盘失败 | `{'phase':'AMBIGUOUS',...}`（若可写）；合法候选 ID 仅记为 `candidate_snapshot_id`，不是成功产物 |

超时/AMBIGUOUS 时快照**可能**已经建立：绝不二次 upload；由 host 侧按
tag+intent 重发现。已发生 backup 而 response 缺失的窗口同理由 tag 重发现。

Response 字段（SUCCEEDED/FAILED/AMBIGUOUS 共有）：
`protocol, phase, exit_code, project, session, commit, checkpoint_key,
checkpoint_key_sha256, requested_roots, metadata{path,sha256,bytes},
provenance, restic_exit_code, written`。

身份与 claim 契约：`checkpoint_key` 原样保留；`checkpoint_key_sha256 =
sha256(utf-8(key))` 是跨主机对照键。本 VM 内同 key marker 先写者胜；
跨主机 claim 表由中央控制面（主会话后续阶段）用该 hash 对照，MLBridge 不做
也不猜测全局唯一性。provenance（含 backend/host/GPU）由 caller 提供、原样
记录，不做任何型号判断。

VM 本地留证：`/content/mlbridge_checkpoint.log`（JSONL，有界裁剪 256KB→64KB；
只含事件/时间/ key hash/状态/excerpt，不含 repo URL 或凭据）；
marker 尽力保留最终状态副本；response 写入失败时记为 AMBIGUOUS，若 marker 所在文件系统也不可写则可能保留 PENDING。日志为辅助诊断，不替代 tag 与不可变 metadata 的恢复核验。

## host 侧只读发现

```
mlbridge checkpoints <SESSION> --project <name> [--key <key>] [--json]
```

- 查询：`restic snapshots --json --tag "mlbridge-project-<P>,
  mlbridge-session-checkpoint-<S>,mlbridge-checkpoint-v1
  [,mlbridge-checkpoint-key-<sha256(key)>]"`（单一 --tag 内逗号分隔才是
  AND；多个 --tag flag 之间是 OR——restic 0.18.0 help + 实证留证，分写会
  把其他 session/key 的快照查回来）。若 backend 仍返回 scope 之外或
  key 归属与查询不符的快照，fail-closed 报错，绝不静默丢弃或当作命中。
- 输出每 key：`snapshot_id`（full64 强校验）、time、tags、metadata 指纹
  （协议 v1 快照必须恰好一个 full64 的 meta tag，缺失/重复/非 hex 均拒）、
  `state`、`pins_verification`（恒为"恢复后待验收"，发现 ≠ 验收）。
- 同 key 多快照 ⇒ `state='conflict'` 且全列出，exit code 3；绝不任挑最新
  当 accepted。畸形 JSON/非法 id ⇒ exit 2（fail-closed）。
- `--key` 时附带 `queried`（exact key、its hash、found/state）供中央跨主机
  原子 claim 对照；MLBridge 不维护 claim 表本身。

## launch/诊断留证（host 侧）

`colab_runner` 在 launch 边界（alloc/stage/exec/starter/probe）把事件追加到
`$XDG_STATE_HOME/mlbridge/diagnostics/launch-<session>.jsonl`（JSONL，
>512KB 时裁剪至最近 64KB）：事件名、rc、耗时、有界 stdout/stderr 尾部、
错误消息、starter 解析出的 `remote_pid`（并写入本地 job 记录）。
规则不变：starter 只执行一次（`exec_starter_once`，不盲重试）；state 缺失只
UNREACHABLE 不断言 VM 死亡；dispatch 的远端 stdout/stderr 不经本接口落盘。
