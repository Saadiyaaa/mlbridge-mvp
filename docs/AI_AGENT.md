# AI Agent 操作协议与交互规范 (Agent Operational Protocol)

本文档面向接入并驱动 MLBridge 的终端自动化 Agent（如 Claude Code、Cursor、终端 Agent），规范 Agent 在调度与管理机器学习计算任务时的行为约束、执行时序与数据契约。

---

## 1. 核心行为准则与安全边界

1. **零代码污染准则 (Non-Invasive Rule)**：
   - 严禁为了运行或适配 MLBridge 修改用户研究项目的业务源码、配置文件、`requirements.txt`、`README.md` 或 `.gitignore`。
   - 所有配置操作仅允许在 `~/.config/mlbridge/` 路径下执行。
2. **代码可复现准则 (Commit-First Rule)**：
   - 调度任务前必须保证本地 Git 工作树干净，且目标提交已推送（`git push`）至远程代码仓。
   - 严禁基于未提交或本地独有的未推送代码提交远程计算。
3. **数据确定性准则 (Immutable Snapshot Rule)**：
   - 调度任务输入必须采用显式 64 位不可变 Snapshot ID。
   - 严禁使用任何形式的动态标签（如 `latest`）作为依赖输入。
4. **实例关停安全规则 (Teardown Guardrail)**：
   - 仅在任务明确报告 `SUCCEEDED` 且已生成新产物 Snapshot ID 时，允许关停 Colab 实例。
   - 状态为 `FAILED` 时必须保留实例现场，供调试与日志排查。
   - 遇到 `UNREACHABLE` 状态时，属于网络通信中断，严禁执行 `stop`，必须提示重连或保持等待。
5. **凭据安全准则 (Credential Redaction Rule)**：
   - 严禁在对话上下文、终端日志或状态报告中输出 `rclone.conf` 内容、OAuth Token 或 restic 仓库密码。

---

## 2. 标准作业时序 (Standard Workflow)

```bash
# 1. 环境与配置预检
mlbridge doctor --project <PROJECT_NAME>

# 2. 查询云端已有快照
mlbridge artifacts <PROJECT_NAME>

# 3. 发布未上云的输入数据（若需）
mlbridge publish <PROJECT_NAME> --label <LABEL> --path <RELATIVE_DATA_PATH>

# 4. 提交远程任务至计算节点
mlbridge run <PROJECT_NAME> --job <JOB_NAME> --input <SNAPSHOT_ID> --gpu <GPU_TYPE>

# 5. 后台轮询与状态追踪（支持分离执行）
mlbridge status <SESSION_ID>
mlbridge logs <SESSION_ID> --lines 100
mlbridge watch <SESSION_ID>
```

---

## 3. 结构化输出规范与状态报告

Agent 在向用户汇报任务执行结果时，输出报告必须包含以下核心要素：

| 汇报字段 | 描述 | 示例 |
| :--- | :--- | :--- |
| `project` | 目标项目名称 | `image-project` |
| `job` | 执行的任务定义名 | `upstream` |
| `session` | 远端实例会话 ID | `mlb-image-project-f38b192a` |
| `commit` | 绑定的代码版本 Commit Hash | `9d3fa81029c782138a0f98d78120491823091248` |
| `inputs` | 依赖的输入快照 ID 列表 | `["a48f93bc21e78491..."]` |
| `phase` | 最终运行状态 | `SUCCEEDED` / `FAILED` / `UNKNOWN` |
| `output_snapshot` | 成功时生成的产物快照 ID | `e182390abff71829...` |
| `instance_stopped` | 计算实例是否已释放停止 | `true` / `false` |

---

## 4. 异常处理与诊断协议

* **诊断连接**：若任务失败，建议用户使用官方命令连接实例现场：
  ```bash
  colab ssh -s <SESSION_ID>
  ```
  *严禁附加 `--rm` 旗标，以防 SSH 会话关闭时误删实例。*
* **日志提取**：优先调用 `mlbridge logs <SESSION_ID> --lines 200` 提取报错关键日志堆栈，不要下载全量日志文件。
* **确认关机**：在排查完毕后，由用户确认再调用 `mlbridge stop <SESSION_ID>` 释放实例，防止持续计费。
