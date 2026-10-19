# MLBridge 系统架构与技术设计规范

## 1. 系统定位与设计边界

### 1.1 核心目标
MLBridge 通过项目外配置，协调 GitHub 代码仓库、restic/rclone 数据快照存储和 Google Colab 算力节点。

### 1.2 系统边界与非目标 (Scope & Non-Goals)
- **代码零修改**：研究项目仅作为 `workspace` 路径和原生 `command` 参数存在，不引入任何平台 SDK 或侵入式 API 调用。
- **全外置配置**：配置与状态存储完全遵循 XDG 标准，不向研究代码库提交配置文件或修改 `.gitignore`。
- **非分布式调度器**：系统专注于点对点的单任务可靠生命周期管理，不涉及多集群自动调度与资源抢占池构建。
- **非双向实时同步器**：数据流转依赖显式的只读快照与不可变产物发布，不实施后台文件变动监听与双向文件冲突合并。

---

## 2. 模块划分与职责

主要模块及职责：

| 模块 | 运行环境 | 主要职责 |
| :--- | :--- | :--- |
| `core.py` | 本地环境 | XDG 路径解析、TOML 配置校验、路径安全沙箱检查、restic 与 rclone 底层调用封装 |
| `colab_runner.py` | 本地环境 | 基于 `google-colab-cli` 的会话分配、轻量引导文件注入、状态轮询与生命周期控制 |
| `remote_worker.py` | Colab 节点 | 运行环境准备、Git 固定版本检出、输入快照还原、命令托管执行、输出快照发布、安全凭据擦除 |
| `cli.py` | 本地环境 | 面向工程师与自动化 Agent 的命令行交互入口及标准输出格式规范 |
| `ssh_observe.py` | 本地环境 | 对已有 Colab 实例建立一次性 SSH 连接，查询系统指标或执行 Agent 指定的诊断命令；不修改研究代码，不自动建机/关机 |

---

## 3. 核心设计不变量 (System Invariants)

1. **版本绝对确定性**：
   - 代码执行严格绑定已推送至远程仓库的固定 Git Commit Hash。
   - 数据输入与产物输出严格绑定 64 位十六进制 Snapshot ID，严禁使用指向不确定的动态标签（如 `latest`）。
2. **路径相对性与安全沙箱**：
   - 快照输入与输出路径必须为相对于项目根目录的合法相对路径。
   - 发布时允许明确选中的末段软链接（包括外部模型/数据）在工作区外临时物化，恢复到原相对路径；不得通过链接祖先或自动跟随目录内部的链接。禁止 `..` 越界；物化暂存树排除嵌套 `.git` 元数据并拒绝特殊文件，普通发布路径本身禁止 `.git` 分量。
3. **输入恢复快照优先（Snapshot Precedence）**：
   - 远端恢复输入快照时必须覆盖与 Git 跟踪文件同路径的历史内容（`--overwrite always`），保证数据快照作为输入的唯一契约。
4. **进程生命周期与网络解耦**：
   - 远端 Worker 以独立操作系统会话（`start_new_session=True`）启动，脱离控制端 SSH/CLI 连接状态，规避终端断开引发的 `SIGHUP` 异常终止。
5. **算力释放与现场保护分离**：
   - Worker 写入 `SUCCEEDED` 后保留实例；本地 `watch` 读取最终状态并按 `auto_stop` 选项关停实例。无输出任务可报告空快照 ID。
   - `--detach` 后必须另行执行 `watch` 或手动停止，否则实例可能持续计费；`--no-stop` 明确保留实例。
   - 遇到 `FAILED` 状态时保留实例现场，供调试与日志排查。
   - 遇到 `UNREACHABLE` 控制信道中断时，系统标记为未知状态，严禁擅自关停运行中实例。

---

## 4. 运行时状态机 (State Machine Specification)

```text
       [ 本地环境 ]                                     [ Colab 远端环境 ]
       
   LOCAL_PREFLIGHT (参数/Git状态检查)
          │
      ALLOCATE (申请计算实例)
          │
        STAGE (注入 Worker 与凭据) ── 启动独立进程 ──>  PREPARING (安装基础工具/克隆仓库)
                                                                 │
                                                             RESTORING (快照解包写入本地盘)
                                                                 │
                                                              RUNNING (执行原生命令)
                                                                 │
                                                            PUBLISHING (备份输出产物)
                                                                 │
                                                             SUCCEEDED / FAILED
                                                                 │
   WATCH / STATUS (状态轮询) <────── 状态回传 ─────────────────────┘
          │
  ┌───────┴────────────────────────┐
  ▼                                ▼
SUCCEEDED ──> watch 确认后按需释放实例    FAILED ──> 保留实例供排障 (需手动 stop)
```

### 状态定义说明
- `PREPARING`：实例初始化阶段，安装 `git`、`restic`、`rclone`，拉取指定 Git Commit。
- `RESTORING`：调用 restic 从云端仓库流式解包输入快照至本地临时工作区 `/content/mlbridge-workspace/`。
- `RUNNING`：执行项目 `setup` 依赖安装，并在独立 OS 进程中启动目标计算 `command`。
- `PUBLISHING`：计算完成后，将项目配置中指定的 `outputs` 目录打包推送至云端存储，生成输出快照。
- `SUCCEEDED`：全部阶段正常结束；本地 `watch` 读取状态后按配置回收实例。
- `FAILED`：任意远端阶段异常中断，保留错误堆栈及临时环境。
- `UNREACHABLE`：本地与远端控制信道连续超时，保持现状等待重连。

---

## 5. 数据存储与快照模型

### 5.1 存储命名空间
云端对象存储按项目实施物理命名空间隔离：
```text
rclone:<remote>:<prefix>/<project_name>
```
每个项目对应独立的 restic 加密仓库及索引元数据，使用配置的 restic 密码文件。

### 5.2 增量去重与传输优化
- restic 在底层利用内容定义分块（Content-Defined Chunking）实施增量去重。
- 连续版本的数据集或相似产物在推送到云端存储时仅上传新增变更数据块。
- 任务执行时，Colab 实例将快照恢复到本地工作区，计算命令读写本地盘。

---

## 6. 安全架构与凭据生命周期

1. **最小凭据权限**：本地密码文件与配置文件权限由程序强制设置为 `0600`。
2. **凭据传输**：rclone 配置文件与 restic 密码在任务执行期间上传至 Colab 的 `/content/`，不写入研究项目的 Git 仓库。
3. **GitHub 私有仓库凭据**：本地 `MLBRIDGE_GITHUB_USERNAME` 与 `MLBRIDGE_GITHUB_TOKEN` 必须成对设置；若均未设置，则从当前用户 `~/.config/zsh/private/secrets.env` 以 `awk` 提取两条赋值（不执行整个文件）；仅接受不含凭据的 GitHub HTTPS repository URL。PAT 以本地临时文件上传，并由 Worker 的 `GIT_ASKPASS` 在克隆时读取，克隆后立即移除。Colab 同一 OS 用户可在上传/克隆窗口读取凭据，不能将此机制视为隔离的可信执行环境；令牌权限应限定到目标仓库且可撤销。
4. **凭据清理与局限**：
   - Worker 启动后正常完成或进入受控异常路径时，在 `finally` 块中执行临时凭据文件的 `unlink`；进程被强制终止或上传阶段断连时无法保证清理。
   - 注入阶段异常时，本地调度器尝试在实例上删除已上传的敏感文件；控制通道不可达时无法确认清理结果。
