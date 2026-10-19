# MLBridge

**面向 Git 机器学习项目的无侵入数据快照与 Google Colab 远程调度桥梁。**

MLBridge 通过 Git 固定代码版本，以 restic 和 rclone 管理数据快照，并通过 `google-colab-cli` 调度 Colab 实例执行任务、归档产物。配置与状态存放在研究项目目录之外，无需修改项目源码或引入 SDK。

---

## 核心架构与数据流

MLBridge 采用代码轨与数据轨分离的双轨控制架构：

```text
本地项目工作区 ── Git push ──> GitHub ──────────> Colab 克隆与检出
      │                                                │
      └── mlbridge publish ──> restic/rclone ──> Drive │
                                                 │     │
                                 restic restore ─┴─────┘
                                                 │
                                         执行原命令读写本地盘
                                                 │
                     Drive <── restic backup <───┘
                       │
                 mlbridge fetch
                       │
                  本地目标目录
```

1. **代码同步**：本地开发环境直接通过 Git 推送至 GitHub，Colab 实例仅基于指定的 Commit Hash 进行只读检出。
2. **数据发布**：本地数据集通过 restic 进行分块去重加密，经由 rclone 存储至 Google Drive 对应的独立仓库。
3. **环境调度**：通过轻量控制通道在 Colab 启动独立 Worker 进程，拉取代码、恢复输入快照、执行计算、打包输出快照。
4. **产物拉取**：本地通过确定的快照 ID 将远端生成的模型权重或日志还原至本地指定路径。

---

## 存储与系统物理边界

配置与状态存放在研究项目目录之外：

```text
研究项目工作区                       用户级全局配置（XDG）
/path/to/project/                     ~/.config/mlbridge/
├── train.py                          ├── settings.toml        (模式 0600)
├── data/                             ├── restic-password      (模式 0600)
└── outputs/                          └── projects/<name>.toml

任务状态存储（本地只读记录）           临时计算实例（按需创建与销毁）
~/.local/state/mlbridge/              /content/
└── jobs/<session>.json                ├── mlbridge_job.json
                                       ├── mlbridge_worker.py
                                       └── mlbridge-workspace/
```

---

## 环境准备与依赖

### 依赖项
- **Python**: 3.11+
- **Git**: 2.25+
- **restic**: 本地 0.17.0+（`fetch` 默认保护已有文件需 `--overwrite never`）；Colab 端可使用发行版 0.16.x，远端输入恢复沿用其缺省覆盖行为。
- **rclone**: 已配置并认证 Google Drive remote
- **google-colab-cli**: 已登录 Google 账号

### 安装
```bash
python -m pip install .
```

---

## 快速上手

### 1. 全局配置初始化
```bash
mlbridge init
```
此命令在 `~/.config/mlbridge/` 下创建 `settings.toml` 并生成 36 字节的加密凭据 `restic-password`（文件权限自动设为 `0600`）。

编辑 `~/.config/mlbridge/settings.toml`，确认存储后端参数：
```toml
[storage]
remote = "gdrive"             # rclone 已认证的 Google Drive remote 名称
prefix = "mlbridge"           # 仓库在云盘中的根路径前缀
password_file = "~/.config/mlbridge/restic-password"
rclone_config = "~/.config/rclone/rclone.conf"

[colab]
default_gpu = "L4"        # 不传 --gpu 时使用；T4/L4/A100/H100/G4/CPU/NONE
high_mem = false          # 不传 --high-mem/--no-high-mem 时使用
```

已有的 `settings.toml` 可以直接追加 `[colab]`；缺省 GPU 为 `L4`，缺省内存机型为普通机型。命令行 `--gpu T4`、`--high-mem` 或 `--no-high-mem` 可覆盖相应配置。要申请 A100 高内存实例，可设置 `default_gpu = "A100"` 与 `high_mem = true`，或运行：

```bash
mlbridge run image-project --job upstream --input <snapshot_id> --gpu A100 --high-mem
```

`CPU`/`NONE` 创建不带 GPU 的实例；L4 只有一种内存机型，不接受 `--high-mem`。高内存实例需要 Colab Pro/Pro+ 权益，实际机型是否可用取决于账号和当时的资源。配置只在创建任务时读取，不会更改已有实例。

### 2. 注册项目
```bash
mlbridge register image-project \
  --workspace /home/user/projects/image-project \
  --repo https://github.com/example/image-project.git
```

编辑生成的配置文件 `~/.config/mlbridge/projects/image-project.toml`，定义计算任务：
```toml
name = "image-project"
workspace = "/home/user/projects/image-project"
repository = "https://github.com/example/image-project.git"

[jobs.upstream]
command = ["python", "upstream.py", "--input", "data/raw", "--output", "outputs/upstream"]
setup = [["python", "-m", "pip", "install", "-r", "requirements.txt"]]
outputs = ["outputs/upstream"]
```

### 3. 环境与仓库诊断
```bash
mlbridge doctor --project image-project
mlbridge repo-init image-project
```

### 4. 发布输入数据
```bash
mlbridge publish image-project --label raw-v1 --path data/raw
```
输出示例：
```json
{"project": "image-project", "label": "raw-v1", "snapshot_id": "a48f93bc21e78491029c782138a0f98d78120491823091248102938102938102"}
```

**外部软链接输入（默认）**：若 `--path` 明确指向软链接，例如指向工作区外模型的 `models/stable-diffusion/stable-diffusion-x4-upscaler`，`publish` 会在工作区外暂存真实文件，按项目相对路径生成快照；Colab 恢复后得到真实文件夹。可同时指定普通路径，所有选中路径只生成一个快照；项目文件和原软链接保持不变。物化暂存拒绝目录内部的软链接、特殊文件、重复或父子重叠路径，并排除 `.git` 元数据。任何模式下都拒绝经过软链接祖先的路径（应直接选择链接本身）。普通路径不启用暂存，其目录内部若含链接则默认报错。`--no-materialize-links` 禁用物化，显式选中的越界软链接也会被拒绝。暂存优先使用硬链接；跨设备或不支持硬链接时会复制，可能需要与模型大小相当的临时空间。发布期间源文件应保持不变；首次上传和恢复仍可能传输数 GB。

```bash
mlbridge publish image-project --label models-and-input \
  --path models/stable-diffusion/stable-diffusion-x4-upscaler \
  --path data/raw
```

### 5. 提交远程任务
```bash
mlbridge run image-project --job upstream --input a48f93bc21e78491029c782138a0f98d78120491823091248102938102938102 --gpu L4
```
系统将自动：
1. 校验本地 Git 干净度及远端推送状态。
2. 分配 Colab L4 GPU 实例。
3. 注入 Worker 及临时访问凭据并启动独立后台作业。
4. 本地读取最终状态后，默认由 `watch` 关停实例释放 GPU。

使用 `run --detach` 后须继续执行 `mlbridge watch <session>`（或手动 `mlbridge stop <session>`）；远端 Worker 不主动关停实例。若无人接管，实例可能持续计费。`watch --no-stop` 会保留成功实例，需自行关停。

**实例分配守卫（默认上限 1）**：`run` 在 `colab new` 前通过 `colab sessions` 查询活跃实例；达到上限时拒绝分配。查询失败、超时或输出格式无法识别时也拒绝分配，请手动运行 `colab sessions` 确认后重试。上限取自 `settings.toml` 的 `[colab].max_active_sessions`（缺省 `1`；`0` 关闭并发上限检查），也可用 `--max-active-sessions <n>` 指定。无论上限如何，`--session` 名若已对应活跃实例都会被拒绝，避免分配失败时的回收操作停止该实例。守卫查询与分配之间存在竞态；本地锁（状态目录 `launch.lock` 和逐任务的 `<session>.lock`）只协调同机 MLBridge 进程，不约束其他工具或用户。守卫不恢复已消失实例及其 `/content` 数据。

**心跳与本地状态**：远端 Worker 在 RUNNING/PUBLISHING 阶段按默认 300 秒的间隔刷新 `/content/mlbridge_state.json` 的 `heartbeat` 时间戳；状态写入由线程锁串行化。`watch` 将观测结果写入本地任务记录的 `last_known`；远端不可达时，`mlbridge status <session>` 报告 `UNREACHABLE` 并附带最后一次已知状态。终态（SUCCEEDED/FAILED 及 error/snapshot_id）另存到本地 `diagnostics/<session>.json`（0600）。诊断文件记录远端上报结果，不验证输出数据完整性；可用 `mlbridge artifacts` 核对 restic 快照。VM 消失后 `/content` 不可恢复。诊断文件不上传、不含凭据；如需异地保存请自行处理，勿在日志中记录凭据。

**私有 GitHub 仓库**：`repository` 与 Git remote 均保持不带凭据的 `https://github.com/<owner>/<repo>.git`。本地默认读取 `~/.config/zsh/private/secrets.env` 中的 `MLBRIDGE_GITHUB_USERNAME=...`（GitHub 用户名）和 `MLBRIDGE_GITHUB_TOKEN=...`（有该仓库 Contents 读取权限的 fine-grained PAT）；也可为本次进程同时设置两个同名环境变量，覆盖文件中的值。不必把这两个变量先 `export` 到 shell 环境。文件须属于当前用户、是普通文件且权限不高于 `0600`；解析使用 `awk` 提取两条简单赋值（可带 `export` 或整值单/双引号），**不会 source/eval 文件**，不展开 `$VAR` 或命令替换。两个来源不会拼接，只提供用户名或只提供 token 会在创建实例前报错；都没有时走公开仓库路径。不要把 token 写入项目配置、命令参数、远端 URL、脚本或提交到 Git。MLBridge 将 token 放进权限 0600 的临时文件，通过 Colab 上传，克隆时由 `GIT_ASKPASS` 读取，克隆成功或失败后立即清理；不会传给研究任务的 `setup`/`command` 环境。文件中的令牌只在需要 `run` 时读取，不会由 `publish`/`doctor` 读取；若本机该文件默认可用，运行公开仓库任务也会上传此令牌，建议仅在信任的作业环境使用该默认文件。Colab 远端属于可执行项目代码的同一 OS 用户，上传至克隆期间和控制通道异常时不能承诺绝对隔离或可靠擦除；本地进程被强制杀死也可能留下权限为 0600 的临时凭据文件，须在本地临时目录检查并删除；请使用最小授权、限定仓库的可撤销 token。SSH URL、其他 Git 服务及 Git LFS/submodule 的二次鉴权不在此通道内。Git 预检仍要求本地仓库有匹配 URL 的 upstream、HEAD 已推送且已跟踪文件干净；不要把 `--allow-dirty` 当作正常流程。

### 6. 拉取输出产物
```bash
mlbridge artifacts image-project
mlbridge fetch image-project <output_snapshot_id>
```

---

## 命令行接口参考 (CLI Reference)

| 命令 | 参数说明 | 描述 |
| :--- | :--- | :--- |
| `init` | 无 | 初始化全局存储配置及生成主仓库密码文件 |
| `register` | `<name> --workspace <path> --repo <url>` | 在系统外部注册现有 Git 项目配置 |
| `doctor` | `[--project <name>]` | 执行环境依赖、工具版本、凭据有效性自检 |
| `repo-init` | `<project>` | 在云端存储上初始化该项目的专属 restic 仓库 |
| `publish` | `<project> --label <tag> --path <rel_path> [--no-materialize-links]` | 默认物化显式选中的软链接并按相对路径发布；可多次 `--path` |
| `artifacts`| `<project>` | 列出该项目在云端存储中的全部历史快照与标签 |
| `fetch` | `<project> <snapshot_id> [--target <path>] [--overwrite]` | 将指定快照恢复至本地目标目录 |
| `run` | `<project> [--job <name>] [--input <id>] [--gpu <type>] [--high-mem\|--no-high-mem] [--detach] [--max-active-sessions <n>]` | GPU 与高内存默认值取自 `[colab]`；单实例守卫默认上限 1（`0` 只关闭上限检查，复用活跃 `--session` 仍被拒绝）；detach 后须自行 watch/stop |
| `status` | `<session>` | 查询指定会话的本地记录及远端运行状态 |
| `logs` | `<session> [--lines <n>]` | 读取远端任务日志末尾内容 |
| `watch` | `<session> [--interval <s>] [--no-stop]` | 连接已提交的后台任务并持续追踪直至结束 |
| `stop` | `<session>` | 手动终止并释放指定的 Colab 会话实例 |
| `ssh` | `[--timeout <秒>] [--identity <私钥>] <session> [-- <命令及参数...>]` | 只连接已存在实例，执行一次系统观测或远端命令；不会建机/关机 |

---

## Agent 的 SSH 短请求观测

在 `run --detach` 后，Agent 可按自己的周期重新连接已存在实例，每次只执行一个有限时长的请求，无需保持长 SSH 会话，也不要求研究代码提供日志或监控接口：

```bash
mlbridge ssh --timeout 30 <session>             # 默认快照：GPU、CPU/内存、磁盘、进程、工作区及输出目录大小、Worker 阶段
mlbridge ssh <session> -- nvidia-smi              # 自定义系统命令
mlbridge ssh <session> -- ps -eo pid,stat,comm    # 一次性进程查询
```

自定义命令在远端以 `root` 权限执行，只允许可信 Agent 使用；不要在命令行输出凭据。默认快照不读取项目日志正文，不打印进程 argv，只展示文件数量/大小与运行指标；输出目录扫描最多取 20 个配置目录，每目录扫描至多 5000 项、约 2 秒，结果可能标为 `partial`。每次调用退出后不主动关停实例。需要已有 ed25519/ECDSA SSH 私钥（优先查找 `~/.config/mlbridge/ssh/id_ed25519`，随后 `~/.ssh/id_ed25519` 或 `id_ecdsa`）；没有时可在项目外生成专用密钥：`mkdir -p -m 700 ~/.config/mlbridge/ssh && ssh-keygen -q -t ed25519 -N '' -f ~/.config/mlbridge/ssh/id_ed25519`，或用 `--identity` 指定现有密钥。命令本身不会生成密钥；OpenSSH、google-colab-cli 0.7.4 及可用的 Colab SSH 服务为前提。

观测命令只从本地 Colab CLI 已有会话记录中选择实例，临时隔离会话 token 并清理；会话不存在时拒绝，**不自动创建或续建计费实例**。SSH 连通失败、限流或超时只说明本次观测失败，不代表 Worker 已失败；`ssh` 不会自动 `stop`。Agent 应按授权策略判定任务结束，并显式调用 `mlbridge stop <session>` 后确认实例释放。默认 SSH 使用每次临时 known_hosts（`accept-new`），并非长期主机身份验证；请勿在不可信网络/实例上放宽安全假设。

## 运行时状态机

```text
PREFLIGHT (本地检查) ──> ALLOCATE (实例申请) ──> STAGE (文件与凭据注入)
                                                    │
                                                PREPARING (依赖安装与仓库克隆)
                                                    ↓
                                                RESTORING (快照优先覆盖恢复)
                                                    ↓
                                                 RUNNING (脱钩独立进程执行)
                                                    ↓
                                                PUBLISHING (打包输出快照)
                                                    ↓
                                                SUCCEEDED ──> 自动释放实例
```

* **异常分支**：
  * **FAILED**：保留实例及日志供故障排障，需人工介入或调用 `stop` 关停。
  * **UNREACHABLE**：标记网络通信中断，严禁误判为任务失败，严禁擅自关停运行中实例。
  * **关停失败**：已读取到的成功状态保存在本地；`watch` 报告 `stop_error` 且退出码为 1，需重试 `watch` 或手动检查并关停实例。

---

## 系统设计不变量与规范

1. **版本确定性**：代码执行唯一绑定 Git Commit Hash，数据输入输出唯一绑定 64 位不可变 Snapshot ID。禁止使用动态标签（如 `latest`）作为跨节点输入。
2. **路径相对性**：所有数据快照严格采用相对于工作区根目录的相对路径，确保本地与云端实例目录结构的一致性。严禁包含 `.git` 目录及越界绝对路径。
3. **输入恢复快照优先**：远端恢复输入数据时采用快照优先策略（`--overwrite always`），避免快照数据被 Git 预置文件静默跳过。
4. **生命周期脱钩**：Worker 以独立进程会话（`start_new_session=True`）启动，避免 SSH 断连信号中断计算；仅本地 `watch` 在读到成功状态后按需关停实例。
5. **最小凭据暴露**：Worker 启动后将临时凭据文件设为 `0600`，并在任务结束或捕获到的异常时清理；上传与 Worker 启动之间的权限及清理依赖 Colab 控制通道，无法保证异常断连时即时擦除。
