# MLBridge

**面向 Git 机器学习项目的无侵入数据快照与 Google Colab 远程调度桥梁。**

MLBridge 在不修改研究项目源码、不写入项目内配置文件、不引入任何 SDK 依赖的前提下，实现本地代码仓库与云端计算资源的解耦协作。系统通过 Git 固化代码版本，通过 restic 与 rclone 管理云端不可变数据快照，并利用官方 `google-colab-cli` 调度云端 GPU 实例完成任务与产物归档。

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

MLBridge 严格遵循系统分层与隔离原则，所有配置与状态均独立于研究项目目录：

```text
研究项目工作区（无任何侵入）           用户级全局配置（XDG 规范）
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
- **restic**: 0.16.2+（支持 `--overwrite always/never` 语义）
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
```

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

### 5. 提交远程任务
```bash
mlbridge run image-project --job upstream --input a48f93bc21e78491029c782138a0f98d78120491823091248102938102938102 --gpu L4
```
系统将自动：
1. 校验本地 Git 干净度及远端推送状态。
2. 分配 Colab L4 GPU 实例。
3. 注入 Worker 及临时访问凭据并启动独立后台作业。
4. 监控运行阶段直至完成，回传输出快照后自动关停实例释放 GPU。

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
| `publish` | `<project> --label <tag> --path <rel_path>` | 将指定相对路径的数据打包为不可变快照推至云端 |
| `artifacts`| `<project>` | 列出该项目在云端存储中的全部历史快照与标签 |
| `fetch` | `<project> <snapshot_id> [--target <path>] [--overwrite]` | 将指定快照恢复至本地目标目录 |
| `run` | `<project> [--job <name>] [--input <id>] [--gpu <type>]` | 调度 Colab 执行指定任务 |
| `status` | `<session>` | 查询指定会话的本地记录及远端运行状态 |
| `logs` | `<session> [--lines <n>]` | 读取远端任务日志末尾内容 |
| `watch` | `<session> [--interval <s>] [--no-stop]` | 连接已提交的后台任务并持续追踪直至结束 |
| `stop` | `<session>` | 手动终止并释放指定的 Colab 会话实例 |

---

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

---

## 系统设计不变量与规范

1. **版本确定性**：代码执行唯一绑定 Git Commit Hash，数据输入输出唯一绑定 64 位不可变 Snapshot ID。禁止使用动态标签（如 `latest`）作为跨节点输入。
2. **路径相对性**：所有数据快照严格采用相对于工作区根目录的相对路径，确保本地与云端实例目录结构的一致性。严禁包含 `.git` 目录及越界绝对路径。
3. **输入恢复快照优先**：远端恢复输入数据时采用快照优先策略（`--overwrite always`），避免快照数据被 Git 预置文件静默跳过。
4. **生命周期脱钩**：Worker 以独立进程会话（`start_new_session=True`）启动，避免 SSH 断连信号中断计算。
5. **最小凭据暴露**：凭据文件以 `0600` 权限管理，运行时临时写入 `/content`，并在任务结束或异常时执行清理。
