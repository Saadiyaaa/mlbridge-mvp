# MLBridge v0.1.1 (可运行的设计原型)

**给现有 Git 项目外挂数据快照和 Colab 执行能力；不改项目源码、不写入项目配置、不要求项目导入 SDK。**

本原型固定使用：Git/GitHub · restic · rclone · Google Drive · 官方 `google-colab-cli`。SSH 是可选监控通道，不传大数据。

> ⚠️ 当前属于 **离线测试通过的 MVP 原型**，而不是已在你的 Google Drive/Colab 账号上验证过的生产级工具。`colab` 未在本环境提供，**未进行真实云端集成测试**；`restic`/`rclone` 的发布会/恢复链路已在 restic 0.18.0 + rclone 1.75.1（local 后端）上真实验证。启用 GPU 前，务必先用一个小目录和短任务跑通全链路。

## v0.1.1 修复与优化（相对 v0.1）

1. `jobs.<job>.outputs` 允许为空数组：纯 setup / 仅产生外部副作用的任务可以运行（remote_worker 本就支持）。所报错误文案随之更准确。
2. 远端恢复输入改为“快照优先”：快照中与 git tracked 文件同路径的内容不再被静默跳过。restic ≥ 0.16.2 显式传 `--overwrite always`，更老版本缺省即无条件覆盖，语义自动一致。
3. restic 版本自适应：`--overwrite` 旗标在 0.16.2 才引入；对 Ubuntu 22.04 / Colab apt 常见的 0.12.x 旧版不再传入不存在参数。本地 `fetch` 在旧版 restic 且未显式 `--overwrite` 时拒绝恢复并提示升级，而不是静默覆盖。
4. Colab 注入失败路径：若凭据已上传，向 kernel 注入清理脚本尽力删除 `/content` 临时凭据副本；失败时保留明确指引。
5. `storage.prefix` 支持绝对路径（用于 local 类 rclone remote 冒烟；云端 remote 相对路径语义不变）。
6. `doctor` 增加 python 版本、密码文件权限、restic `--overwrite` 支持情况的自检。
7. 其他：清理重复代码与未使用 import；`tests` 从 10 条增至 15 条，新增旧版 restic 兼容、快照优先语义、staging 失败清理等回归测试。
8. 明确约束：一个 Colab VM 只应服务一个 job（worker 固定 clone 到全新目录，同 session 重跑同一 job 会失败）。

## 核心流程

```text
已存在的项目代码 ── Git push ──> GitHub ──────────> Colab clone/checkout
       │                                         │
       └── mlbridge publish ──> restic/rclone ──> Drive
                                                 │
                                 restic restore ─┘
                                                 │
                                      运行原命令；写入原路径
                                                 │
                     Drive <── restic/rclone <────┘
                       │
                 mlbridge fetch
                       │
                     本地目录
```

## 安装

推荐 Python 3.11+。在工具目录（**不是研究项目目录**）中：

```bash
python -m pip install .
# 或者：uv tool install .
```

先在你自己的服务器上准备好命令：`git`、`restic`、`rclone`、`colab`（官方 `google-colab-cli`），以及 Colab CLI 登录状态。`rclone config` 必须已能访问 Google Drive，并推荐配置你自己的 Google OAuth client ID。

## 第一次配置（不会改动研究项目）

```bash
mlbridge init
# 编辑 ~/.config/mlbridge/settings.toml
# 将 remote 设置为已经认证的 rclone remote 名称
# 核实 rclone_config 指向现有的 rclone.conf

mlbridge register image-project \
  --workspace /home/user/projects/image-project \
  --repo https://github.com/you/image-project.git

# 编辑 ~/.config/mlbridge/projects/image-project.toml
# 把 jobs.example 改成项目原本的运行命令和输出目录

mlbridge doctor --project image-project
mlbridge repo-init image-project  # 项目仓库只初始化一次
```

`mlbridge init` 生成一个仅保存在用户配置目录中的 restic 仓库密码文件（0600）。**务必安全备份这份密码**。其他 Worker 想访问同一个仓库，必须获得同一个密码和可用的 Drive 凭据。密码与 rclone.conf **不能进入 GitHub**。

仓库路径固定为 `rclone:<remote>:<prefix>/<project>`；不同项目独立 restic repository，便于后续隔离、恢复、管理。

## 1. 发布数据（Colab 尚未启动）

```bash
mlbridge publish image-project \
  --label raw-v1 \
  --path data/raw
```

命令会输出 `snapshot_id`。再次发布时会利用 restic 已存在内容做增量去重。注意这是**显式快照**，不是后台目录监听器。

输入/输出始终用**项目根目录的相对路径**（例如 `data/raw` 或 `outputs/run37`）；restic 在当前项目目录中以相对路径备份，从而可以在另一台机器恢复成相同相对目录结构。

## 2. 将任务送到 Colab

编辑外部配置：`~/.config/mlbridge/projects/image-project.toml`：

```toml
name = "image-project"
workspace = "/home/user/projects/image-project"
repository = "https://github.com/you/image-project.git"

[jobs.upstream]
command = ["python", "upstream.py", "--input", "data/raw", "--output", "outputs/upstream"]
setup = [["python", "-m", "pip", "install", "-r", "requirements.txt"]]
outputs = ["outputs/upstream"]
```

运行：

```bash
mlbridge run image-project --job upstream --input <snapshot_id> --gpu L4
```

本地 CLI 会：
1. 检查本地 Git HEAD 是否干净，并记录完整 commit ID。**运行前务必 `git push`**，使该 commit 在远端可用。
2. 创建一个命名 Colab session。
3. 通过 Colab 文件 API 只发送几个小文件（任务声明、外部 worker、rclone.conf、restic 密码）；**不会直接传几十 GB 的训练图片**。
4. 在 Colab 内启动与 SSH 无关的独立 worker：检查工具、克隆代码、恢复指定 snapshot、运行外部配置的普通命令、对 output 做 restic backup。
5. 轮询状态；成功后记录结果 snapshot ID，并显式停止 session。

需要脱离本地前台运行：

```bash
mlbridge run image-project --job upstream --input <snapshot_id> --gpu L4 --detach
mlbridge status <session>
mlbridge logs <session> --lines 200  # 不下载整个日志
mlbridge watch <session>  # 重连监测；成功后释放 runtime
```

**重要：** `--detach` 并不是后台自动停止保证。需要有一个本地 watcher 在运行，或在完成后执行 `watch/stop`。本地监控长时间不可达时，工具会返回 UNKNOWN，**不会把 SSH 断开当成任务失败，也不会擅自 stop Colab**。Colab 平台仍可能回收临时 VM；没有工具能保证不中断。

通过 SSH 查看任务（不带 `--rm`）：

```bash
colab ssh -s <session>
# 在 Colab 看 /content/mlbridge_run.log
# 看 GPU: nvidia-smi
```

如果研究项目需要不同环境（conda、CUDA、额外 apt 软件等），应在外部 `jobs.<job>.setup` 中描述。**不能凭空自动推断任意项目的环境**。

### SSH / Colab CLI 限制

这个实现依赖 Colab 帐号和 runtime 支持官方 CLI/SSH 能力；并不保证所有 Colab 套餐或临时 VM 都能使用。**不要为保持 session 而使用 `colab ssh --rm`**；它可能在连接关闭时停止 runtime。

### Colab 断线

任务是由 Colab 的 Python 环境启动的独立 OS 进程，即使本地 SSH 断开，也不会因为 SSH SIGHUP 被立即终止。不过 Colab 官方依据运行时活动来维持 VM 存活；**不能仅凭 OS 进程独立就保证平台不会误判空闲并回收**。这一点必须通过真实 Colab 任务验证。对于耗时很长的任务，更稳妥的做法是让 kernel 执行保持活跃，或使用任务自身 checkpoint。

## 3. 获取结果

```bash
mlbridge artifacts image-project
mlbridge fetch image-project <output_snapshot_id>
```

`fetch` 默认 `--overwrite never`，不覆盖已有文件。**如果目标路径已有同名但内容不同的文件，这种安全模式可能跳过它们，因此不保证目标工作树与快照字节完全相同**。推荐恢复到空工作区，或在明确接受覆盖后加 `--overwrite`。默认不会删除目标中多余的文件。

也可以用 `--target /some/isolated/workspace` 恢复到指定工作区。

## 数据一致性与安全约定

- 所有已完成的 artifact 用 snapshot ID 标识；禁止使用含糊的 `latest` 作为跨 Worker 输入。
- 每个任务指定明确的 output 路径；`jobs.<job>.outputs` 允许为空（不发布快照），但提供了 outputs 时任务在 `RUNNING → PUBLISHING → SUCCEEDED` 后才允许自动释放 GPU。
- `FAILED` 会保留运行时供排查；用户需要主动 `mlbridge stop <session>`，否则会继续消耗资源。
- `UNREACHABLE` 是网络控制面故障，不代表 GPU 任务失败。
- restic snapshot 提供成功提交的边界；它不是多端双向目录合并器，也不会帮多个任务解决输出覆盖冲突。
- 模块不会自动调用 `restic forget/prune`，仓库维护应由管理员单独、受控执行。
- rclone OAuth 配置及 restic 密码复制到 Colab 临时 VM 是敏感凭据暴露；v0.1 没有细粒度的临时凭据隔离。只在你信任的 Colab runtime 使用。
- v0.1 默认支持公开可克隆的 Git 仓库；私有仓库认证未自动部署。
- 当前没有 Colab 容器预热/自定义镜像，因此 **40GB 冷数据下载、解包、环境安装会消耗 Colab runtime 时间**。这是现有设计的主要性能边界。

## 目录边界

```text
研究项目（原样）           XDG 用户配置（模块自有）
/home/.../project/         ~/.config/mlbridge/
├── train.py              ├── settings.toml
├── data/                 ├── restic-password
└── outputs/              └── projects/<name>.toml

本地任务状态               临时 Colab VM
~/.local/state/mlbridge/  /content/mlbridge_*.json
└── jobs/<session>.json    /content/mlbridge-workspace/
```

该工具不修改 `.gitignore`、`requirements.txt` 或研究代码文件。

## 验证方式

```bash
python -m pip install -e .
pytest -q
```

本版本已有离线单元测试，但没有真实的 Drive/Colab 集成测试。请先用小型测试项目、少量图片验证：初始化仓库 → 上传 → Colab 恢复 → 普通命令运行 → 发布 → 本地恢复 → 自动 stop。

## 尚未实现（故意不做）

DVC/MCP/AI 框架适配、自动多机调度、NAS 缓存、同步冲突合并、自动打包 shard、网页管理台、后台 daemon、私有 Git 仓库凭据配置、多项目统一元数据库、Colab 断电强制恢复。

更完整的设计边界和风险清单见 `docs/DESIGN.md`。
