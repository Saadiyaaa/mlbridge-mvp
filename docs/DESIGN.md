# MLBridge 技术设计 v0.1.1

## 目标 / 非目标

**目标：** 给任何现有 Git/ML 项目外挂一个 CLI，实现显式数据增量发布、按 snapshot 获取数据、在 Colab 临时执行现有 argv 命令，结果持久化后结束 runtime；全部配置和状态均在项目外。

**非目标：** 将多个项目硬套进新框架、监听全项目实时双向同步、自动推断输入输出、实现 DVC Pipeline、自动修复任意环境/任意训练任务。

## 模块边界

1. `core.py`：XDG 路径、TOML 项目注册、静态配置、路径保护、restic + rclone 仓库接口。
2. `colab_runner.py`：官方 `colab` CLI 的会话创建、文件注入、状态监控与释放。它不直接传输大型数据。
3. `remote_worker.py`：Colab VM 临时执行器，处理 Git checkout、输入恢复、原命令调用、输出发布及 JSON 状态。
4. `cli.py`：向人和 AI Agent 暴露少量命令。

研究项目仅被作为 `workspace` 和 `command`，完全不需要导入本工具。

## 不变量

- 一份已发布的数据版本用明确的 restic snapshot ID 引用，不通过 `latest` 隐式绑定。
- snapshot 的文件路径必须是相对项目根目录的路径，避免本地绝对路径在 Colab 上失效。
- 所有 Colab 训练输入恢复到 `/content/mlbridge-workspace`，不直接从挂载 Drive 随机读取图片。
- 恢复输入时快照优先：restic ≥ 0.16.2 用 `--overwrite always`；更老版本缺省即覆盖（`--overwrite` 旗标需探测，避免 unknown flag）。
- 绝不把 SSH 的存活当作任务存活；不传 `--rm`。
- 只有远端 worker 报告 `SUCCEEDED`（完成 output backup），本地 watcher 才自动 stop。
- 网络不可达状态是 `UNKNOWN`，而不是 `FAILED`；不会根据隧道掉线自动释放资源。
- `jobs.<job>.outputs` 可为空：无输出任务合法，不进入 PUBLISHING 阶段，也不发布 output snapshot。
- restic 密码和 rclone.conf 不被写入 Git 仓库或 job JSON，临时提供给 Colab，作业结束后从 VM 临时路径删除；`launch` 注入失败路径上会尽力注入清理脚本删除已上传凭据。

## 状态机

```text
LOCAL_PREFLIGHT → ALLOCATE → STAGE → REMOTE_STARTED
                                  |
                               PREPARING
                                  ↓
                               RESTORING
                                  ↓
                                RUNNING
                                  ↓
                              PUBLISHING
                                  ↓
                               SUCCEEDED ── stop

任何远端阶段的异常 → FAILED ── 保留 VM + 日志，手动 stop
任何本地控制通道错误 → UNREACHABLE / UNKNOWN，禁止自动 stop
```

注：Colab 不保证 VM 生命周期；SSH 独立进程仅保证在正常 SSH 断开时不因挂断信号退出，不构成对 Google 后端回收策略的保证。

## 多 Worker 逻辑

每台机器都应安装 CLI 和 Git/Restic/rclone，并复制用户自有的外部项目配置与**同一仓库密码**。代码以 Git commit 固定版本；数据借助每个项目的独立 Drive restic 仓库实现多机发布与拉取；不要求所有服务器同步整个目录。第一版不自动共享完整本地数据缓存，也不解决同时写同一 output 路径的问题。

## 性能权衡

restic 减少 Google Drive 的物理对象数量与重复内容传输，但初次 Colab restore 需要读取当前任务的完整 working set。项目有大量 PNG/JPEG，restic 分块不能显著缩小已经压缩的不同图片；主要收益是跳过不变文件和降低 Drive 小文件 API 压力。Colab 工具安装、环境安装和冷数据恢复都耗时。后续可增加复用已有 Colab session、读取更少输入快照、按需逐阶段执行等优化，但不应先开发完整调度器。

## v0.1 局限与下一步

1. 最重要：需要在实际账号上确认官方 Colab CLI 的 `upload/download/exec` 调用、远端分离进程持续运行、session 回收策略、工具安装与 runtime 权限。**未验证前不能承诺无人值守或运行时不中断。**
2. 必须在真实 restic / Drive 后端验证 snapshot 的 JSON 输出、初次/重复 publish、恢复路径与 API 请求量；目前仅离线模拟。
3. 当前本地轮询器必须存活才能自动 stop；若本地机器掉线或 CLI 崩溃，Colab 可能保持运行直到后端回收。通过 `watch` 恢复，或运行外部 supervisor 托管 watcher。不要在远端执行未经授权的自动 keep-alive 逻辑。
4. 私有 Git 仓库需要独立的只读、短期认证方案；此版本仅建议公开仓库。
5. 恢复到已有项目目录时默认为不覆盖。若要严格校验完整一致性，应实现恢复至隔离临时目录 + 清单校验 + 显式合并策略。
6. 原程序内部若有硬编码绝对路径、依赖交互式终端、依赖特定本地环境，可能需要外部路径映射/适配。模块能做到源码透明，但不能保证任意程序零配置运行。
