# 给 Pi Coding Agent / 其他终端 Agent 的使用约定

这是一份**外部工具使用约定**，不是研究项目必须安装的 Agent SDK。使用前要由人完成 Colab/Drive 授权和对应项目外部配置。

## 核心边界

- **不要为了运行 MLBridge 修改现有研究项目的代码、配置文件、requirements、README 或 Git 跟踪结构。**
- GitHub 是代码版本来源；只运行已 push 的 Git commit。
- restic snapshot ID 是数据输入来源；不要偷偷用 `latest` 代替指定的输入版本。
- 在有可能需要大数据时，**先检查 snapshot 是否已存在 Drive**，不要在 Colab 启动后才开始中国侧的上传。
- 任务结束后必须看到 `SUCCEEDED` 且带 output snapshot ID（针对有 outputs 的 job），然后才能停止 Colab。
- SSH 用于诊断；不要使用 `--rm`，不要将 SSH 断线解读为训练失败。
- `UNREACHABLE` 表示不可观测，不表示远程任务已结束；禁止执行 stop。

## 最小工作流

```bash
mlbridge doctor --project PROJECT
mlbridge artifacts PROJECT
# 当输入未发布时：
mlbridge publish PROJECT --label LABEL --path data/input
# 确认当前 Git commit 已 push
mlbridge run PROJECT --job TASK --input SNAPSHOT_ID --gpu L4
# 可在第二个终端随时查看：
mlbridge status SESSION
mlbridge logs SESSION --lines 100
# 如果之前采用 --detach，恢复状态监控和最终资源释放：
mlbridge watch SESSION
```

需要 GPU 诊断时，使用官方 CLI `colab ssh -s SESSION` 查看 `nvidia-smi`、进程及 `/content/mlbridge_run.log`。若任务失败，先提取日志，再询问用户是否停止运行时以避免无意丢失诊断环境。若监控不可达，报告状态未知并建议重连；**不要尝试规避 Colab 的平台会话限制**。

## 面向用户报告的任务结果

- `project`、`job`、`session`
- Git commit hash
- 输入 snapshot ID 列表
- 远端最终状态、错误摘要
- 成功时的新 output snapshot ID
- Colab runtime 是否已停止

不要在对话、任务报告、日志中输出 rclone OAuth token、client secret、restic 密码。
