# 教程与清理检查

## 教程与清理

使用 [code-journey](../../code-journey/README.zh.md#验证) 中的检查.
教程修改后检查桌面 / 移动截图和全部 13 条路线.

所有 GPU 程序经过 `scripts/with-gpu.sh`.
完成或失败后停止全部所属进程及后代.
确认 GPU 进程退出, listener 释放和 IPC 清理.
只保留用户明确要求持续运行的服务; 本机信息记入 `LOCAL.md`.
