# 项目原则

构建在单张 NVIDIA B200 上运行的 Qwen3.8-27B-FP8 服务.
Rust 负责服务, 调度器和逻辑 KV cache.

Python 负责 GPU 计算. 使用 ZMQ DEALER socket 和 msgpack 进行进程通信.
遵守[运行时契约](docs/requirements.zh.md).

工作前先查看 Git 状态, [交接](docs/handoff.zh.md)及适用的目录说明.
遵守 [CONTRIBUTING.md](CONTRIBUTING.zh.md).
Codex 使用 `main`.

根目录指南保持简短. 局部说明放在对应目录.
所有英文文档遵循 ASD-STE100. 每篇英文文档都提供完整中文翻译.

使用清晰名称和小函数. 仅当代码无法提供必要信息时添加注释.
测试保持少量. 新增测试前, 向用户说明必要性.

自行执行适用的已有测试. 不在项目文件中记录测试结果.
使用[测试范围](docs/testing.zh.md).

每轮交付前由另一个代理进行代码审查. 修正错误.
保持 hooks 启用. 只提交本轮修改的文件.

每轮交付后将工作分支推送到 GitHub.
在其他仓库工作或进行其他发布前, 先取得维护者明确指示.

主机配置放在被忽略的 `LOCAL.md` 中. 停止自己启动的 worker 和临时服务.
