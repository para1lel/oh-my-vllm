# 开发工具

工具不得成为推理依赖. 实验和原始测量放在外部存储.
使用 `uv` 在选定的项目环境中安装 Python 工具.
环境选择见[开发](../docs/development.zh.md), 验收见[测试](../docs/testing.zh.md).

## GPU 所有权

设置 `OH_MY_VLLM_MODEL` 或提供 `--model`. 使用 `scripts/with-gpu.sh` 等待空闲 B200 并固定 UUID.
成功, 失败, 取消或交接后停止自有 GPU 进程及子进程.
检查进程出生身份和 `nvidia-smi`. 只停止任务拥有的进程.

仅在用户明确请求时保留服务; 在交接中记录用途, 在 `LOCAL.md` 中记录主机身份.

## 依赖与证据

按依赖顺序使用兼容稳定依赖, 通过兼容测试后固定组合.
TileFoundry 是固定 fork 的开发依赖. 简单不兼容可以直接修复.
重大 fork 修改先讨论. 不开 upstream pull request.

已许可的适配分支不授权主项目创建分支.
记录测量源码, 环境, 配置, 模块和 checkpoint 身份.
不挑选有利样本, 不通过修改理论下界隐藏开销.
