# GPU 与 worker 所有权

修改 runner 前读取 [Python 所有权](../docs/architecture.zh.md).
Python 负责 tensor, 模型计算, 采样和 CUDA Graph. 不得自行创建逻辑缓存 block.
保持已接受状态写入, 推理草稿回滚, graph 恢复和调用者 stream 的寿命契约.

保持模型提示, grammar 约束和数值容差有效.

## 代码与检查

使用项目环境包装脚本, 通过 `uv` 安装 Python 依赖.
优先使用明确的类型, 名称和职责集中的模块. 删除重复代码含义的注释.
保留简短的接口, 数值, 寿命和并发契约.

使用 Ruff 和基于语法的定义空行检查. 不拆开装饰器与函数.
常规集成覆盖不足时, 保留独立数学参考和故障案例.
运行受影响的 CPU 测试和适用的 GPU 验收. 每次运行后停止自有 GPU worker.
