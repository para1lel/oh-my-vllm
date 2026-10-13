# Rust 所有权

修改调度器, 缓存或服务前读取[架构](../docs/architecture.zh.md).
Rust 控制逻辑槽位和已接受的请求历史. GPU tensor 所有权留在 Python.
保持先验证后提交状态, FA/GDN 协同分配, 以及仅发布已提交的前缀.

取消和 worker 替换必须释放预留资源, 并使正确的 worker epoch 失效.

## 代码与检查

使用 Rust edition 2024, rustfmt 和 Clippy. 所有物理行最多 100 个字符, 包括宏和字符串.
每个 `unsafe` block 都必须有 `// SAFETY:` 说明.
名称无法表达的寿命, 并发或数值契约保留简短注释.

函数和定义块之间留空行. import 和 module 声明保持分组.
测试关键状态和失败路径. 常规服务路径放在集成测试中.
通过环境包装脚本运行 workspace 测试, 格式, 行宽和 Clippy 检查.
