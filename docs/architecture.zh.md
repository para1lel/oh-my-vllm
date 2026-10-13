# 架构

Rust 负责服务, 调度和逻辑缓存. Python 负责 GPU tensor 和计算. 按改动需要读取对应主题.

- [运行时所有权](architecture/runtime.zh.md)
- [调度与缓存概念](architecture/scheduler.zh.md)
- [目标模型概念](architecture/target.zh.md)
- [推理草稿概念](architecture/speculation.zh.md)
- [编译与 graph 概念](architecture/compilation.zh.md)
- [测量与扩展边界](architecture/extensions.zh.md)
