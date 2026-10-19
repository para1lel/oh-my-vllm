# 工作状态

工作使用 `nightly`.
GPU 迁移以 [tile 执行设计](tile-execution.zh.md)为准.
[实施计划](tile-migration.zh.md)给出后续任务.

运行时仍使用 native FFN, FA, GDN, vision, 采样和 MTP 实现.
Rust 提供 HTTP, 调度, 逻辑 KV 和 worker 生命周期控制.
Python 通过现有 ZMQ DEALER 和 msgpack 边界提供模型执行.
Tile 引擎和新 HTTP 计费实现尚不可用.

最终目标覆盖全部 GPU tensor 计算, 也包含准备和初始化阶段.
迁移阶段不设性能门槛.
最终验收同时检查完整 task 覆盖, 数值与状态正确性及全部 24 种标称资源 HTTP 性能配置.

先实现独立 HTTP 计费, 并迁出共享构建/参考工具.
然后实现 task 引擎, 贯通直到词表投影的普通 decode.
继续处理 prefill, MTP/采样, vision 及准备/辅助计算.
替换实现通过适用正确性检查后, 删除旧实现.

保持原有模型, 状态和进程契约.
主机路径和产物位置放在被忽略的本地配置中.
