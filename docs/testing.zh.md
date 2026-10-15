# 测试范围

添加测试前使用[贡献流程](../CONTRIBUTING.zh.md).
Agent 负责替维护者执行测试.
执行[开发命令](development.zh.md#git-hooks)中的 hooks.

Rust 测试包括服务, 调度器, 逻辑 KV 和 worker 生命周期的检查.
HTTP 测试通过进程边界使用 Python 占位实现.
这些测试不提供模型推理或 GPU 性能的验收证据.

算子契约测试包括负载路径和性能计算的检查.
参考测试包括低精度舍入和 GDN carry state 的检查.

GPU 数值检查和测量使用[算子开发命令](operators/development.zh.md).
FFN 输出参考使用选定的行. 报告注明抽样范围.
每个 provider 声明其数值抽样范围.

Sampling 检查带种子的选择及 MTP 状态 slot 选择.
FA, GDN 和 MTP 检查持久写入及未改动的保护存储.
Vision 检查 patch 顺序和帧隔离.
覆盖率不完整或检查失败会阻止完整算子验收.

GPU 工作使用 GPU wrapper.
每轮运行后停止自己启动的进程.
原始产物保存在仓库外.
不在项目文档中记录测试结果.
