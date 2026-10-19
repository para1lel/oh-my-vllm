# 测试范围

添加测试前使用[贡献流程](../CONTRIBUTING.zh.md).
Agent 负责替维护者执行测试.
执行[开发命令](development.zh.md#git-hooks)中的 hooks.

Rust 测试包括服务, 调度器, 逻辑 KV 和 worker 生命周期的检查.
HTTP 测试通过进程边界使用 Python 占位实现.
这些测试不提供模型推理或 GPU 性能的验收证据.

CPU 检查涵盖数值容差, checkpoint identity, 规范化 HTTP 计费和 manifest 验证.
参考测试包括低精度舍入, GDN carry state 和 MTP 短尾的检查.
约束测试检查 schema mask 及独立的 speculative grammar 状态.
Rust 检查流式 JSON 工具帧以及媒体请求的前缀 cache 隔离.

GPU 数值检查使用[算子开发命令](operators/development.zh.md).
FFN 输出参考使用选定的行. 报告注明抽样范围.
每个 provider 声明其数值抽样范围.

Sampling 检查带种子的选择及 MTP 状态 slot 选择.
FA, GDN 和 MTP 检查持久写入及未改动的保护存储.
Vision 检查 patch 顺序和帧隔离.
覆盖率不完整或检查失败会阻止正确性验收.

完整推理计时使用 [HTTP benchmark](performance.zh.md#测量命令).
它检查准确的输入与输出长度, 未缓存的前缀及实测 MTP 接受情况.
GPU 工作使用 GPU wrapper.
每轮运行后停止自己启动的进程.
原始产物保存在仓库外.

每次交付前, 更新两种语言的 README 中规定的性能表格.
不在项目文档中记录其他测试结果.
