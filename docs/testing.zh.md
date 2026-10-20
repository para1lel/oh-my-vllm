# 测试范围

新增测试前使用 [贡献流程](../CONTRIBUTING.zh.md).
代理为维护者执行测试.
运行[开发命令](development.zh.md#git-hooks)中的 hooks.

Rust 测试包含服务, scheduler, 逻辑 KV 和 worker 生命周期检查.
HTTP 测试通过进程边界使用 Python placeholder.
它们不测量模型推理或 GPU 性能.

CPU 检查涵盖 checkpoint 和 native identities, 全模型报告验证, 规范化 HTTP 计费和 manifest 验证.
约束测试检查 schema masks 及独立的 speculative grammar 状态.
Rust 检查流式 JSON tool frames 和媒体请求的 prefix cache 隔离.

[执行测试](../tests/test_execution.py)检查完整 Execute 的决策和 graph 资源释放.
它们检查录制失败, 调用者 stream 上的重放, 中断及关闭顺序.
它们不提供模型数值或性能证据.

数值检查使用 [全模型评估](operators/evaluation.zh.md).
在固定短提示和一次 decode 迭代后, 将全部词表 logits 与独立 PyTorch FP64 模型比较.
不使用抽样中间输出或旧低精度路径作为数值参考.
后续变更使用同一参考 artifact.

这些路径变化时使用固定的 MTP 恢复, 图像/视频和长上下文用例.
它们的仓库 fixture 包含完整 PyTorch FP64 endpoint logits.
常规测试不运行参考生成器.
[开发命令](operators/development.zh.md#数值命令)负责生成和比较这些 artifacts.

固定算子数值和性能用例使用[算子 microbench 设计](operators/microbench.zh.md).
数值 fixture 和预期 PyTorch 输出保存在仓库内.
常规数值测试直接加载这些 fixture, 不生成参考数据.
性能验收使用冻结的理论延迟倍率门槛.

保持状态选择, 约束, 媒体顺序和请求隔离的服务契约.
内存安全和 graph 资源检查与模型数值检查分开.
Device 内存或同步行为变化时使用 Compute Sanitizer.

完整推理计时使用 [HTTP benchmark](performance.zh.md#测量命令).
它检查精确的输入和输出长度, 未缓存 prefix 及实测 MTP 接受率.
GPU 工作使用 GPU wrapper.
每次运行后停止自己启动的进程.
测量 artifacts 保存在仓库外.

每轮交付前更新两种语言 README 中要求的性能表.
不要把其他测试结果写入项目文档.
