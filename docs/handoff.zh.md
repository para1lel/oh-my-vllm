# 当前状态与剩余工作

更新日期: 2026-10-13.

## 实现

Rust 管理服务, 调度及逻辑 KV. Python 管理一张 B200 GPU 的计算. 两者边界使用 ZMQ DEALER 和 msgpack.
保留 784-token block, 16 个 FA layer, 48 个 GDN layer, `mamba_cache_mode="align"` 和 262144 总输入输出 token.
普通解码, MTP4, DSpark, 两种 API, 工具, 思考, 取消, 约束和前缀复用继续可用.
DSpark 权重保持不变.

源码 `efd24d0` 修正服务控制 token ID, 并加入有界的生成 token 循环保护.
源码 `47232e2` 加入持久原生 goal 和可销毁的 rootless Docker sandbox.
源码 `ae070e4` 通过请求准备接受工具描述性关键词, 不改变推理计算或约束解码实现.
模型 prompt 保留原始工具描述. OMP 校验描述中的参数限制, 并向模型返回错误.
显式 strict schema 保留现有合同.

这一小时检查推理框架可用性, 服务合同, 资源上限和清理.
OMP goal 状态, 任务进展, 数据库测试, 工具错误和最终答案属于负载观察.
原始客户端和 collector 失败保留记录中的状态.
Collector 独立检查框架记录, 并单独保留原生负载审计.
期限到达后暂停续行, 发送原生 RPC `abort`, 并排空模型请求.

公共请求和 worker step deadline 分别为 3600 和 600 秒.
原生客户端 first-event 和 idle deadline 均为 900 秒. Goal collector turn deadline 为 7200 秒.
这些策略变更不改变 GPU 计算或解码约束.
用户免除了仅针对超时参数的额外测试.
Oneoff 客户端生成必需的 settings overlay, 并在 handler 停机完成前保存原始 SSE.

## 验证

正式 GPU 采集在干净且冻结的 `ae070e4` 上使用全新项目 cache 及固定依赖组合.
15 个阶段案例均完成完整的 2 次 warmup 和 5 次 measurement, 门槛通过.
普通/MTP4 的 6 个最大 context 案例在 batch 1, 2, 4 通过, 无 OOM 或 recompute preemption.
487 个 GPU 测试全部通过, 无跳过. 60 个思考预算案例全部通过.

两种草稿模式和 API 的约束, lifecycle 和长 context 检查通过.
旧 Oneoff OMP 客户端因缺失 settings overlay 在模型执行前停止. 原始失败记录保留.
源码 `179a56f` 通过 Chat 和 Responses 的 4 项 oneoff 框架检查, 共 31 个真实模型请求.
3 项 native 负载审计通过. DSpark Responses 负载审计失败并保留记录.

517 个算子案例全部通过数值检查. 速度检查通过 514 个, 失败 3 个.
另一张 B200 的全新 cache 测量复现了这 3 个速度失败.
因此完整算子矩阵尚未通过验收.
遵守用户要求, 保持计算不变. 不降低固定 TileLang 性能门槛.

源码 `179a56f` 在 724.00 秒内通过 1565 个 CPU 测试和 156 个子测试, 487 个 GPU 案例被 deselect.
源码哈希保持不变. 8 项必检全部通过.
教程通过 10 个 Node 测试, 9 个浏览器测试, 以及 Rust 调度测试, 格式检查和 lint 检查.

8 个所选 allocator-OOM 阶段通过: `ae070e4` 上的 6 个, 以及 `179a56f` 上缺失的 2 个 `after_commit` 阶段.
这些记录跨两个提交. 新 collector 因未选择全部 4 个阶段而保留失败状态.

## 框架一小时证据

独立审查按用户限定的框架范围, 接受每种模式的一个 3600 秒窗口.
这些窗口使用源码 `ae070e4` 及记录中的超时配置.

MTP 有 412 个请求, 407 个服务端完成及 5 个已匹配终止, 包括停止尾部.
DSpark 有 87 个请求, 79 个服务端完成及 8 个已匹配终止, 包括停止尾部.
源码, 存活进程出生身份和 worker epoch 保持固定. 无意外 OOM 或 worker 重启.
8 条 lane 的工作目录, 容器, 复制工具, cache 及自有镜像均已删除.

Worker RSS 在 MTP 中从 1.136 增至 6.822 GiB, 在 DSpark 中从 1.280 增至 8.519 GiB.
Worker snapshot 最大滞后分别为 136.230 和 138.827 秒. 采样资源上限及最终请求释放通过.
这些结果不能证明内存平台, 完整分配归因, 每个中间时刻, 或该一小时之外的稳定性.

原始 journal/SDK 审计失败单独保留. MTP 有 6 个服务端完成请求缺少同 ID 原生结束消息.
客户端 stream stall 和 incomplete-read 原因仍不确定. 结果不声称每个请求都成功.

## 已知限制

3 个小 normalization shape 未通过不变的算子速度门槛. 用户同意交付且暂不优化这些案例.
见[审计](audit.zh.md). 不降低固定 TileLang 门槛, 也不宣称完整算子矩阵通过.
更完整的内存归因和自然压缩覆盖仍是单独的限制. 模型任务进展属于负载观察.

适用的修复及交付检查已在此记录范围内完成.
[验收索引](acceptance.zh.md) 和[可移植记录](../bench/evidence/README.zh.md) 提供源码身份, 测量及验证限制.
逐日修复经过保留在 Git 历史. 用户未要求 push 或 pull request.

## 教程预览

交互式代码教程继续由用户要求保留的 CPU 预览服务提供.
主机地址, 进程身份和维护例外记录在忽略的 `LOCAL.md` 中.
保留该预览服务. 每次运行后停止本任务 GPU worker 和临时推理服务端口.
