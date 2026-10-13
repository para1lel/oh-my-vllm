# 编译与 graph 概念

## Semantic IR 与 graph

Semantic IR 在选择实现前命名数学算子, 状态影响和 provider 契约.
[Registry](../../python/oh_my_vllm/ir/core.py) 将 reference 和 mutation schema 与算子共同保存.
Compiled fullgraph 单元包含 GPU 计算; 主机调度和传输具有不同边界.

CUDA Graph 使用持久输入与输出存储 replay 工作.
变化的 metadata 不能在这些存储中留下过时值.
Warmup 与 capture 不得改变已提交缓存状态.
[Capture 事务](../../python/oh_my_vllm/worker/prefill_graph.py)
在恢复正常执行前恢复持久数据. Decode 的对应契约见 [decode graph](../../python/oh_my_vllm/worker/decode_graph.py).

Compiler 条目, 生成 metadata, executable 所有者和 graph pool 具有不同寿命.
其预算放在 [compiler cache](../../python/oh_my_vllm/ir/cache.py),
[metadata 所有权](../../python/oh_my_vllm/ir/metadata_cache.py) 和
[graph cache](../../python/oh_my_vllm/worker/graph_cache.py) 中.
Weak 诊断不得延长 executable 寿命.

Rewrite 必须保持可观察 BF16 cast, 状态顺序, 异常行为和 alias 契约.
只有已证明的临时存储可以 donation. 持久缓存不能.
独立 reference 和 replay 测试验证这些边界.

## CUDA stream 与 PDL

Stream 可以重叠独立工作. 共享数据仍须遵守明确依赖和寿命规则.
Programmatic dependent launch 可以在前一个 grid 完成前启动依赖 block.
依赖 block 读取未完成输入前必须等待.

读取 [原生 launch 契约](../../python/oh_my_vllm/kernels/cuda_backend/operators/common.cuh)
和 [worker graph 执行](../../python/oh_my_vllm/worker/decode_graph.py).
调用者 stream 拥有原生 launch. 被 capture 的工作在 replay 期间保持存储所有者.

Graph capture 可以保留普通执行会释放的 allocator 存储.
测量与归因见 [环境配置](../development.zh.md) 和 [内存诊断](../profiling.zh.md).
不能仅依据有限条目数推断内存平台.
