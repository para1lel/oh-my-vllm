# DeepSeek V4.1 适配器

DeepSeek-V4.1-Flash 使用四个 Python rank 和[共享服务](../../serving.zh.md).
公开能力为文本 target 生成和贪心 DSpark5 验证.
图像服务和完整工作负载验收尚不可用.

[适配器](../../../python/oh_my_vllm/models/deepseek_v41/adapter.py)检查发布的配置.
[加载器](../../../python/oh_my_vllm/models/deepseek_v41/loader.py)按 rank 本地放置规则读取原始 48 shard checkpoint.
启动身份覆盖配置和 index 元数据.
加载过程不在 Ready 之前计算完整 weight shard 的 hash.
测量 artifact 绑定完整 checkpoint 身份.

## 指南

- [数值验收和独立参考恢复](evaluation.zh.md)
- [量化和数值次序](quantization.zh.md)
- [投影算术和原生接口](gemm.zh.md)
- [物理缓存和上游格式参考](cache.zh.md)
- [HTTP 工作负载](performance.zh.md), [参考计量](accounting.zh.md)和 [AgentX 工作负载](agentx.zh.md)

## 执行

[Worker](../../../python/oh_my_vllm/models/deepseek_v41/inference.py) 支持 `tp4`, `tep4` 和请求持有的 `dep4`.
绑定允许捕获时, TP4 target decode 使用 CUDA Graph.
TEP4 和 DEP4 target decode 在主机读取路由计数, 因此使用 Eager 执行.
Prefill 和 draft 验证使用 Eager 执行.
[Graph 实现](../../../python/oh_my_vllm/models/deepseek_v41/decode.py)持有捕获的 buffer 和缓存 lease.
未捕获的 shape 和超出所配置 Graph 数量上限的调用使用 Eager 执行.

`OH_MY_VLLM_DEEPSEEK_BATCH_DECODE=1` 选择实验性批量 target 执行.
默认关闭.
[批量算术契约](gemm.zh.md#批量-target-算术)规定每个请求的投影和集合通信规则.
DEP4 和实验性批处理尚未完成真实 checkpoint 数值, 状态, 容量和 HTTP 验收.

DEP4 将每个请求保留在 Rust 选择的 owner rank.
该 owner 使用完整权重计算稠密层, 并从主机存储读取完整 Engram 行.
专家 dispatch 发送 BF16 行, 返回 FP32 本地和.
没有自有行的 rank 也参加专家交换和完成确认.
[DEP4 执行器](../../../python/oh_my_vllm/models/deepseek_v41/dep4.py)协调 target 和 proposal 调用.

### MTP proposal Graph

`OH_MY_VLLM_DEEPSEEK_MTP_GRAPH=1` 在[执行设置](../../../python/oh_my_vllm/models/deepseek_v41/trace.py#L40)<!-- source: def execution_modes( -->中选择 proposal Graph.
默认值为 `0`.
只有启用 TP4 target Graph 执行且 decode Graph 数量上限为正数时, 它才运行.
TEP4 和 DEP4 不使用此 proposal Graph.

[Proposal 路径](../../../python/oh_my_vllm/models/deepseek_v41/target_execution.py#L276)<!-- source: def _proposal_graph_step(self, input_ids, main_hidden, start_pos, slot): -->使用形状为 `[1,1]` 的一个 token 行和形状为 `[1,1,15360]` 的一个 hidden-state 行.
Proposal Graph 仅用于完整窗口分支内的贪心 proposal.
捕获和重放使用相同的[位置限制](../../../python/oh_my_vllm/models/deepseek_v41/proposal_graph.py#L33)<!-- source: def bind(self, input_ids, main_hidden, position): -->.
[Seed 路径](../../../python/oh_my_vllm/models/deepseek_v41/target_batch.py#L194)<!-- source: def _seed_mtp_batch( -->使用 Eager 执行.
Target decode Graph 和 proposal Graph 是不同的图.
[缓存指南](cache.zh.md#proposal-graph-所有权)给出 proposal Graph 所有权和释放顺序.

## GPU 配置

使用[开发配置](../../development.zh.md#原生模块)和 [GPU 选择规则](../../distributed.zh.md#gpu-租约).
启动 worker 前构建物理存储模块:

```bash
scripts/with-env.sh scripts/with-gpu.sh python -m development.native_build --module paged
```

`OH_MY_VLLM_DEEPSEEK_CACHE_CONTEXT` 设置物理上下文容量, 默认值为 65536 tokens.
描述符上限为 1048576 tokens.
物理容量必须覆盖请求的 prompt 和 output 总和.
描述符上限不证明该容量可用.

`OH_MY_VLLM_DEEPSEEK_PREFILL_ROWS` 设置原生 prefill chunk 的总行数.
默认值为 8192, 最大值为 32768.
较小的 chunk 使用较少临时存储.
[路由投影 wave](gemm.zh.md#路由-prefill-波次) 有独立的行数设置.
