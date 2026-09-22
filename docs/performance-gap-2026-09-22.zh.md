# MTP 吞吐差距调查 — 2026-09-22

## 范围与差距

更新后的基线为官方 vLLM main `e9f169d16b9408bb9ae44f75072b91a5521d733c`，使用其官方 cu130 wheel、V2 GPU runner 和 Torch2.13.0。本项目使用独立 Torch2.14.0、Triton3.8.0、FlashInfer0.6.18.post1，不导入/链接 vLLM。参考工作负载为单张 B200、MTP4、batch4、输入 32768、输出 4096。基线吞吐中位数 729.621 tok/s，95% 门槛为 693.140 tok/s。

提交 5166885 上，六组 ordinary 和 MTP batch1/2 均通过正式吞吐/TTFT/稳定性/审计。三组 prefix 的 TTFT 失败；MTP batch4 多次运行被外部 GPU 任务打断，保留了接近 683 tok/s 的诊断结果。这不是完整最终验收。

随后 GDN prefill 归一化及原生短前缀注意力在单次测量诊断中达到 689.346 tok/s，仍需增加 0.550% 吞吐，或减少 130ms batch 总时长。MTP prefill 从 FA2 改用 TRT ragged 后测得 692.012 tok/s，仍略低于门槛，且需正式两次预热/五次重复验证。估算加速不能作为验收。

### ee2fb63 上的正式后续测量

全部 12 组在相同干净源码上完成两次完整预热和五次测量，CPU 亲和性匹配基线，并通过编译/捕获审计。十组通过。MTP batch4 稳定中位数为 692.190 tok/s，基线的 94.8698%，低于 693.140 门槛。还需 0.1373% 吞吐，等价于完整 16384 输出 token 的 batch 减少 32.45ms。TTFT 为 6.668s，基线的 96.998%，已通过。

六组 ordinary 全部通过，吞吐为基线的 108.96–116.53%。MTP batch1/2 为 98.83/96.97%，通过。Prefix batch2/4 的两项门槛及稳定性均通过，TTFT 比例 81.05/78.04%。Prefix batch1 达到吞吐和时延门槛（116.97/82.44%），但未通过 5% TTFT 稳定性要求。两次 prefix1 尝试保留完整重复数据，最新范围 44.72–51.16ms，极差比例 14.20%。这是独立于 MTP 吞吐的未解决问题。

三次 MTP4 和一次 prefix4 因外部 GPU 进程进入而中断并排除。完整结果和失败来源在 `bench/baseline/2026-09-22-performance-gap-candidates.json`。采集结束后自有 GPU 进程全部退出。这些结果取代前面的诊断差距估计；整体 12 组验收仍未完成。

## 匹配 shape 的 decode kernel 对比

两个诊断 profile 都在 32768 输入/512 输出 MTP 运行的约第 30 步，为四个请求执行 20 个 target query token。下表累加实际 CUDA kernel event，不是 CPU wrapper 时间或嵌套 CUDA Graph 注解。基线 target/draft FMHA 按执行顺序区分（16 次 target +4 次 draft）；项目 draft 数值包含 split-KV merge。

| 每步工作 | 项目 ms | vLLM ms | 观察 |
|---|---:|---:|---|
| FP8 linear GEMM，272 次 |6.944|7.176|项目 TRT GEMM 已有竞争力 |
| 词表投影，5 次 |1.761|1.869|CuTe-DSL 投影不是主要差距 |
| Target 全注意力，16 次 |1.397|1.386|原生 target 路径基本相当 |
| Draft 注意力，4 次，含 merge |0.964|0.338|最大且可明确隔离的不利 kernel 差异 |
| 所有 CUDA kernel |14.279|14.095|其他较快的项目 kernel 抵消了不少 draft 损失 |
| CUDA kernel 数 |1349|1125|项目仍启动更多独立逐元素/元数据 kernel |

这些是在不同 B200 和 CPU 亲和性下的单次 profiler 诊断，不是正式吞吐对比。profiler 开销及 GPU 重叠意味着累加 kernel 时间不能直接代替墙钟时间。基线 profile 使用预热后的迭代，项目选定步骤在短运行结束前使用已捕获的完整 batch 图。框架/库版本也不同。

## 路径不同的原因

1. **Draft 注意力及缓存语义。** 项目 MTP 行 p 配对 target hidden[p-1] 与 input[p]，位置 0 不存在，以在跨页时保持 Rust 不可变 token-prefix hash。split-KV Triton kernel 显式排除 0。vLLM 自回归 speculator 以自身缓存/元数据契约配对 hidden[P] 与 token[P+1]，draft 也运行原生 FMHA。直接对 draft 开启 target 原生路径会包含无效行，需要正确适配器或更快的首有效位置 kernel，不能移除 mask 或修改共享前缀边界。
2. **MTP prefill 分派此前仍为 FA2。** Target 长 prefill 已迁移 TRT ragged，而独立 MTP 层仍用 FA2。排除 first0 的受控 gather 比较中，32K query/32K KV 的 FA2/TRT 为 35.18/9.55ms，32K query/131K KV 为 241.40/71.90ms。新路径保留相同有效 token、因果语义和 FP32 softmax。这是单独的 prefill 改进，不能解决上面的 draft decode kernel 问题。
3. **逐元素融合不同。** vLLM 融合卷积后 GDN 递归、gating 和输出归一化，融合 Q/K RMS+RoPE+gate 准备，并有编译器管理的残差/量化融合。项目使用较小自有 kernel 和显式不可变卷积快照。本 profile 中单独递归 kernel 更快，因此直接与 vLLM 范围更广的融合 GDN kernel 比较会误导，应比较完整工作序列。
4. **已接受 token 的记账经过 CPU。** 项目贪心采样返回 `.tolist()`，在 CPU 验证候选、构造 draft 元数据，再分派 catch-up/proposal 图。vLLM 在 GPU 保留 sampled/rejected 计数，并在 GPU kernel 准备 draft 输入。项目 trace 中 target 输出约 12.54ms 到主机，catch-up GPU 工作约 13.20ms 开始，提示潜在关键路径空隙，但不能据此确定对 vLLM 的净差距。11.12ms CPU 采样注解主要等待 target GPU；D2H 本身仅 3.52us。不能称其为 11ms CPU 或 ZMQ 开销，尚未证明 Rust 调度/ZMQ 是瓶颈。

MTP 接受率没有崩塌：完整 batch 基线诊断使用 1121 步，最新项目诊断为 1117 步，小差异无法解释逐步 draft 注意力成本。合法 BF16 后端舍入可能导致 token 轨迹不同；这是工作量诊断，不是生成 token 相同的声明。

## 受控检查和优先级

- 仅把 draft KV 存储从 NHD 改为 head-major 没有帮助：batch4/query1 为 0.25625/0.25510ms，query5 为 0.21202/0.21999ms。不能仅因上游布局不同就重写布局。
- 量化 warp 数扫描未优于原 launch 设置。
- GDN prefill Q/K 归一化融合移除重复 FP32 中间量；32768 行算子诊断从 1.661ms 降至 0.081ms，保留 FP32 norm/epsilon 和 BF16 输出，并用 FP64 验证。
- 原生短前缀 context attention 将 batch4/624-query/32768-KV 算子从 6.54ms 降至 1.68ms；项目 TTFT 诊断从 216ms 降至 138ms。长 target prefill 保留 ragged，因为原生分页原型在该场景较慢。
- 正式测量确认 MTP4 吞吐仍未达标。优先优化显式 mask 的 draft 注意力 kernel 和已接受 token 的 GPU 记账。保留前缀隔离、约束解码、全部状态快照及 block784，避免缺乏证据的大规模 GEMM/Rust 调度重写。

## 源码与证据

项目源码：`python/oh_my_vllm/worker/mtp.py`、`worker/model_runner.py`、`worker/sampler.py`、`kernels/decode_attention.py`、`kernels/mtp_attention.py`。冻结 SHA 的上游源码：

- [V2 自回归 speculator](https://github.com/vllm-project/vllm/blob/e9f169d16b9408bb9ae44f75072b91a5521d733c/vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py)
- [V2 model runner](https://github.com/vllm-project/vllm/blob/e9f169d16b9408bb9ae44f75072b91a5521d733c/vllm/v1/worker/gpu/model_runner.py)
- [FlashInfer 注意力后端](https://github.com/vllm-project/vllm/blob/e9f169d16b9408bb9ae44f75072b91a5521d733c/vllm/v1/attention/backends/flashinfer.py)
- [融合 GDN CUDA 实现](https://github.com/vllm-project/vllm/blob/e9f169d16b9408bb9ae44f75072b91a5521d733c/csrc/libtorch_stable/gdn/fused_gdn_decode_kernel.cu)
- [RMS/量化融合](https://github.com/vllm-project/vllm/blob/e9f169d16b9408bb9ae44f75072b91a5521d733c/vllm/compilation/passes/fusion/rms_quant_fusion.py)

[当前 Developer Guide](https://docs.vllm.ai/en/latest/contributing/) 作为背景；测量和代码对比使用上述冻结 SHA。`bench/baseline/2026-09-22-performance-gap.json` 保留摘要和原始日志 hash。原始 profiler trace 留在 `/tmp`，有意不提交。
