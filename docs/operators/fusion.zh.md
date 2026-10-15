# 算子 fusion 计划

18 个内部语义阶段的张量契约见[阶段清单](../../python/oh_my_vllm/kernels/specification.py).
此实现使用 CUDA C++ 和 CUTLASS.
[运行时契约](../requirements.zh.md)也允许 CuTe DSL.
允许使用 CUTLASS, cuBLAS, CUB 及原生 CUDA 库.

仅使用项目 CUDA 算子及此处允许的原生 CUDA 库.
不使用其他外部算子库, 如 DeepGEMM 或 FlashInfer.
Python 可以分配张量及启动 CUDA 实现.

## Serving 边界

实现与验收使用[负载模型](../../development/kernels/workload.py)中 `SERVING_OPERATORS` 的 7 个 serving 边界.
内部阶段是这些边界的组成部分, 不是独立的 serving 或验收边界.

| Serving 算子 | 完整计时操作 |
|---|---|
| `gdn_attention` | GDN 输入, recurrent 或 prefill 计算及输出 |
| `fa_attention` | FA 输入, attention, gate 及输出投影 |
| `ffn` | Gate/up 投影, activation 及 down 投影 |
| `sample` | 词表投影及采样. 适用时包含 MTP 接受, 拒绝采样及已接受状态 slot 选择 |
| `mtp_context` | MTP 输入投影及 KV-only 变体的 FA 输入 |
| `mtp_draft` | 适用时的 MTP 输入投影, 完整的一层 FA, FFN 及采样 |
| `vision_block` | 一个视觉层. 首层包含 patch 输入. 末层包含空间合并 |

第一个 MTP draft 使用 `query_only`. 其余 3 个 draft 使用 `full`.
第一个 draft 使用已就绪的 context 端点.

原生 provider 提供这些边界. 完整负载验收是必要的.
实现计划不提供达到 60% 性能门槛的证据.

## 内部语言阶段

| 内部阶段 | 计算 |
|---|---|
| `gdn_input` | 适用时的 embedding 和 media gather, residual RMSNorm, FP8 投影, BF16 gates, 卷积, SiLU, Q/K norm, decay 和 beta |
| `gdn_recurrent` | Decay, delta 更新, 输出, 验证私有状态 |
| `gdn_prefill` | 分块 scan, 输出, 对齐状态 checkpoint |
| `gdn_output` | Head RMSNorm, SiLU gate, 激活量化, 输出投影 |
| `gqa_input` | Residual RMSNorm, Q/gate/K/V 投影, head norm, MRoPE, 分页 KV 写入 |
| `gqa_attention_output` | 因果 GQA, sigmoid gate, 激活量化, 输出投影 |
| `ffn_gate_up` | Residual RMSNorm, 激活量化, 打包 gate/up 投影 |
| `ffn_down` | SiLU 乘积, 激活量化, down 投影 |
| `output_sample` | 最终 norm, 词表投影, penalties, grammar, 贪婪或随机选择 |
| `mtp_input` | Embedding gather, 两个 norm, 通过投影完成 concat |
| `mtp_accept_commit` | 条件接受, 拒绝采样, 已接受状态选择 |

MTP 与 target 模型使用相同的 FA 和 FFN 算子.
无需为这些算子维护第二份实现.

## 内部视觉阶段

| 内部阶段 | 计算 |
|---|---|
| `vision_patch_input` | Resize, 归一化, patch 投影, bias, 位置插值, patch 顺序 |
| `vision_qkv` | Residual LayerNorm, Q/K/V 投影, bias, 空间 RoPE |
| `vision_attention_output` | 帧内非因果 attention, 输出投影, bias |
| `vision_ffn_up` | Residual LayerNorm, 第一次投影, bias, GELU |
| `vision_ffn_down` | 第二次投影及 bias |
| `vision_merge_up` | 最终 residual LayerNorm, 空间 gather, 第一次投影, bias, exact GELU |
| `vision_merge_down` | 第二次投影及 bias |

## 实现选择

阶段表格给出 serving 边界内部必要的计算.
实现可以融合阶段. 每个计时操作必须提供对应 serving 边界的全部阶段.

在能降低完整操作耗时的位置, 将 producer 计算融合到 GEMM 输入加载或输出存储中.
必要时分别实现小行数与 prefill 路径.

小型 norm, activation, quantization 和位置操作只能包含在完整 serving 算子中.
对应参考函数只保留在评测工具中.

一次完整操作可以包含多个 launch.
计时包含所有 launch, copy, reduction 和临时存储.

使用相同输入及状态, 比较 fusion 与完整未融合链.
仅保留正确且降低完整操作耗时的 fusion.

不通过从计时操作中移除 launch 来提高效率.
验收使用[评测流程](evaluation.zh.md).
