# 目标模型概念

## Target 计算与 attention

目标模型结合 full-attention 层与 recurrent GDN 层.
Attention 可以读取适用的 KV 前缀; GDN 通过紧凑 recurrent state 保存历史.
两类访问模式说明了框架为何要协调两个 cache group.

[模型配置与 forward 单元](../../python/oh_my_vllm/models/qwen.py)定义维度, 权重和层顺序.
此指南不复制其验证或实现阈值.

FP8 投影组合量化值, 显式 block scale 和 FP32 accumulation.
BF16 转换点属于数值契约. 融合链必须保持这些转换点的可观察行为.
读取[投影语义](../../python/oh_my_vllm/ir/fp8.py)与
[归一化/投影语义](../../python/oh_my_vllm/ir/normalized_linear.py).

RoPE 依据 token 位置改变 Q/K 坐标. 长上下文需要正确的相位精度.
Q/K 准备和缓存写入必须指向相同逻辑位置.
[Attention 语义](../../python/oh_my_vllm/ir/attention.py)定义 reference 行为;
[attention 源码](../../python/oh_my_vllm/kernels/attention.py)定义执行选择.

[GDN 语义](../../python/oh_my_vllm/ir/gdn_prefill.py)定义 recurrent 更新和 checkpoint 输出.
因果卷积与状态恢复必须对应已接受 token 历史.
实现细节见 [GDN 执行](../../python/oh_my_vllm/kernels/gdn.py)与
[卷积](../../python/oh_my_vllm/kernels/convolution.py).

正式测试检查输出, 已写状态, 未修改存储, page 边界和 graph replay.
[精度需求](../requirements/core.zh.md#req-acc-001-数值精度)定义保持不变的容差.
