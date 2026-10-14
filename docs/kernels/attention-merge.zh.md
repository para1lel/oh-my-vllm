# 注意力合并契约

[自有注意力源码](../../python/oh_my_vllm/kernels/cuda_backend/operators/decode_attention.cuh)按权重合并部分注意力输出.
保留 FP32 累加, BF16 边界, 调用流和项目容差. 空头精确输出零.

[合并测试](../../tests/numerical/test_attention_merge_replay.py)使用独立 FP64 参考.
测试覆盖抵消, 主导和空分段, 输出哨兵, 输入不变, 以及改变输入后的图重放.
各源码范围的数值报告和待完成整模型检查见 [验收索引](../acceptance.zh.md).
