# 物理上界

## 范围

任何 operator family 和 execution mode 都尚无低于 0.50 的合格物理上界.
普通门槛为 eager execution 和 CUDA Graph replay 各达到 50%.
只有完整物理分析证明 50% 不可达后, 才适用 75%-of-U 替代条件.
还必须具备完整 numerical coverage, 且测得的 family efficiency 至少达到 0.75U.
该替代条件本身不构成 operator acceptance.
[24 个 HTTP configuration](../performance.zh.md) 各自有硬性 50% 门槛.

[Workload](../../development/kernels/workload.py#L323)<!-- source: def modeled_cases(profiles=PROFILES): --> 为七个 serving boundary 提供 75 个 case.
两个 execution mode 对应 14 个 family-mode 决策.
遵守[评估流程](evaluation.zh.md) 中原始 work, precision, named tolerance, cold-weight policy 和 serving boundary.

## Family 计算

对于 case i, R_i 是固定的 roofline 时间, T_i 是实测时长, calls_i 是调用次数.
[Work calculation](../../development/kernels/work.py#L50)<!-- source: def work_for(shape, *, queries=None, state_writes=None): --> 定义 useful computation 和必要的 boundary traffic.
[Aggregator](../../development/kernels/roofline.py#L61)<!-- source: def expected_efficiencies(observations, required_cases, source_id): --> 使用:

`p_i = calls_i / sum_j calls_j`

`eta_family = sum_i p_i * (R_i / T_i)`

p_i 的分母只对同一 operator family 求和. 总时间之比是另一种 metric.

物理证明必须为全部 case 和选定 execution mode 给出有效的正 time floor:

`T_i >= L_i > 0`

`eta_family <= U_family = sum_i p_i * R_i / L_i`

在 reference accounting model 中, 选择 `L_i = R_i` 可得 `U_ref = 1`.
这一选择不是物理证书, 不能证明普遍适用的 physical ratio 不超过一.
不得将 physical `R_i/L_i` 截断到一.

使用固定的 dense B200 peak 和[中间精度](model.zh.md#中间精度).
Tensor precision time 在共享 resource 上相加. 独立 resource floor 使用 `max` 组合.
只有强制串行执行的证明, 才能支持将完整 memory time 与 compute time 相加.
Padding, recomputation 和不必要的 copy 不能增加 useful work.
全部必要 launch 和 temporary traffic 都属于 measured operation.

## GDN state path

[Paired state recurrence](../../python/oh_my_vllm/kernels/cuda_backend/gdn_value_pair.cuh#L64)<!-- source: gdn_value_pair_kernel(const __nv_bfloat16 *qkv, __nv_bfloat16 *pool, const int *starts, --> 在原始 rounded FP32 graph 中有如下最长路径:

| Operation | 连续 rounded node 数 |
|---|---:|
| Decay multiply | 1 |
| Ordered recall FMA | 4 |
| Original reduction level | 5 |
| Subtraction | 1 |
| Beta multiply | 1 |
| State-update FMA | 1 |
| 合计 | 13 |

RN 表示舍入至最近值, 距离相同时取偶数.

这 13 个 node 给出源码依赖深度, 不是通用 native instruction count 或 duration.
独立的 batch, head, value-row 和 packed-component chain 不会乘大此路径.
Output dot product, head RMS 和 output gate 不反馈到下一个 state.
Prefill Q/K normalization 和 beta/decay 在 carry 之前计算.
[已准备的 operand](../../python/oh_my_vllm/kernels/cuda_backend/gdn_kernels.cuh#L203)<!-- source: __device__ inline void gdn_prefetch_rows(unsigned char *storage, const __nv_bfloat16 *qkv, --> 已提供 scaled Q 和 gate coefficient.
BF16 checkpoint store 不会替换 live FP32 carry, 也不能证明下一 token 之前必须完整等待 HBM.

设 q_i 为 query length, tau 为每个 rounded parent edge 经认证的 minimum delay.
条件性的 span 为:

`span_i = 13 * q_i * tau`

最长 modeled query 为 32144 个 token, 此路径有 417872 个 node.
没有经认证的 edge delay 或 clock cap 支持该路径的 physical duration.
证明还必须处理合法的其他 realization 和 input-dependent shortcut.

Workload 给出的 q1 probability 为 0.737516005121639, q5 probability 为 0.24583866837387963, prefill probability 为 0.016645326504481434.
q1 份额含 embedding case. 仅限制 prefill, 会留下 0.9833546734955185 的 reference 余项.
该余项不表示可达到的效率.

全部 24 个 GDN case 的条件方程为:

`sum_i p_i * R_i / max(R_i, 13*q_i*tau) = 0.50`

其解为每个 rounded edge 的 tau=1.96191198467 microseconds.
它是证明必须达到的 threshold, 不是 measured 或 assumed hardware latency.
在 time floor 获得独立物理依据之前, 该方程不能提供 physical U.

## 物理证明的证据

当前 register, spill, occupancy, tile, layout, launch count 和 KV-slot 选择只是某一种实现的属性.
它们不限制全部合法实现.

最快 observed runtime 不小于未知 optimum, 不能作为 physical lower time bound.

Clock reading, event quantum, host gap 和未拟合的 model unit 不能提供缺失的证书.
Packed FP32 语义不能固定 minimum native delay.
每个 case 的 useful work 较低, 不能证明 hardware impossibility.
Normalization path 也没有经认证的 delay 或 clock cap.

未来的证明有四部分:

1. 绑定 shape, call, work, precision 和 source identity.
2. 为各 execution mode 的全部合法实现及适用输入证明 time floor.
3. 计算按 invocation 加权的 family U, 不得截断.
4. 只有不可达证明与完整 numerical 和 timing coverage 齐备后, 才应用替代条件.

证明之后的测量遵守[测量规则](evaluation.zh.md#测量), 它们不改变物理证明条件.
