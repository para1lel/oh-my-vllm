# 性能契约

## REQ-PERF-001: 阶段时延

| 模式 | 输入 token | 输出 token | Batch sizes |
|---|---:|---:|---|
| ordinary, MTP4, DSpark | 32768 | 4096 | 1, 2, 4 |
| MTP4 prefix-hit | 32768 | 4096 | 1, 2, 4 |
| MTP4 | 131072 | 4096 | 1, 2, 4 |

这些组合构成十五行.
使用合成 token IDs, greedy 采样和固定输出数量, 忽略 EOS.
每次重复前重置前缀复用.
prefix-hit 行为每个请求预置 32144 个可复用 token.
整个负载保持驻留, 重算抢占次数为零.

为每个请求记录提交, 首个保留 token 和末个保留 token 的时间.
Prefill 从提交到首个 token. Decode 从首个 token 到末个 token.
每次重复分别取所有请求中最长的 prefill 区间和最长的 decode 区间.
Batch 内阶段交错时, 使用各请求自己的边界.
计入注册, 排队, 调度, Python, 传输和采样.
排除加载, 分词, HTTP 和预热.
独立报告整批耗时, 清理耗时, TPS, 接受率和 GPU 诊断.

除 prefix-hit prefill 外, 每个阶段的墙钟时延中位数最多为理论下界中位数的两倍.
墙钟时延的相对极差最多为 10%.
记录 prefix-hit prefill 的时延倍率, 墙钟时延和理论下界, 不设时延门槛.
其 10% 波动检查继续有效. Prefix-hit decode 保持两倍限制.

## REQ-PERF-002: 理论模型与测量

使用固定语义 DAG, 保留数据, token, 层和持久状态依赖.
独立分支可以并行.
CUDA stream 顺序不增加依赖.
为每个执行 step 记录有效 query, KV 长度, 状态访问, 草稿, 验证和回滚.
Kernel 拆分或融合时, 规范成本规则保持一致.

理论下界取以下限制中的最大值:

- 语义 DAG 的最长路径.
- 必需 HBM 流量除以官方单卡带宽.
- 每类共享执行资源的必需工作除以其官方峰值.

计入参数读取, scale, 模型计算, GPU 采样, KV / state 更新, 草稿和被拒绝的验证行.
使用各项运算的精度和执行资源.
算法执行七行 DSpark backbone 时, 七行全部计入.
Padding, 重复传输和无用计算单独记录为开销诊断.
这些工作不得增加理论下界.

使用单卡非稀疏 B200 峰值: FP8 Tensor 4.5 PFLOPS, BF16 Tensor 2.25 PFLOPS, FP32 75 TFLOPS 和 FP64 37 TFLOPS.
使用 HBM 带宽 7.7 TB/s.
其他已知执行资源使用公开指令速率.
记录硬件身份, 峰值来源, 运算规则和模型版本.
实测速率和经验效率系数不得增加下界.

允许理想缓存复用和融合后中间值驻留片上.
使用有限存储容量推导最少必需 HBM 流量.
在该放松模型中, 未公开的片上带宽视为无限.
识别全部 GPU 运算. 未识别的运算阻止验收.

每行执行两次完整预热和五次正式测量.
根据每次重复的执行调度重新计算下界.
Prefill 与 decode 独立应用以下两项检查:

```text
median(wall_time) <= 2 * median(theoretical_lower_bound)
(max(wall_time) - min(wall_time)) / median(wall_time) <= 0.10
```

Prefix-hit prefill 记录第一项检查的结果, 该结果不作为失败门槛.
第二项检查适用于每个阶段.

保留完整失败集和中断尝试, 附上原因.
调查失败后, 对受影响的负载重新执行两次完整预热和五次测量.
不得筛选样本.

测量中的编译, 图捕获, 干扰或源码变化使该测量失效.
每次尝试绑定源码, 二进制, 环境, CUDA module, 配置和 checkpoint 身份.
正式验证使用完整原始记录, 可移植摘要不能替代原始记录.
