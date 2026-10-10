# 当前状态和开放工作

更新日期: 2026-10-10.

## 当前实现

Rust 控制服务, 调度和逻辑 KV. Python 控制单张 B200 上的 GPU 计算.
默认后端为 CUDA. 固定 TileLang 提供算子参考.
保持 block size 784, 16 个 FA 层, 48 个 GDN 层及 `mamba_cache_mode="align"`.
输入输出总长度上限为 262144 token.

普通解码, MTP4 和 DSpark 在启动时选择.
Chat, Responses, 流式输出, 工具, thinking 设置, 约束, 取消及前缀复用继续可用.
普通解码保留 FP32 GDN 状态. 推测模式保留 BF16 状态和回滚快照.

DSpark 保持固定权重, 共享目标 embedding 与词表头.
五层草稿模型读取从零编号的目标特征 `(5,19,33,47,61)`.
只有保留的目标行进入持久 context.

Markov 提议和累积 confidence 选择零至七个候选. 初始阈值为 `0.2`.
阈值设为 `0.0` 可做固定数量诊断. 输出, context 和 grammar 限制继续生效.
随机接受使用 `min(1,p/q)`, 拒绝后使用归一化 `max(p-q,0)`, 全部接受后从目标分布生成 bonus.

环境选择和命令见 [开发](development.zh.md).
设置 `OH_MY_VLLM_MODEL` 或 `--model`, DSpark 还需 `OH_MY_VLLM_DRAFT_MODEL` 或 `--draft-model`.
[架构](architecture.zh.md) 定义所有权, 状态及缓存契约.

## 当前性能工作

十五项收集器记录逐请求的提交, 首 token 和末 token.
prefill 与 decode 分别检查墙钟耗时, 理论下界及波动.
已审查的语义模型计入必需参数读取, 共享资源, 依赖, 有效形状, 状态写入与草稿工作.
源码契约现已包含自有累加器头文件. 本次身份更新保持成本方程与容差不变.

执行使用 CUDA Graph, 多 CUDA stream 和 PDL.
GDN 投影 / 卷积分支与独立 gates 分支汇合.
prefill 图恢复持久写入与动态 metadata.
捕获拓扑复用将诊断存储从 29.13 降至 7.82 GB. 新 prefill 捕获要求 32 GiB 空闲内存.

大型 gate/up 使用 K256 tile, 三级流水, 列主序 scale 和 swizzle 16.
K-major 参考保持 K128 tile 与五级流水.
624 至 2496 行的五种投影使用两个 block 的 cluster, M128/N128/K128 tile 与五级流水.
自有累加器先读取两个独立 TMEM tile, 再执行一次等待.
epilogue 按相同舍入直接将 FP32 累加值转为 BF16.
其他形状保持原分派.

残差 RMS 与 GDN gated RMS 在规定的 BF16 舍入点融合量化.
选中输出路径保留必需 KV 和边界特征.
调度器仅在当前 step 已成功分配至少一个 block 的 chunk 后, 延迟取整得到的一 token prefill.
decode 和推测验证保持进度. 生产 token budget 为 32768.

观测工具为每个实际 kernel 请求 21 类指标, 分开记录原生计数器与 TileFoundry 代表形状的 HIR 估计.
它保留每个 profiler worker 的输出, 已加载 CUDA 身份和进程身份.
计数器记录必须匹配选定的进程, 用例和后端.
[Profiling](profiling.zh.md) 解释测量结果与所选配置.

## 已验证的源码范围

[源码绑定记录](../bench/evidence/2026-10-10-current-source-acceptance.json) 适用于 `c27ff1a`.
十五项完整输出负载中十三项通过. 前缀命中 batch 1 和 2 的 prefill 失败.
prefill 倍率约为 3.790 和 3.162. 全部 decode 阶段及全部墙钟波动检查通过.
完整 317 项算子矩阵通过数值和速度检查.
九项总长度 262144 用例均通过, 重算抢占为零且无 OOM.
DSpark 和 MTP4 服务套件通过两种 API, 约束, 取消, 长前缀及真实 oh-my-pi 工具循环.

完整 Python 收集通过 1181 项测试和 65 项 subtest. 两项测试假设失败.
后续修正从已注册列表之外选择无效 variant, 并清除 GPU selector mock 继承的 GPU UUID.
修正后的专项测试通过.
当前 CPU 收集通过 753 项测试和 89 项 subtest, 排除 445 项 GPU 测试.
完整 GPU 重测仍待完成.

后续自有累加器生产路径通过 26 项检查.
覆盖分派边界, 五种形状, 变化的图数据和权重, PDL 开关, 侧流重放及输出边界.
18 项受影响的完整算子通过数值与速度检查.
这些筛选的未提交源码测量用于诊断. 当前完整源码重测仍待完成.
[累加器诊断](../bench/evidence/2026-10-10-fp8-accumulator-observations.json) 保留所选用例, 105 条 profiler 计数器记录, 被拒候选及源码身份.

[验收](acceptance.zh.md) 列出原始摘要, 源码限制及独立审查范围.
保留失败和中断的尝试. 过去源码的结果不能提供后续修改的验收结论.

## 文档与预览

英文 Markdown 遵循 ASD-STE100 Issue 9, 同步完整中文翻译.
文档和十三章 Twine 教程描述当前项目代码与契约.
可移植证据保留测量值, 源码身份, 原始摘要及限制.
主机配置与原始记录存放在忽略目录或仓库外.

用户要求的教程预览继续使用端口 18084.
忽略的 `LOCAL.md` 保存地址, 进程身份和维护命令.

## 开放工作

- 降低前缀命中 prefill 时延, 重测每项受影响的完整负载.
- 推理修改后完成当前源码的阶段, 算子, 容量, 服务及完整 Python 验证.
- 下次提交前完成独立证据 / 文档审查和必需构建检查.
- 源码修改后运行教程检查, 保持预览可用.
- 完成用户对教程的验收.
- 按 [审计](audit.zh.md) 的 `PY-04`, 在实测 262k 内存压力下测试跨形状图淘汰 / 再捕获.
- 证明形状持续变化时编译器存储仍有界.
- 用测量推进 `PY-06` 的主机重叠工作.
- `KRN-06` 被拒候选继续排除, 直到完整操作证据表明节省时间.

阅读 [要求](requirements.zh.md), [测试](testing.zh.md) 和 [审计](audit.zh.md).
