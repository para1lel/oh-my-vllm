# 当前状态与后续工作

更新日期: 2026-10-13.

## 实现

Rust 管理 HTTP 服务, 请求调度和逻辑 KV Cache. Python 管理一张 B200 GPU.
使用 784-token block, 16 个 FA 层, 48 个 GDN 层和 `mamba_cache_mode="align"`.
输入与输出的总长度上限为 262144 token.

普通解码, MTP4 和 DSpark 在启动时选择.
Chat, Responses, 流式输出, 工具, thinking 设置, 约束生成, 取消和前缀复用均保持可用.
DSpark 保持固定权重, 使用目标层特征 `(5,19,33,47,61)`.
Markov 候选和累计置信度选择零到七个候选, 初始阈值为 `0.2`.
随机采样使用 `min(1,p/q)` 接受概率, 拒绝后从归一化的 `max(p-q,0)` 采样, 全部接受后增加一个目标 token.

CUDA Graph, 多 stream 和 PDL 控制模型计算.
所有自有 CUDA backend 代码文件均符合 800 行上限. pre-commit 使用 clang-format 23.1.3.
头文件内容进入构建 key, 加载出处, 正式契约和 profiler 检查.

前缀 identity 在 worker 状态经过校验并提交后进入查询.
准入 lease 对排队请求, 已准备请求, 编码历史, 输出状态和已完成 stream drain 具有数量与字节上限.
HTTP supervisor 在故障后使旧逻辑 KV 失效, 并加载替换 worker.
受影响请求停止并报告错误. 恢复期间 HTTP 保持存活.

服务和请求支持数值 `thinking_token_budget`, 默认 `-1` 表示不单独限制.
达到内容预算后, target mask 强制生成 `</think>`, 然后继续生成. Effort 和总量限制继续生效.
OMP 的 Chat, Responses 和自动压缩请求省略此字段, 由服务默认值统一覆盖.
[ADR-012](decisions/ADR-012-reasoning-token-budget.zh.md) 记录引擎比较和推测解码契约.

Worker 单步期限使用可配置的正数, 默认 120 秒.
该期限包含首次编译和图捕获.

编译器元数据具有数量与字节上限. 图缓存的驻留条目具有数量上限, 并检查捕获所需的剩余显存.
编译器退休在安全的主机边界启动垃圾回收, 并保留驻留图重放所需的 owner.
Prefill 准入按计划存储量计算. 图缓存压力驱逐释放不再使用的 pool owner.
DSpark 图重捕获在 shape 反复变化时使用冷却期.

Shutdown 释放资源 owner, 然后 bridge 退出进程, 不再执行末尾 allocator flush.
正常图压力淘汰保留 allocator 清理. Rust shutdown deadline 保持 `30` 秒.

全部七个捕获入口在捕获前同步并清除缓存的 cuBLAS 工作区.
这使新工作区属于对应图的 pool, 同时保留工作区预算和数值容差.
启用图的启动要求启用 native caching allocator, 且不能设置 `max_split_size_mb`.
不支持的显式配置会被拒绝, 原值不改写. 显式 eager 模式仍可使用.
[ADR-013](decisions/ADR-013-cuda-graph-workspace-ownership.zh.md) 记录生命周期修复及其范围.

长 prefill 预算分配在分配 prefill 前, 为活动 decode 输入预留额度.
至少两条符合条件的长 prefill 共享按 block 对齐的额度, FIFO 准入与队尾抢占不变.
`prefill_start` 将未命中区间的分类保持到最终 chunk, 抢占后重置.
四条不复用前缀的长请求每个完整 step 各使用 `7840` token. 短提示词保持普通路径.
[ADR-014](decisions/ADR-014-long-prefill-budget-sharing.zh.md) 记录失败负载与新分配方式.

[架构](architecture.zh.md) 规定所有权与缓存契约.
[服务契约](serving.zh.md) 规定预算, worker 恢复和请求计时.
[开发](development.zh.md) 说明可移植环境选择. 本机配置保存在忽略的 `LOCAL.md` 中.

## 验证

持久会话采集器保留 4 个 OMP 进程和历史, 最高 effort 和运行时 context usage.
它保存增量 journal, 并将 provider-input 锚点与服务计数核对.
正式验收还需独立资源审查.
采样器在查询 GPU 进程前保存失败的 worker 观测.
采集器在最终检查和证据哈希计算前停止采样器.

工作区修复在源码 `22dd754` 上通过八项必需检查.
其 CPU 套件通过 1389 项测试和 116 项子测试, 排除 449 项 GPU 案例.
十六项 GPU 工作区回归通过, 排除一项主动 allocator-OOM 测试.
四十二个 checkpoint context 案例提供 210 次层比较, 最大绝对误差为零.
这些 context 诊断使用合成 target 特征, 未执行完整 target, scheduler 或 HTTP 负载.

源码 `22dd754` 通过六十项 reasoning 预算 GPU 案例和全部六项无预热的普通解码 / MTP4 完整 context 案例.
预算案例覆盖普通解码, MTP4, DSpark, 两种 API, 流式输出, JSON, 工具, 随机采样和服务默认值.
预算案例的 DSpark 置信度阈值为零. 正式阶段和服务负载使用 `0.2`.
容量案例在 batch `1/2/4` 下完成输入 `258048` 加输出 `4096`, 无 OOM 或重算抢占.
独立审查核对原始记录, 源码身份, 输出和自有进程退出.

`22dd754` 完成十三项阶段: 十二项通过, 长 MTP batch `4` 的 decode 倍率未通过.
失败行中位数为 `39.9496` 秒, 下界为 `12.2634` 秒, 倍率为 `3.2576`.
它的 prefill 和离散度门槛通过. 上限保持为 `3`.

普通解码 batch `2` 被中断, batch `4` 未启动.
记录保留原状态, 不能证明十五行验收通过.

首轮持久 MTP4 尝试因普通正文触发 OMP 重复输出检查而失败.
首轮 DSpark 尝试达到采集器 `1800` 秒 turn 期限, 没有 GPU 或工具故障.
后续尝试改用不复述分隔符的提示, 采集器 turn 期限为 `7200` 秒.
它们在一小时前为必需的 scheduler 修改而取消.
全部采集器和自有 GPU 进程已退出. 失败与取消记录保持不变.

长 prefill 候选版本通过 53 项 scheduler 测试, 覆盖持续推进, 尾段, decode 预留, 前缀复用和暂停后的 slot 限制.
十八项静态用例专项测试通过, 保留全部原有 `317` 项配置, 并确定 `200` 项新增案例.
八项必需检查和 pre-commit 检查通过. Workspace 有 `152` 项 Rust 测试.
教程通过十项 excerpt 测试和九项浏览器测试.

较早的 CPU 尝试在行为断言前, 因 HTTP fixture 首次就绪失败而停止.
保留日志显示其中一次尝试在 `15` 秒 timeout 前没有建立 ZMQ peer 连接.
Fixture 现使用已有的 `30` 秒首次就绪 timeout. 请求, 恢复与关闭 deadline 保持不变.
完整 CPU 套件重跑通过 `1398` 项测试和 `116` 项 subtest, deselect `455` 项 GPU 案例.

完整算子矩阵有 `517` 项. 新算子, 阶段, 容量, 预算和持久会话 GPU 验收待完成.
发布前在已提交源码上重做适用检查.

`4100d1a` 的新阶段与算子尝试因外部 GPU 进程而停止, 没有完整结果行.
容量尝试生成了普通 batch `1` 的完整输出, 无 OOM 或抢占, 随后 worker shutdown 失败.
GPU 套件在 GPU 查询 timeout 后停止. 六项边界测试失败, 未保存 pytest 异常报告.
这些尝试不提供正式验收. 所有所属进程已退出.

独立小检查表明, 内层 GPU 检查会拒绝父 pytest 进程此前建立的 CUDA context.
完整测试 launcher 现在分别用两个新进程执行边界模块与其余测试.
独立 CPU 收集检查保留了原始 node multiset, 显式 selector, 重复次数, 空分组与失败证据.

Shutdown 栈检查停在资源 owner 释放后的 `torch.cuda.empty_cache()`.
仅跳过此末尾调用的诊断完成了普通 batch `1`, 输入 `258048` 加输出 `4096`.
它保留 `30` 秒 shutdown deadline, 没有所属进程存活.
该诊断使用保存的 cache, 允许外部 GPU 工作. 它不提供正式容量或性能验收.

Shutdown 与测试进程改动通过八项必需检查和 pre-commit 检查.
完整 CPU 套件通过 `1406` 项测试和 `116` 项 subtest, deselect `455` 项 GPU 案例.
收集与生命周期检查通过 `23` 项测试. 教程通过十项 excerpt 测试和九项浏览器测试.

下一轮持久采集每种模式保留四个会话, 每条至少 `3600` 活动秒, 中途不排空.
使用服务 reasoning 预算 `8192` 和直接行动的审计提示, 不设置精确工具调用次数配额.
提示以名称指代模型分隔符, 不复述字面拼写.
采集器 turn 期限为 `7200` 秒. Worker 单步和服务请求期限保持为 `600` 和 `1800` 秒.
首次编译与捕获计入负载, 不单独预热.

Typed inventory 仅在 receiver 为 Tensor 且结果为整数时, 将 Tensor `numel` 识别为主机 metadata.
旧 prefix 失败记录没有此类型证据, 仍被拒绝.
[验收](acceptance.zh.md) 标明此前测量范围. 此前结果不能证明调整后的负载已通过.

## 后续工作

- 扩展 `PY-04`, 在接近最大支持 context 时验证跨 shape 图驱逐与重捕获.
- 改动有依赖的主机等待前, 测量 `PY-06` 的主机重叠.
- 依据固定容量契约, 检查 shape 持续变化时的编译器存储.
- 保持重型 soak 显式执行, 运行时变动后重测受影响阶段负载.
- 完成教程的用户审阅.

## 教程预览

用户要求保留的教程预览使用端口 18084.
忽略的 `LOCAL.md` 保存地址, 进程身份和维护命令.
教程包含十三个中文章节, 引用当前项目源码.
