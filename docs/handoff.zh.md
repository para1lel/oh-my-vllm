# 当前状态与后续工作

更新日期: 2026-10-12.

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

全部七个捕获入口在捕获前同步并清除缓存的 cuBLAS 工作区.
这使新工作区属于对应图的 pool, 同时保留工作区预算和数值容差.
启用图的启动要求启用 native caching allocator, 且不能设置 `max_split_size_mb`.
不支持的显式配置会被拒绝, 原值不改写. 显式 eager 模式仍可使用.
[ADR-013](decisions/ADR-013-cuda-graph-workspace-ownership.zh.md) 记录生命周期修复及其范围.

[架构](architecture.zh.md) 规定所有权与缓存契约.
[服务契约](serving.zh.md) 规定预算, worker 恢复和请求计时.
[开发](development.zh.md) 说明可移植环境选择. 本机配置保存在忽略的 `LOCAL.md` 中.

## 验证

持久会话采集器保留 4 个 OMP 进程和历史, 最高 effort 和运行时 context usage.
它保存增量 journal, 并将 provider-input 锚点与服务计数核对.
正式验收还需独立资源审查.
采样器在查询 GPU 进程前保存失败的 worker 观测.
采集器在最终检查和证据哈希计算前停止采样器.

Rust 测试, 格式, 行宽, Clippy 和 CUDA 格式检查通过.
CPU 套件通过 1389 项测试和 116 项子测试, 排除 449 项 GPU 案例.
工作区回归通过十六项 GPU 测试, 排除一项主动 allocator-OOM 测试.
生产捕获边界通过四十二个 checkpoint 上下文案例, 210 次层比较, 最大绝对误差为零.

工作区修复前, reasoning 预算 GPU 套件在源码 `ee05583` 上通过六十个案例.
它覆盖普通解码, MTP4, DSpark, 两种 API, 流式输出, JSON, 工具, 随机采样和服务默认值.
这些案例的 DSpark 置信度阈值为零. 该结果不能证明阶段或持久会话验收通过.

新的 MTP4 和 DSpark 一小时采集以及调整后的十五项阶段负载待完成.
下一次 DSpark 采集因 CUDA 非法内存访问失败. 同时运行的 MTP4 与阶段尝试已取消.

同步诊断将故障定位到驻留的 29 行上下文图.
同 stream 工作区生命周期复现在无关图 reset 后失败.
私有 pool 修复通过四十二个真实 checkpoint 上下文案例, 210 次层比较, 最大绝对误差为零.
该诊断使用合成 target 特征, 未执行完整 target, 调度或 HTTP 负载.

全部采集器及所属 GPU 进程已退出. 原始失败与取消记录保持不变.
在修复后的已提交源码上重新测试预算套件, 十五项阶段和两组持久会话采集.

用户要求在预算修改前停止验收. 剩余阶段尝试已取消, 所属进程已退出.

下一轮采集使用已提交源码, 服务预算 `8192`, 以及直接行动的审计提示, 不设置精确工具调用次数配额.

之前的冷启动采集因 worker 单步期限失败, 原始记录保留失败状态.
采集器使用 600 秒单步期限和 1800 秒请求期限, 无单独预热.

此前测量源码的十二项阶段结果通过门槛, 保留原始源码和 binary 身份.
它们不能证明思考预算修改后的版本通过. 更新源码后重新测试十五项负载.
Typed inventory 仅在 receiver 为 Tensor 且结果为整数时, 将 Tensor `numel` 识别为主机 metadata.
旧 prefix 失败记录没有此类型证据, 仍被拒绝.

短诊断及取消尝试在外部证据中保留原有状态.
[验收](acceptance.zh.md) 标明之前测量的范围, 这些结果不能证明修改后的负载已通过.

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
