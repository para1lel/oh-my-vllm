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

编译器元数据具有数量与字节上限. 图缓存的驻留条目具有数量上限, 并检查捕获所需的剩余显存.
编译器退休在安全的主机边界启动垃圾回收, 并保留驻留图重放所需的 owner.
Prefill 准入按计划存储量计算. 图缓存压力驱逐释放不再使用的 pool owner.
DSpark 图重捕获在 shape 反复变化时使用冷却期.

[架构](architecture.zh.md) 规定所有权与缓存契约.
[服务契约](serving.zh.md) 规定预算, worker 恢复和请求计时.
[开发](development.zh.md) 说明可移植环境选择. 本机配置保存在忽略的 `LOCAL.md` 中.

## 验证

持久会话采集器保留 4 个 OMP 进程和历史, 最高 effort 和运行时 context usage.
它保存增量 journal, 并将 provider-input 锚点与服务计数核对.
正式验收还需独立资源审查.

Rust 测试, 格式, 行宽, Clippy 和 CUDA 格式检查通过.
CPU 套件通过 1307 项测试和 99 项子测试, 排除 448 项 GPU 案例.
新增上下文与压缩证据回归测试通过.

新的 MTP4 和 DSpark 一小时采集以及调整后的十五项阶段负载待完成.
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
