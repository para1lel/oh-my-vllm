# ADR-009: 可选 DSpark 模式

状态: 有效. [验收索引](../acceptance.zh.md) 记录测量源码身份及范围.

## 背景

原生 MTP4 通过 target 模型的 MTP 层提供 4 个草稿 token.
使用本地 DSpark checkpoint.
两种 HTTP API 必须在此模式完成真实 OMP 工具任务.
所有模式使用阶段时延, 精度和服务门槛.
上下文容量门槛覆盖普通解码和 MTP4.

DSpark 使用 5 层 BF16 GQA, 从 0 开始编号的 target 层输出 `(5,19,33,47,61)`, 以及 Markov / confidence head.
至多提出 7 个 token, 由 target 使用 8-row verification.
草稿 block7 与 target block784 不同.

## 一手参考

[DSpark 论文](https://arxiv.org/abs/2607.05147) (v1, 2026-07-06) 与 [DeepSpec](https://github.com/deepseek-ai/DeepSpec) 提供算法参考.
[Qwen3.8-27B-DSpark model card](https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark) 说明 checkpoint.
本项目支持范围以实际加载的本地配置 / 权重 SHA-256 身份为依据.
发布方测量使用其硬件 / 运行时配置, 本项目验收必须使用当前源码在 B200 上的独立记录.

## 决策

增加显式 worker 模式 `none`, `mtp`, `dspark`.
保留旧 0 / 4-token CLI 行为.
DSpark 通过 `OH_MY_VLLM_DRAFT_MODEL` 或 `--draft-model` 指定独立 checkpoint 路径.
执行前校验固定架构及全部 BF16 tensor 名称 / shape.
加载的 tensor 绑定稳定的配置 / 权重文件哈希.

Rust 保留调度, 分配, 前缀复用和回滚职责.
Python 通过项目内模型和 IR 代码计算 target feature 与 DSpark proposal.
项目构建, 测试和推理保持独立运行时合同.
不执行 checkpoint 内的 Python 代码.

草稿共享 target embedding 和 vocabulary 权重.
Context KV 使用不同的物理存储, 其 FA page ID 由 Rust 分配.
只有接受的 target 输入行进入该 context.
临时双向 proposal KV 保留在持久 context 之外.
GDN pool 保持 BF16 推测状态及既有舍入约束.

随机采样使用完整条件 proposal 概率, 接受规则为 `min(1,p(x)/q(x))`.
拒绝时使用归一化的 `max(p-q,0)`, bonus token 使用 target 概率.
Target 和 proposal 分布均应用用户采样参数及 grammar mask.
使用独立草稿随机数生成器, 保持 target 生成器所有权.

Confidence 可缩短前缀, 默认阈值为 `0.2`.
首项 confidence 乘积低于阈值时返回 0 个 candidate, 该步由目标执行单 token 解码.
设置为 `0.0` 保留至多 7 个固定数量 proposal. 输出, 上下文和 grammar 限制可减少 proposal 数量.
该设置来自文本 / 工具历史的小样本实验, 输入与正式输入不同.
[交接](../handoff.zh.md) 记录其范围.

target graph 与草稿 proposal graph 使用不同内存 pool.
使用 `fullgraph=True` 编译 target feature, context injection, backbone, Markov step 和 greedy proposal.
保持显式 provider 失败和语义 mutation 合同.

worker 启动时选择一种模式. 在 worker 停止前保持该模式.

## 验收

每种模式使用阶段时延流程.
接受率为接受草稿总数除以已调度草稿总数.

保留完整成功, 失败和中断尝试, 记录源码, 二进制, 环境, 实际加载 checkpoint 和配置身份.
记录实际已调度草稿数, 接受数, compile / capture 审计, 容量和显存峰值.
返回的下一步 proposal 不能作为接受率分母.
可移植摘要继续作为派生证据.

检查 DSpark 两种 API 的约束, 混合 batch, 取消和长上下文前缀复用.
OMP 必须读取两个源码文件, 将结果回传后续模型请求, 随后生成最终答案.
通过 response ID 匹配实际模式的服务日志, 单独检查答案语义.

原 TileLang 文件, manifest 哈希和用例 ID 保持不变.
新增独立补充参考及静态推导的 DSpark 算子用例.
这些用例沿用原有精度及 3 轮算子速度 / 置信门槛.

## 替代方案与影响

仅 greedy token 相等无法提供随机采样接受语义.
完整条件概率增加 proposal 存储和采样工作.
实现保留这些工作, 以正确处理配置的随机输出.
最终性能声明由实测验收确定.

参见 [需求](../requirements.zh.md) 和 [测试](../testing.zh.md#框架性能).
