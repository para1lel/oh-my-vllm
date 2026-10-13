# ADR-015: 公共服务采样默认值

状态: 生效. 日期: 2026-10-13.

## 问题

`249bbb0` 上的持续 MTP4 测试在 `2379.1744` 秒后失败.
OMP 检测到重复的普通文本. Reasoning 通道没有该重复周期.
普通文本连续出现 `23` 次结束标签序列.
服务在取消时记录 `5035` 个生成 token, 尚未达到显式服务思考预算 `8192`.

这不是总上下文耗尽. 输入与预留输出为 `150941 + 32768` 个 token, 低于 `262144`.

原始请求省略了采样字段. 服务已使用 checkpoint 的 temperature `1`, top_p `.95` 和 top_k `20`.
Presence penalty 为 `0`, repetition penalty 为 `1`.

这些已支持的值符合 Qwen 公布的思考模式默认值.
在生成切换到普通文本前, 之前的 grep 结果包含思考结束标记.
输入可能有影响. 这一观察不能确定重复输出的原因.

源码 `98a6eb1` 使用公共 presence penalty `0.5` 和生产思考预算 `-1`.
其 MTP4 持续测试在 `1276.9908` 秒后失败.
OMP 在普通文本中检测到连续四次重复的 `257` 字符周期.
Reasoning 通道没有该周期.

输入与预留输出为 `92124 + 32768` 个 token. 服务在取消时记录 `1725` 个生成 token.
失败的 stream 没有终止事件或最终 SDK usage.

循环标记和重复文本由模型生成. 实际请求文本不含这些字面标记.
该重复的原因仍未知.
下一次参数修改前, 全部八项采集根进程均已回收.
原始失败和取消记录保留各自源码与状态.

## 决策

普通, MTP4, DSpark 和测试 HTTP 服务统一使用 presence penalty 默认值 `0.5`.
保持 checkpoint 的 temperature `1`, top_p `.95` 和 top_k `20`, repetition penalty 为 `1`, frequency penalty 为 `0`.
独立思考预算默认保持 `-1`. Reasoning effort, 总输出限制和上下文限制保留原有契约.

用 `serve --override-generation-config` 和 JSON object 配置支持的采样默认值.
Rust 按与请求采样相同的规则校验这些值.
拒绝未知键, null 默认值, 无效值及超过 `4096` 字节的配置.
CLI 校验发生在 worker 启动之前.

请求显式值优先于服务值. 服务值优先于 checkpoint 值.
请求采样字段为 null 时使用默认值. 显式 presence penalty `0` 关闭该惩罚.
修改其他服务字段不会移除公共 presence penalty 默认值.

OMP Chat, Responses 和自动压缩在省略采样字段时使用这些值.
持续验收使用生产默认值, 不另设采样或思考预算启动选项.
最大 effort 和要求的输出 / 上下文限制仍为显式工作负载设置.
预算边界测试和离线 greedy 测量保留各自规定的显式值.

服务 diagnostics 报告合并后的服务默认值. 请求 metric 报告服务和请求显式采样字段.
这些 metric 字段不含 Python 补充的 checkpoint 值.
CPU fixture 单独记录 checkpoint 合并后实际 Python 采样参数.

对生成 token 循环使用公共默认值 `repetition_guard=true`. 允许显式布尔覆盖值.
[服务契约](../serving.zh.md#采样-工具与停止) 定义有界规则和 grammar 回退.
该规则在 target 分布与 DSpark 提议分布的 top-k/top-p 之前应用.
离线固定长度 greedy 工作负载显式禁用它.
它改变重复输出, 包括有意重复, 且不保证任务完成.

## 官方指导与限制

[vLLM 引擎参数](https://docs.vllm.ai/en/stable/configuration/engine_args/#override-generation-config) 说明服务覆盖值与模型 generation configuration 的合并方式.
[Qwen 模型卡](https://huggingface.co/Qwen/Qwen3.8-27B#best-practices) 给出思考默认值, 并允许 presence penalty 在 `0` 到 `2` 之间调整.
Qwen 指出更高惩罚可能降低质量并导致语言混杂.
选定的 `0.5` 是该范围内的项目配置. 它不是 Qwen 固定推荐的默认值.

它改变随机生成概率. 需要新的推理和持续会话验收.

`presence_penalty` 和 `frequency_penalty` 应用于当前请求生成的 token, 包括前面的推测候选.
`repetition_penalty` 还使用提示词 token.

思考预算不能检测思考结束后的重复.
使用 presence penalty 后仍可能重复.
保留 OMP 原始检测器, semantic progress timeout, 请求, usage 记录和失败证据.
项目代码, 依赖和 build cache 仍独立于 vLLM.

## 验证

Rust 测试检查公共默认值, 优先顺序, null 省略, 显式零, 边界和不支持的字段.
Rust HTTP/ZMQ 测试检查两种 API 和实际 Python checkpoint 合并.
原始集成测试在此选项尚不存在时失败, 实现后通过.
仍需完整 CPU, GPU, 预算, 阶段, 算子, 持续会话验收和独立资源验收.
