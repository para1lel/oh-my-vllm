# 术语参考 A

| 术语 | 词性 | 项目含义 |
|---|---|---|
| `inference` | 名词 | 产生输出 token 的模型计算. |
| `token` | 名词 | 输入或输出中的一个 vocabulary ID. |
| `checkpoint` | 名词 | 模型权重 / 配置, 或 token 边界保存的 recurrent state. |
| `prefill` | 名词 | 处理 prompt 或重计算历史的模型计算. |
| `decode` | 动词 | 从已接受前缀计算下一 target 输出. |
| `scheduler` | 名词 | 选择 token 工作和逻辑 slot 的 Rust 组件. |
| `scheduling` | 名词 | 在 token 和 sequence 预算下选择请求工作. |
| `cache` | 名词 | 为继续计算或前缀复用保存的 KV 或状态. |
| `cache` | 动词 | 保存已计算数据以供后续复用. |
| `KV` | 名词 | Attention key 和 value. |
| `FA` | 名词 | 针对适用 token 前缀的 full attention. |
| `GQA` | 名词 | KV head 少于 query head 的 grouped-query attention. |
| `GDN` | 名词 | GatedDeltaNet recurrent 计算及状态. |
| `MTP` | 名词 | 带草稿 proposal 和 target verification 的 multi-token prediction. |
| `MTP4` | 名词 | 提出 4 个草稿 token 的 MTP 模式. |
| `draft` | 名词 | 未验证 token proposal 或提供它的模型. |
| `bonus` | 名词 | 已接受 speculative prefix 后的 target token. |
| `slot` | 名词 | Group 内 tensor 的逻辑块或状态位置. |
| `page` | 名词 | 保存 784 个 target token 的持久 attention 存储. |
| `subpage` | 名词 | Target page 内 16 个 token 的 native attention view. |
| `batch` | 名词 | 一起计算的请求或 token row. |
| `batching` | 名词 | 在一个 step 中组合活跃请求工作. |
| `logits` | 名词 | 采样前模型对输出 vocabulary 的分数. |
| `sampling` | 名词 | 按模型分数和配置约束选择 token. |
| `sample` | 动词 | 按配置的分布选择 token. |
| `grammar` | 名词 | 思考控制, JSON 答案或函数参数的 token 级约束. |
| `mask` | 名词 | 采样前使用的允许 token 信息. |
| `tensor` | 名词 | 具有 device 和 layout metadata 的类型化多维数组. |
| `dtype` | 名词 | Tensor 元素表示, 例如 BF16 或 FP32. |
| `stride` | 名词 | Tensor 某维相邻元素之间的存储偏移. |
| `metadata` | 名词 | 与 tensor value 分离的配置和 shape/state 描述. |
| `provider` | 名词 | 语义算子的已注册实现. |
| `IR` | 名词 | 语义 GPU 算子的 intermediate representation. |
| `DSL` | 名词 | 用于项目语义算子的 domain-specific language. |
| `schema` | 名词 | 数据字段或算子 mutation 的显式合同. |
| `mutation` | 名词 | 改变持久 tensor 存储的算子 effect. |
| `donation` | 名词 | 将已证明临时值的存储复用于其他输出. |
| `graph` | 名词 | 带持久输入以供 replay 的 CUDA launch 序列, 或编译器算子表示. |
| `capture` | 动词 | 将 GPU launch 记录到 CUDA Graph. |
| `replay` | 动词 | 使用当前输入启动先前 capture 的 CUDA Graph. |
| `lower` | 动词 | 将语义 IR 节点替换为选定 provider 算子. |
| `compile` | 动词 | 将模型或内核代码转换为可执行 GPU 程序. |
| `compilation` | 名词 | 将模型或内核代码转换为可执行程序. |
| `allocate` | 动词 | 为计算预留逻辑 slot 或 tensor 存储. |
| `allocation` | 名词 | 逻辑 slot 或 tensor 存储的预留. |
| `preempt` | 动词 | 释放请求逻辑块, 将已接受历史重新排队以重计算. |
| `preemption` | 名词 | 逻辑缓存压力下请求退出当前运行集合. |
| `evict` | 动词 | 删除可复用 cache 或 graph 条目以释放容量. |
| `rollback` | 名词 | 拒绝后恢复 computed progress 或 speculative state. |
| `validate` | 动词 | 按数据, schema 或算子合同比较输入, 拒绝无效输入. |
| `validation` | 名词 | 按数据, schema 或算子合同比较输入并拒绝无效输入. |
| `verify` | 动词 | 接受 / 拒绝 speculative token, 或对照独立参考比较输出. |
| `verification` | 名词 | Speculative 接受决策或独立输出比较. |
| `configure` | 动词 | 设置软件运行参数或环境选择. |
| `configuration` | 名词 | 显式软件参数和环境选择. |
| `tokenize` | 动词 | 将文本转换为 vocabulary ID. |
| `detokenization` | 名词 | 将 vocabulary ID 转换为输出文本. |
| `serialize` | 动词 | 将数据编码为 msgpack 或 JSON 字节. |
| `dispatch` | 名词 | 主机选择和 launch 内核实现. |
| `quantization` | 名词 | 使用显式 scale 和舍入转换为 FP8 值. |
