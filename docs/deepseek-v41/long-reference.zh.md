# DeepSeek 长上下文参考生成

## 范围

[生成器](../../development/model/deepseek_v41/reference_long_case.py) 提供两个提示词长度下的独立 CPU FP64 参考构建.
它不启用 DeepSeek 推理, 也不证明服务容量.
目前尚未固定端点产物.
数值和性能限值是 [阶段计划](plan.zh.md) 中的独立前置条件.

提示词重复 [短参考](reference.zh.md#短文本生成) 中的 16 个 token IDs.
它使用 65537 tokens, 不采用位置偏移, 截断历史或近似窗口重放.
第一个端点位于 16385 个提示词 tokens 之后.
该长度超过第 20 层候选池中每块八个位置的 2048 个块.

第二个端点位于 65537 个提示词 tokens 之后.
该长度超过原始的 65536-token 旋转位置编码上下文.
这些选定边界不能证明 1048576-token 服务容量或 64 个请求准入.

## 计算和状态

生成器使用已有目标模型, 以及解码到 FP64 的原始量化检查点值.
它不使用原生运行时算子或运行时激活量化.
[参考指南](reference.zh.md) 规定算术和逐层流式权重读取.

Prefill 默认使用最多 2048 tokens 的连续分块.
每个分块的已接受状态为下一个分块提供完整续算状态.
端点之前的最后一个分块在该端点结束.
分块之间不跳过或重复任何 token.
工具通过目标模型状态保留来源 KV, 局部窗口, 部分压缩器和 Engram 历史.
有界分块限制临时行存储, 而非总前缀存储.

生成器在每个端点保存全部 129280 个 prefill logits.
其 argmax 选择一个 decode token.
独立的 decode 分支从该端点的已接受状态开始, 并提供全部 129280 个续算 logits.
该分支不会成为下一个端点提示词的一部分.
产物包含四个 FP64 数组: `prefill_16385`, `decode_16385`, `prefill_65537` 和 `decode_65537`.

## 身份和发布

工具在计算前验证完整检查点清单.
它从已验证的字节缓冲区读取配置和分词器控制数据.
短提示词必须编码为其固定的 16 个 IDs.
压缩 token 词表必须与检查点配置一致.
模型上下文必须能容纳提示词和一个 decode token.

元数据包含全部提示词 IDs, 端点长度, decode tokens, 完整检查点身份, 源码身份, 依赖版本和 token 映射身份.
它还包含 CPU 设置, 分块大小, 生成时间, 以及完整数组的形状, dtypes 和哈希.
检查点文件状态和参考源码身份必须保持不变, 直到发布.
最后一次检查在序列化之后, 排他链接发布产物之前执行.
已有输出路径, 缺失数组, 错误形状, 非 FP64 数组和非有限值会导致错误.

生成期间保持源码文件和参考环境不变.
将生成日志和未经审查的产物保留在仓库外部.
仅在独立身份和输出审查后冻结端点夹具.
常规验收必须加载固定数组, 而不是生成新的预期输出.

## 命令

使用 `requirements/deepseek-reference.lock` 中的固定环境.
将 `REFERENCE_PYTHON` 设为其 Python 可执行文件.
将 `MODEL_ROOT` 设为检查点目录.
将 `CHECKPOINT_MANIFEST` 设为其完整身份清单.
将 `LONG_OUTPUT` 设为新的外部 NPZ 路径.

```bash
CUDA_VISIBLE_DEVICES= scripts/with-env.sh "$REFERENCE_PYTHON" \
  -m development.model.deepseek_v41.reference_long_case \
  --model "$MODEL_ROOT" --checkpoint-manifest "$CHECKPOINT_MANIFEST" \
  --output "$LONG_OUTPUT"
```

默认设置为八个 CPU 线程, 每次权重读取 2048 个投影行, 每个分块 2048 个提示词行.
分块大小选项接受 1 至 4096 的整数.
不同分块大小可能改变浮点归约顺序.
将记录的分块大小与冻结输出一起保留.

小规模检查将分块数学端点与整个前缀调用比较.
它们还检查发布失败条件.
它们不生成完整检查点输出:

```bash
CUDA_VISIBLE_DEVICES= scripts/with-env.sh "$REFERENCE_PYTHON" \
  -m unittest development.model.deepseek_v41.test_reference_long_case
```
