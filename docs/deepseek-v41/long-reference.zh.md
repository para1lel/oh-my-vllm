# DeepSeek 长上下文参考

## 范围

[夹具读取器](../../development/model/deepseek_v41/fixtures.py) 中的 `load_long()` 加载四个冻结的 CPU FP64 端点向量.
[夹具索引](fixtures.zh.md) 给出其产物身份.
数值限值和服务容量是 [阶段计划](plan.zh.md) 中的独立前置条件.

提示词重复 [短参考](reference.zh.md#短文本生成) 中的 16 个 token IDs.
它使用 65537 tokens, 不采用位置偏移, 截断历史或近似窗口重放.
第一个端点位于 16385 个提示词 tokens 之后.
该长度超过第 20 层候选池中每块八个位置的 2048 个块.

第二个端点位于 65537 个提示词 tokens 之后.
该长度超过原始的 65536-token 旋转位置编码上下文.
这些选定边界不能证明 1048576-token 服务容量或 64 个请求准入.

## 计算和状态

[生成器](../../development/model/deepseek_v41/reference_long_case.py) 使用目标模型, 以及解码到 FP64 的原始量化检查点值.
它不使用原生运行时算子或运行时激活量化.
[参考指南](reference.zh.md) 规定算术和逐层流式权重读取.

冻结产物使用最多 4096 tokens 的连续 prefill 分块.
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

元数据绑定完整检查点清单, 15 个源码身份, 全部提示词 IDs, 依赖版本, 执行设置和完整数组哈希.
生成期间, 检查点文件, 源码和参考环境必须保持不变.
工具仅接受完整的有限 FP64 数组和新的输出路径.

将生成日志和未经审查的产物保留在仓库外部.
常规验收必须加载固定数组, 而不是生成新的预期输出.

## 续算状态

每个完成的分块和端点分支都会生成完整提示词续算状态的检查点.
检查点还保留已完成端点和生成时间.
身份变化, 状态形状无效或数据损坏会导致错误.

工具在原子替换前完成文件同步.
`<stem>.previous.npz` 保留上一个有效版本.
工具不会自动回退.
这些检查点文件不是端点 fixture.

## 命令

使用 `requirements/deepseek-reference.lock` 中的固定环境.
将 `REFERENCE_PYTHON` 设为其 Python 可执行文件.
将 `MODEL_ROOT` 设为检查点目录.
将 `CHECKPOINT_MANIFEST` 设为其完整身份清单.
将 `LONG_OUTPUT` 设为新的外部 NPZ 路径.
将 `LONG_STATE` 设为另一个外部 NPZ 路径.

```bash
CUDA_VISIBLE_DEVICES= scripts/with-env.sh "$REFERENCE_PYTHON" \
  -m development.model.deepseek_v41.reference_long_case \
  --model "$MODEL_ROOT" --checkpoint-manifest "$CHECKPOINT_MANIFEST" \
  --output "$LONG_OUTPUT" --checkpoint-state "$LONG_STATE" \
  --threads 8 --projection-row-limit 2048 --chunk-size 4096
```

此命令使用冻结产物的执行设置.
不同分块大小可能改变浮点归约顺序.

中断后重复相同命令.
已有状态文件提供经过验证的续算状态.
如需选择上一版本, 将 `LONG_STATE` 设为该版本的路径.
不使用 `--checkpoint-state` 时, 工具不保存续算状态.

[小规模检查](../../development/model/deepseek_v41/test_reference_long_case.py) 覆盖分块与续算等价性, 身份, 检查点损坏和发布失败.
它们不重新生成完整检查点输出.
