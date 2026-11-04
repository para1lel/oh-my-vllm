# 数值验收

完整模型验收使用独立冻结的 MP4 文本参考, 并记录量化器同步补丁.
Expected array 覆盖 target 词表向量, hidden 端点, DSpark5 proposal, confidence, verification 和已接受状态恢复.
[量化契约](quantization.zh.md)规定模型的数值次序.

原始外部参考树已不存在.
当前 checkout 不提供完整 fixture manifest 和端点字节.
仓库没有受维护的命令来恢复并运行完整的独立模型检查.
现有 artifact 无法重跑全模型数值验收.

## 冻结契约

原始参考 driver 在执行 candidate 之前固定以下限制:

| 输出 | 限制 |
|---|---|
| 完整 target 和 proposal logits | 相对 L2 误差不超过 0.01 |
| Hidden 端点 | 相对 L2 误差不超过 0.01 |
| Confidence | 绝对误差不超过 0.01 |
| Greedy, proposal 和 acceptance ID | 完全相等 |

恢复参考只全新重放已接受 token.
在执行拒绝的计算后截断状态, 不能作为独立恢复参考.
参考包含强制接受零至五个 draft 的情况, correction, 以及后续 target 和 DSpark 端点.
Candidate 验收期间保持 expected array 和限制不变.
算子检查不能替代此契约.

冻结对象具有以下 SHA256 身份:

| 对象 | SHA256 |
|---|---|
| `manifest.json` | `2c40ce592665c14a7334657be7fdcf88fd7ea29c9a276c6fa7b159f5cdd8bbe4` |
| `endpoints.safetensors` | `3089f8e3d0327a07f166c744b25d7751fc65175963fbbd588533d34643568093` |
| `execution-manifest.json` | `82c8f49a4c6341d68716099825e10fdbeddd206295f302f2738db6507fe7579d` |
| 参考 driver | `c4ac8b03345dd465c1a20b34e7cfbfe2a4891b6bb99b8a920e4f05d89900c909` |

恢复的外部记录包含 execution manifest, driver 和部分 fixture-manifest 字段.
这些记录提供了上述身份和限制.
部分字段不能替代原始 manifest 或端点字节.
主机 artifact 位置放在忽略的 `LOCAL.md` 中.

## 恢复独立 fixture

1. 获取原始 manifest, endpoint 文件, execution manifest, source 文件, 同步补丁, driver 和 checkpoint 摘要清单.
2. 将这些 artifact 放在仓库外目录.
3. 按冻结身份检查原始 manifest 和 endpoint 的 SHA256 值.
4. 按完整 manifest 检查 source, patch, driver, tokenizer, checkpoint, compiler 和 runtime 身份.
5. 只有记录的参考 driver 使用转换后 shard 时, 才使用发布的 MP4 checkpoint 转换.
6. 将转换后权重与小型 endpoint fixture 分开存放.
7. 将完整冻结端点向量与独立执行的 candidate 比较.
8. 将比较报告和 candidate 身份保存在仓库外.

如果原始 artifact 不可用, 将全模型检查报告为受阻.
Shape, token ID 或数值容差相同, 不能证明原始字节已恢复.
不要用新生成数组的 hash 替换冻结的 SHA256 值.
新参考是需要单独审查的参考变更.

## 参考生成

使用隔离的开发环境执行发布的 TileLang 参考.
不要将 TileLang 加入推理依赖.
TileLang 0.1.8 可能在量化器全部输入存储完成前复用输入 shared storage.
已记录的补丁在 `act_quant_kernel` 和 `fp4_quant_kernel` 中, 紧接 `T.copy(y_local, y_shared)` 前加入 `T.sync_threads()`.
它不改变 equation, cast 或 checkpoint 值.

补丁 SHA256 为 `a2fcc1970f543f1a360a5f7554e37b21a1a272cfcab463228d6c4377194f4968`.
修改后量化器 SHA256 为 `6206c3bad60de34247481e9c1ab8c487c7930ffca0f773f8c296ba05db879084`.
[量化契约](quantization.zh.md)给出未修改量化器的身份.
两个 barrier 的描述不能替代原始补丁字节.

在参考 manifest 中记录 source, patch, compiler, runtime, tokenizer, checkpoint, input case 和 expected-array 身份.
生成 expected array 前, 使用 Compute Sanitizer Racecheck 检查 packed 和 in-place 格式.
只有重复的 target 和 DSpark 端点逐位相同, 才能冻结数组.
将此 backend 标识为带同步补丁的参考.
不要导入 candidate 模块生成 expected array.

## 辅助检查

[发布操作数检查](../../../development/operators/check_deepseek_released.py)逐位比较捕获的算子输出.
[投影检查](../../../development/model/check_quantized_gemm.py)比较不可变的 native baseline.
[批量检查](../../../development/model/check_deepseek_batch.py)检查 candidate 执行和状态行为.
这些检查与独立全模型参考的范围不同.
它们不证明完整 HTTP, AgentX, 长上下文或 DEP4 验收.
