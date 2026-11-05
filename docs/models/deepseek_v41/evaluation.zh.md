# 数值验收

完整模型验收使用独立冻结的 MP4 文本参考, 并记录量化器同步补丁.
Expected array 覆盖 target 词表向量, hidden 端点, DSpark5 proposal, confidence, verification 和已接受状态恢复.
[量化契约](quantization.zh.md)规定模型的数值次序.

[仓库检查入口](../../../development/model/check_deepseek_reference.py)读取冻结 fixture, 并通过 native 算子运行 candidate Worker.
恢复的 fixture manifest 和端点字节的 SHA256 值均与冻结值相等.
Fixture 包保存在仓库外.
初始参考的执行包仍不完整.

## 冻结契约

冻结的参考 driver 在执行 candidate 之前固定以下限制:

| 输出 | 限制 |
|---|---|
| 完整 target 和 proposal logits | 相对 L2 误差不超过 0.01 |
| Hidden 端点 | 相对 L2 误差不超过 0.01 |
| Confidence | 绝对误差不超过 0.01 |
| Greedy, proposal 和 acceptance ID | ID 必须相等 |

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

冻结的 driver 和同步修复后的源码在两次执行中复现了端点文件.
这两次执行的全部 143 个张量和完整端点文件均逐字节相等.
Fixture manifest 根据初始字段次序和已记录的身份重新构造.
完整 manifest 的 SHA256 与其冻结值相等.
这不表示找到了初始 manifest 文件.

本次恢复的执行记录, checkpoint 摘要清单和重建补丁在已审查的恢复包中具有不同的身份.
初始补丁字节, checkpoint 摘要清单文件, preparation 附件和 compiled-cache 附件仍缺失.
安装库的 hash 不能说明执行时实际映射了哪些 GPU 库.
恢复的初始参考 execution manifest 不提供这些缺失附件.
主机 artifact 位置放在忽略的 `LOCAL.md` 中.

## 运行独立检查

从本地配置设置 `DEEPSEEK_REFERENCE_ROOT`, `MODEL_ROOT` 和 `ARTIFACT_ROOT`.
将 `CHECKPOINT_DIGESTS` 设为本次参考转换所用的 48 个输入 shard 的摘要记录.
将转换后的 MP4 权重保存在与端点 fixture 不同的目录中.

首先在 CPU 上检查冻结身份, 完整张量 schema 和 acceptance 字段:

```bash
scripts/with-env.sh python -m development.model.check_deepseek_reference \
  --reference "$DEEPSEEK_REFERENCE_ROOT" \
  --output "$ARTIFACT_ROOT/deepseek-reference-validation" \
  --validate-only
```

将 `NUMERICAL_GPU_UUIDS` 设为按顺序排列的四个 GPU UUID.
每次执行使用新的输出目录:

```bash
OH_MY_VLLM_GPU_COUNT=4 OH_MY_VLLM_GPU_UUIDS="$NUMERICAL_GPU_UUIDS" \
  scripts/with-env.sh scripts/with-gpu.sh \
  python -m development.model.check_deepseek_reference \
  --reference "$DEEPSEEK_REFERENCE_ROOT" \
  --model "$MODEL_ROOT" --strategy tp4 \
  --checkpoint-digests "$CHECKPOINT_DIGESTS" \
  --evidence "$DEEPSEEK_REFERENCE_ROOT/bundle-manifest.json" \
  --output "$ARTIFACT_ROOT/deepseek-reference-tp4"
```

分别以 `--strategy tep4` 和 `--strategy dep4` 重跑, 并使用不同输出目录.
每次执行都将 candidate checkpoint 字节与本次参考转换的输入进行比较.
它记录 source, native binary, configuration, tokenizer, runtime, strategy 和有序 GPU 身份.
本次恢复的摘要记录不能替代初始参考中缺失的摘要清单文件.

默认选择包含全部九个 case 和 143 个冻结张量.
Case 包含短 target 调用, window 边界, 真实 proposal, 以及强制接受零至五个 draft.
检查器将每个完整词表向量和 hidden 端点与冻结 expected array 比较.
它使用冻结契约中的相对 L2 限制.
Greedy, proposal, verification, acceptance 和 correction ID 必须与冻结的 expected ID 相等.
Confidence 使用冻结契约中的最大绝对误差限制.

检查器还执行六 token native verification.
随后对已接受端点执行 state commit.
它通过 candidate cache 恢复该端点.
它将恢复的状态与仅包含已接受 token 的全新重放进行比较.
它将后续 target 和 DSpark 端点与冻结 expected array 比较.

全新重放不生成或修改 expected array.
原始数组, 状态记录, 身份和比较报告保存在仓库外的输出目录.

检查器使用单请求 (`max_num_seqs=1`) Eager 执行, 并关闭 batch decode.
DEP4 使用 rank zero 作为 dense request owner, 另三个 rank 只参与 expert 计算.
Expert exchange 使用生产 Worker 的配置.
此范围不包含多个 request owner 或包含多个请求的 batch.

执行失败后不会选择不同策略.
使用 `--cases` 子集的检查不包含全部 case.
此入口不表示 CUDA Graph, Rust service, HTTP, AgentX 或长上下文验收.

Fixture 缺失或冻结 SHA256 不同时, 检查立即停止.
Candidate 验收期间不要替换冻结身份或限制.
新参考必须经过新的审查.

## 参考生成

使用隔离的开发环境执行发布的 TileLang 参考.
不要将 TileLang 加入推理依赖.
TileLang 0.1.8 可能在量化器全部输入存储完成前复用输入 shared storage.
已记录的补丁在 `act_quant_kernel` 和 `fp4_quant_kernel` 中, 紧接 `T.copy(y_local, y_shared)` 前加入 `T.sync_threads()`.
它不改变 equation, cast 或 checkpoint 值.

补丁 SHA256 为 `a2fcc1970f543f1a360a5f7554e37b21a1a272cfcab463228d6c4377194f4968`.
修改后量化器 SHA256 为 `6206c3bad60de34247481e9c1ab8c487c7930ffca0f773f8c296ba05db879084`.
[量化契约](quantization.zh.md)给出未修改量化器的身份.
两个 barrier 的描述不能替代初始补丁字节.

在参考 manifest 中记录 source, patch, compiler, runtime, tokenizer, checkpoint, input case 和 expected-array 身份.
生成 expected array 前, 使用 Compute Sanitizer Racecheck 检查 packed 和 in-place 格式.
只有两次执行的 target 和 DSpark 端点逐位相等, 才能冻结数组.
将此 backend 标识为带同步补丁的参考.
不要导入 candidate 模块生成 expected array.

## 辅助检查

[发布操作数检查](../../../development/operators/check_deepseek_released.py)逐位比较捕获的算子输出.
[投影检查](../../../development/model/check_quantized_gemm.py)比较不可变的 native baseline.
[批量检查](../../../development/model/check_deepseek_batch.py)将 candidate batch 执行和状态行为与 candidate 单请求执行比较.
这些检查与独立全模型参考的范围不同.
它们不表示完整 HTTP, AgentX, 长上下文或 DEP4 验收.
