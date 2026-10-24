# Qwen 适配器

Qwen3.8-27B-FP8 在单张 NVIDIA B200 上运行.
适配器为文本, 图片和视频实现[共享服务功能](../../requirements.zh.md#服务功能).
使用普通解码或 MTP4, 输入与输出合计不超过 262144 tokens.

[Python 适配器](../../../python/oh_my_vllm/models/qwen38)负责计算, 权重, 缓存, 编解码器和 MTP4.
[Rust 策略](../../../crates/zmq-worker/src/models/qwen38/policy.rs)负责默认值和 readiness 验证.
Checkpoint 声明 `Qwen3_5ForConditionalGeneration` 和 `qwen3_5`.
保持这些原始标识符.

使用块大小 784, 16 层 FA, 48 层 GDN 和 `mamba_cache_mode="align"`.
默认调度器提供四个请求槽位和 32768-token 步骤预算.
默认公开模型名称为 `qwen3.8-27b-fp8`.

## 指南

- [模型数学和精度](model.zh.md)
- [缓存容量, 前缀和主机换出](cache.zh.md)
- [全模型数值验收](evaluation.zh.md)
- [算子用例](operators.zh.md), [数值限值](precision.zh.md)和[性能计算](operator-performance.zh.md)
- [HTTP 流程](performance.zh.md), [AgentX 流程](agentx.zh.md)和[最新 HTTP 结果](results.zh.md)

命令和 API 字段见[共享服务指南](../../serving.zh.md).
MTP4 在子命令前使用 `--speculative-mode mtp --num-speculative-tokens 4`.
普通解码使用 `--speculative-mode none`.

## 媒体

Worker 下载并解码媒体.
每个编码媒体项的上限是 64 MiB.
图片解码后的上限是 67108864 pixels.
采样视频帧的上限是 512 MiB.
源高度和宽度不得超过 65536 pixels.

视频每秒采样两帧, 最多 768 帧.
每个时间帧对在 prompt 中有一个时间戳.
媒体 token 使用三轴 RoPE 位置.
媒体请求具有独立的前缀缓存身份.

## 数值命令

运行 CPU 证据和计费检查:

```bash
scripts/with-env.sh python -m unittest development.model.test_identity development.model.test_logits development.serving.test_roofline
```

对照使用仓库内冻结的 [FP64 reference](../../../development/model/reference.npz).
将 `MODEL_ROOT` 设为 checkpoint 目录.
将 `LOGITS_REPORT` 设为仓库外的新 JSON 路径.
运行全模型对照:

```bash
scripts/with-env.sh scripts/with-gpu.sh python -m development.model --model "$MODEL_ROOT" --report "$LOGITS_REPORT"
```

命令还将推理 logits 保存到同名的 `.npz` 文件.
使用 `--mode prefill` 对照短提示的末行.
[评估流程](evaluation.zh.md)规定输入, 精度, 指标及范围.

只有明确变更参考时, 才需要生成新文件.
在独立环境中安装 `requirements/model-reference.lock`.
使用[官方 PyTorch 包](https://pytorch.org/get-started/previous-versions/).
推理 worker 不使用 PyTorch.
将 `LOGITS_REFERENCE` 设为仓库外的新 `.npz` 路径.
用参考环境生成文件:

```bash
scripts/with-env.sh scripts/with-gpu.sh python -m development.model.reference --model "$MODEL_ROOT" --output "$LOGITS_REFERENCE"
```

生成器不覆盖文件.
只有 checkpoint, 输入, 生成器和数值策略的审查全部完成后, 才更新冻结文件及其 SHA256.

## 算子命令

将 `OPERATOR_REPORT` 设为仓库外的新目录.
在推理环境中运行全部固定数值用例:

```bash
scripts/with-env.sh scripts/with-gpu.sh python -m development.operators numerics --model "$MODEL_ROOT" --report "$OPERATOR_REPORT"
```

该命令加载仓库内 fixture 和批准的门槛, 不使用 PyTorch.
fixture 缺失, identity 改变或未通过门槛都会导致错误退出.

明确更新 fixture 时, 将 `OPERATOR_FIXTURES` 设为仓库外的新暂存目录.
在参考环境中生成固定数组:

```bash
scripts/with-env.sh scripts/with-gpu.sh python -m development.operators.reference --model "$MODEL_ROOT" --output "$OPERATOR_FIXTURES"
```

将 `OPERATOR_LIMITS` 设为仓库外的新 JSON 路径.
按已批准的[精度策略](precision.zh.md)提议门槛:

```bash
scripts/with-env.sh python -m development.operators.budgets --fixtures "$OPERATOR_FIXTURES" --output "$OPERATOR_LIMITS"
```

更新仓库内 fixture, 用例哈希和[门槛](../../../development/operators/limits.json)前, 检查数组, 清单, 生成器 identity 和提议门槛.

将 `OPERATOR_PERFORMANCE` 设为仓库外的新目录.
按已批准的倍率门槛运行全部固定性能用例:

```bash
scripts/with-env.sh scripts/with-gpu.sh python -m development.operators performance --model "$MODEL_ROOT" --report "$OPERATOR_PERFORMANCE"
```

[资源与计时策略](operator-performance.zh.md)定义测量的操作和理论下界.
失败或无法判定的用例会导致错误退出.

初始校准时, 将 `OPERATOR_CALIBRATION` 设为仓库外的新目录.
采集五个完整 session:

```bash
scripts/with-env.sh scripts/with-gpu.sh python -m development.operators calibrate --model "$MODEL_ROOT" --report "$OPERATOR_CALIBRATION"
```

此命令在外部报告中提议门槛.
它不修改仓库门槛.
将批准的性能条目复制到门槛文件前, 审查完整报告及其来源记录.

完整推理计时使用 [HTTP benchmark](performance.zh.md#测量命令).
比较变更前后的测量时, 使用相同的 `--prompt-seed` 值.
HTTP 产物记录此 seed 和执行策略.
