# HTTP 性能测量

使用四张 B200 显卡和完整 Rust service.
选择 GPU 0 以外的显卡.
原始 artifact 保留在 repository 外.

approved matrix 有 38 个 case.

| 模式 | Strategy | Batch | Input tokens | Output tokens | Case 数 |
|---|---|---:|---:|---:|---:|
| DSpark5 | TP4, TEP4, DEP4 | 4, 16, 64 | 400000, 800000 | 64, 8192 | 36 |
| Ordinary | TP4 | 4 | 400000 | 64, 8192 | 2 |

Input 和 output length 适用于每个 request.
使用 uncached prefix 和 exact tokenizer length.
在 model sampling contract 扩展前使用 greedy text execution.
runner 设置 `ignore_eos` 并检查完整 usage record.

使用以下命令运行 matrix:

```bash
OH_MY_VLLM_GPU_COUNT=4 scripts/with-env.sh scripts/with-gpu.sh python -m development.serving.deepseek_benchmark \
  --model "$MODEL_ROOT" \
  --artifacts "$ARTIFACT_ROOT" \
  --csv docs/models/deepseek_v41/assets/http-latest.csv
```

使用 `development/serving/deepseek_matrix.py` 中的 identifier 传给 `--case` 选择子集.
使用 runner 的 `--cache-context` option 设置 `OH_MY_VLLM_DEEPSEEK_CACHE_CONTEXT`.
runner 记录 source, executable, native module, checkpoint, tokenizer, topology 和 configuration identity.
它还记录测试程序的 Python, PyTorch, CUDA, NCCL 和数值设置.
这些设置不独立标识每个工作进程的运行时.

完整 timer 包含 request submission, prompt preparation, queueing, model execution 和 response parsing.
它不包含 model loading 和 initialization.
CSV 记录 mode, strategy, batch, exact length, request count, full seconds 和 output rate.

当前还没有记录完整的 approved matrix result.
HTTP result 需要 DeepSeek-specific reference calculation 后才能报告 efficiency.
当前会话的 SOTA 目标不加入 repository performance gate.
