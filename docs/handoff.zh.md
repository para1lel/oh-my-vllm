# 工作状态

继续在 `nightly` 上工作.

## 当前服务路径

Qwen 保留现有的单 worker 服务路径.
共享 Rust service, scheduler, logical cache, DEALER transport 和 msgpack protocol 保持启用.

DeepSeek-V4.1-Flash 已有 model adapter 和四 rank Python bridge.
adapter 使用原始 48 shard checkpoint 和 rank-local placement.
它支持 text target execution 和 DSpark5 greedy proposal verification.
bridge 支持 `tp4`, `tep4` 以及已明确记录的 DEP4 executor fallback.

adapter 使用 DeepSeek descriptor, cache groups, tokenizer identity, inference-config identity, native projection, routing, activation, mHC, RMS, sparse attention 和 transformer support.
prefill 使用最多 8192 个 total row 的 working chunk.
设置 `OH_MY_VLLM_DEEPSEEK_PREFILL_ROWS` 可以使用更小的 model-local chunk.
完成的 chunk 会在下一次 native call 之前释放 hidden tail.

DSpark verification 保持 rejected draft 为 private state, 并输出 target correction.
Engram table 使用 file-backed host mapping.
Lookup 只将选中的 row 分阶段传到 GPU.

released DeepSeek shape 支持 native projection, routing, activation, mHC, RMS 和 sparse attention path.
native contract 支持长 row range.
大型 prefill 使用有界的 8192-row launch, 以限制 temporary storage.

sparse attention status failure 会禁用 native path, 并使用 reference path.
short query 的 native sparse packing 通过 `OH_MY_VLLM_DEEPSEEK_NATIVE_SPARSE=1` opt-in.
long query 使用 native sparse packing, 因为 reference path 的扩展性较差.

DeepSeek physical page 现在会在 request slot 之间保存 window KV, compressed KV,
index key, Engram hash state 和 compressor tail.
cache operation 支持 logical block store, load, drop, partial prefix copy 和
page-backed host snapshot.

accepted DSpark target state 会 seed MTP window, speculative rollback 会保持
rejected draft state 为 private state.
nonzero multi-token prefill 使用 fused chunk path.
target 和 MTP seed chunk 会将 total native row 保持在 operator limit 以下.
aligned prefill batch 最多使用四个 request 组成一个 subbatch.

非 final prefill 会跳过 vocabulary projection.
DSpark 只保留 MTP window 所需的 target hidden tail.
ordinary execution 不会收集 target hidden state.

完整的 initial prompt 可以作为一个 aligned request batch 执行.
每次 target call 都会 reset shared query attention state.
index source 总是为当前 request slot 发布 cache view.

serving engine 会执行配置的 `max_num_seqs` resident limit.
它会保留一个额外的 request 做 preparation, 以便让 CPU admission 与 GPU work overlap.

DeepSeek execution 当前使用 Eager.
共享 Qwen CUDA Graph path 已有 stream handoff, bounded graph workspace, graph upload 和 cleanup checks.
DeepSeek multi-rank Graph capture 尚未接入.
对照当前 vLLM 和 SGLang graph pattern 后, 在定义 static multi-rank
workspace 和 capture bucket 之前, DeepSeek 保持 Eager.
共享 Qwen graph path 已经管理 stream handoff, graph resource 和 cleanup.

DeepSeek startup 只读取 `config.json` 和 `model.safetensors.index.json` 生成 descriptor identity.
Ready 之前不会 hash weight shard.
rank-local checkpoint tile 使用 65536 行, 以减少小块 host-to-device copy.
设置 `OH_MY_VLLM_DEEPSEEK_LOAD_TILE_ROWS` 可以覆盖此 startup setting.
外部 startup trace 使用 `OH_MY_VLLM_STARTUP_TRACE`, 并使用与 repository 分开的 location.

## 当前证据

已在 GPU `4,5,6,7` 上使用 TEP4 完成短真实 checkpoint execution.
它覆盖 readiness, ordinary decode, DSpark proposal output 和 draft rejection.
CPU 和 Rust checks 覆盖 descriptor, model, operators, launcher cleanup, matrix scope, speculation 和 shared runtime.

2026 年 10 月 28 日, 完整 Rust HTTP service 使用 TEP4 和短 context 成功 ready.
Smoke run 检查 model listing, diagnostics, ordinary text, DSpark5, SSE usage, validation error 和 shutdown.
DSpark5 HTTP log 记录了 2 个 proposed draft token 和 2 个 verified draft token.
这次 smoke run 是功能证据, 不替代 approved matrix.

physical cache change 之后, HTTP service 使用 GPU `1,2,3,4`, TEP4, 8 个 block 和 1024-token physical context 成功 ready.
两个 ordinary request 都成功返回.
第二个 request 从 physical prefix cache 复用了 13 个 prompt token.
一个 DSpark5 target request 也成功返回, 且没有 worker restart.
这个 2-token DSpark5 request 没有 proposal budget, 所以 proposal counter 为零.
cache behavior test 覆盖 host snapshot store, load, drop, prefix copy 和 restoration.

最新的 native-operator smoke 使用 GPU `1,2,3,4` 成功 ready.
它覆盖 ordinary decode, DSpark5 verification, prefix reuse 和 graceful shutdown.
ordinary 和 DSpark5 request 都完成, 且没有 worker replacement.

2026 年 10 月 28 日, GPU `1,2,3,4` 上的 TEP4 smoke 在 31.012 秒内达到 Ready.
它使用 8 个 physical block 和 1024-token context, 并返回 4 个 ordinary output token.
四个 startup trace 的 metadata identity 都低于 0.1 秒, 且没有 shard hash phase.
rank-local weight load 约用时 19-20 秒, service 没有 worker replacement.

[DeepSeek matrix](../development/serving/deepseek_matrix.py) 中有批准的 workload definition.
HTTP runner 定义 36 个 DSpark5 case 和 2 个 ordinary TP4 case.
AgentX runner 定义 TP4/4, TEP4/16 和 DEP4/64 case.
当前还没有记录完整的 approved HTTP 或 AgentX result set.
CodeGraph 1.6.2 已安装到 Codex.
repository 已有 local CodeGraph index, 并忽略 `.codegraph/`.

## 下一步

- 为并发 16 和 100K 至 200K token trace 调整 AgentX physical block 和 temporary workspace capacity.
- 用稳定的 request ownership 和 expert dispatch 替换 DEP4 fallback.
- 将 native DeepSeek path 与 reference path 和公开 baseline 做性能比较.
- 将 large-row native temporary storage 移入 adapter-owned graph workspace.
- 用 fused grouped expert GEMM 替换 TEP4 和 DEP4 的 Python expert loop.
- 在添加 DeepSeek CUDA Graph capture 和 replay 之前定义 static multi-rank workspace.
- 在 GPU 0 以外的显卡上运行 38 个 HTTP case 和 3 个 AgentX case.
- 交付前取得 independent code review.

原始报告, host settings, model paths 和临时 process records 保留在 repository 外.
host settings 放在 `LOCAL.md` 中.
