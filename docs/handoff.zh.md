# 交接记录 — 2026-09-24

[English](handoff.md)。agent 以英文原文为准。

## 当前状态

CUDA 迁移已完成并通过验收。下一项任务是修复整库代码审计发现的问题。

- **Kernel 后端：** B200 上默认使用 CUDA。在 Python 启动前设置
  `OH_MY_VLLM_KERNEL_BACKEND=tilelang` 则改用冻结的 TileLang 对照。运行时身份报告实际使用的
  后端，不会静默回退到 TileLang。
- **职责划分：** Rust 负责服务、调度和逻辑 KV 缓存；Python 负责 GPU 计算。不使用任何 vLLM
  运行时、源码或环境依赖。
- **算子（干净提交 `c36d1c9`，显式选择 CUDA）：** 全部 147 项正式用例通过。每项都通过三轮、
  每轮 20 对交替测量，且单侧 95% bootstrap 下界为正。
- **框架（同一提交）：** 全部 12 组通过。
  - 吞吐为冻结 vLLM 基线的 97.52–125.09%。
  - TTFT 为基线的 76.01–97.50%。
  - 每组都满足 10% 极差规则。
  - 余量最小的是 MTP batch 4，吞吐 97.52%。
- **默认选择变更（`030f60f`）：** 只修改后端选择和身份报告。174 项测试加 31 个子测试的
  套件以及 MTP4 服务约束和生命周期检查，是在 `ef07b2b` 之后、后来成为该提交的工作树上
  通过的（CUDA 源码未变），而不是在 `030f60f` 上单独运行的。
- **其他已通过的检查：**
  - 六组 258048+4096 边界运行完成，无 OOM、无抢占。
  - 真实文本的抢占/前缀检查。
  - 两种 oh-my-pi API。

证据见 [acceptance.zh.md](acceptance.zh.md) 以及
`bench/baseline/2026-09-22-cuda-{operators,framework,features}.json`。

## 待办：修复代码审计问题

对 `030f60f` 的审计记录在 [audit-2026-09-23.zh.md](audit-2026-09-23.zh.md)，共 51 项。
修复工作于 2026-09-24 开始，进行中。

### 已完成批次

**批次 1 — 服务可用性**（`d0d286a`，2026-09-24）：SRV-01/02/03/04/06 已修复。
- SRV-01：`request.rs` 和 `sampling.py` 都将 temperature < 1e-5 夹到 0。
- SRV-02：子进程 pre-exec 调用 `prctl(PR_SET_PDEATHSIG, SIGKILL)`；Python 监控父进程 PID，
  父进程消失时退出；`sock.recv` 设置 5 秒超时。
- SRV-03：HTTP ready channel 携带 `(StatusCode, String)`，非 200 错误正常传播。
- SRV-04：`Parser::feed` 在 `LengthFinish` 时冲刷暂存字节；已新增测试。
- SRV-06：`register_request`/`abort_request` 降级为 fire-and-forget。

**批次 2 — 证据完整性**（`895d57b`，2026-09-24）：EVD-01/02/06/07/11/12/13 已修复。
- EVD-01：`compare_vllm.py` 默认 reps=5、warmup=2；输出离散度警告。
- EVD-02：`ttft.py` 通过 `_parse_bench_config()` 读取 worker 输出的 BENCH_CONFIG 日志行；
  `zmq-worker/src/main.rs` 在启动时输出 `info!("BENCH_CONFIG", …)`。
- EVD-06：`test_kernel_reference.py` 断言冻结参考 JSON 中的精确文件集合。
- EVD-07：`tests/test_context_boundary.py` 新增六个 REQ-CONTEXT-001 边界运行的占位测试。
- EVD-11：`measurement.py` 在缓存根目录缺失时追加到 `problems`。
- EVD-12：`cuda_backend/__init__.py` 新增 `provenance()`，返回 nvcc 版本和编译后
  `.so` 的 SHA-256。
- EVD-13：稳态审计将 `TVM_FFI_CACHE_DIR` 作为第四个缓存根目录。

**批次 3 — Kernel/worker 契约（宿主端）**（`eb5ff62`，2026-09-24）：
  KRN-01/04/05、PY-01/02、MNT-02 已修复；KRN-02/03 涉及设备代码，需 GPU 重跑，延后。
- KRN-01：`decode_attention.py` 在提供 `starts` 时断言 `group_size ≤ 5`。
- KRN-04：`elementwise.py` 和 `fp8.py` 防止 int32 溢出。
- KRN-05：`gdn.py` 将 dtype/shape/stride 校验移入 `normalize_qk`。
- PY-01：`batch_plan.py` 在 debug 模式断言可写 FA 尾页跨请求不相交。
- PY-02：`qwen.py` 对每个 FP8 缩放投影断言 `w.shape[0] % 128 == 0`。
- MNT-02：`elementwise.py` 在 `silu_mul`/`delta_gates` 对空张量提前返回。
- 新增测试：`tests/test_validation_guards.py`、`tests/test_batch_plan.py`、
  `tests/test_independent_decode_attention.py`。

**批次 4 — 调度器加固**（`42197f7`，2026-09-24）：SCH-01/02/03/04/05 已修复。
- SCH-01：`update()` 在修改状态前校验 token 数量和 `is_final_chunk` 标志。
- SCH-02：`pool.rs` 中引用计数/空闲列表不变量从 `debug_assert!` 改为 `assert!`。
- SCH-03：`aligned_prefill` 在 `count > 0` 时将结果夹到至少 1。
- SCH-04：`add_request()` 返回 `bool`，在入队时拒绝超容量请求（服务层返回 HTTP 413）。
- SCH-05：`remove_blocks_in_range` 遇到 NULL_BLOCK_ID 时改用 `continue` 而非 `break`。
- SRV-05/07 仍为 open（取消安全性和侧信道覆盖需要较大改动，见审计文档）。

**批次 6 部分 — 清理**（`62101a3`，2026-09-24）：MNT-04 已修复。
- 从 `client.rs` 删除死代码 `abort_request()` 方法及 `AbortMsg` import。
- 更新 `lib.rs` 中的协议文档注释，与实际报文格式保持一致。

### 剩余未修复问题

以下问题已延后（设备代码需 GPU 重跑）或归入 P2/P3 待后续处理：

- **KRN-02/03**（P1）：设备代码修复需在 B200 上重跑正式算子用例。
- **SRV-05/07**（P1）：RPC 取消安全性与 `serving_outputs` 侧信道。
- **P2 性能问题：** PY-03/04/05/06/07、SRV-08/09/10、SCH-06/07、KRN-06/07/08/09/10。
  每项需先测量再决定是否保留。
- **P3 清理/测试问题：** EVD-03/04/05/08/09/10、MNT-01/03。

## 文档清理（本次变更）

- 新增审计文档。
- 从 `acceptance.md`、`cuda-development.md`、`plan.md` 和 `bench/baseline/README.md` 删除了
  已被取代的里程碑叙述。原始产物仍然保留，并由 `acceptance.md` 中的历史表索引。
- 删除了 `performance-gap-2026-09-22.md`。它描述的 MTP 差距已被后续工作消除，汇总数据仍保留在
  `bench/baseline/2026-09-22-performance-gap.json`。
- 修正了过时内容：
  - 命令改为 `ttft.py` 和完整的 pytest 套件。
  - 协议和抢占的描述与当前代码一致。
  - ADR 状态已更新。
- 同步更新了中文译本。

## 操作规则

- **环境：** 使用 `scripts/with-env.sh`。
- **GPU 选择：** 用 `scripts/with-gpu.sh` 按 UUID 选择空闲 GPU。
- **Git：**
  - 只在 `main` 上开发和提交。
  - 只暂存有意修改的文件。
  - 保留 hooks。
  - 维护中文 Markdown 译本。
  - 带上要求的提交署名。
- **GPU 清理：** 及时停止自己启动的所有 GPU 程序，并确认端口和显存已释放。
- **不进仓库的内容：** 临时调优和原始 GPU trace。正式的汇总证据放在 `bench/baseline`。
