# Term reference B

| Term | Part of speech | Project meaning |
|---|---|---|
| `projection` | noun | Model linear computation with ordinary or FP8 weights. |
| `normalization` | noun | Scaling by the defined norm and epsilon. |
| `recurrent state` | noun | State used by GDN computation across tokens. |
| `convolution` | noun | Model causal convolution with persistent history. |
| `alias` | noun | Shared storage between tensor views or API parameter names for the same behavior. |
| `runtime` | noun | The active inference implementation and its software environment. |
| `worker` | noun | Python child that controls device computation and request preparation. |
| `process` | noun | Operating-system program instance with its own lifetime. |
| `process` | verb | Transform request data or token rows through a defined software operation. |
| `baseline` | noun | Pinned measurements from original evidence used as the performance denominator. |
| `throughput` | noun | Kept output tokens per second for a measured workload. |
| `TTFT` | noun | Submission-to-first-kept-output-token duration at the caller. |
| `roofline` | noun | Compute and bandwidth limits used to analyze attainable performance. |
| `warmup` | noun | An unmeasured full workload before measured repetitions. |
| `spread` | noun | Maximum minus minimum divided by the median for one measured metric. |
| `NRMSE` | noun | Root mean square error divided by reference root mean square magnitude. |
| `fixture` | noun | Controlled test input or scripted worker. |
| `harness` | noun | Program that constructs cases and collects formal output/timing results. |
| `provenance` | noun | Source, compiler, module, and raw hash identity for a result. |
| `profile` | verb | Collect device execution traces or counters for diagnosis. |
| `redact` | verb | Remove host identities from derived evidence. Keep original data independently. |
| `export` | verb | Write a portable derived evidence record from an unchanged original record. |
| `repository` | noun | Project source and tracked documents in Git. |
| `submodule` | noun | Pinned third-party Git source tree. |
| `lockfile` | noun | Pinned dependency versions for reproducible installation. |
| `RPC` | noun | Correlated request/reply operation over the worker channel. |
| `IPC` | noun | Local interprocess transport endpoint. |
| `SSE` | noun | HTTP server-sent events for incremental output. |
| `backpressure` | noun | Paused generation caused by a full bounded stream queue. |
| `timeout` | noun | Maximum permitted wait for a request or RPC. |
| `TTL` | noun | Time to live of a stored response after completion. |
| `passage` | noun | Twine story unit for a page or reading transition. |
| `companion` | noun | Full same-directory Chinese translation of an English Markdown source. |
| `update` | verb | Change software state, configuration, or its authoritative document. |
| `store` | verb | Keep data in device memory or bounded response storage. |
| `decode` | noun | Incremental inference stage that computes target output tokens. |
| `capture` | noun | CUDA launch recording that supplies a reusable graph. |
| `replay` | noun | Execution of a recorded CUDA Graph with current inputs. |
| `profiling` | noun | Diagnostic measurement of device traces or counters. |
| `lowering` | noun | IR replacement of semantic nodes with selected providers. |
| `store` | noun | Memory operation that writes values to device storage. |
| `update` | noun | Software operation that changes configured or persistent state. |
| `pin` | verb | Fix a software version or device identity for reproducible execution. |
| `run` | verb | Start execution of a software command or program. |
| `pass` | verb | Give satisfactory results for all criteria in a software test or gate. |
| `fail` | verb | Give unsatisfactory results for one or more criteria in a software test or gate. |
| `test` | verb | Use software validation cases to compare behavior with specified criteria. |
| `check` | verb | Use software validation to find violations of specified rules. |
| `independent runtime` | noun | Project inference runtime with no vLLM execution, environment, or build-cache dependency. |
| `independent reference` | noun | Expected operator output computed with an implementation different from the candidate. |
| `original evidence` | noun | Full byte-unchanged measurement artifacts before portable export. |
| `original record` | noun | One full byte-unchanged measurement record before portable export. |
| `original hash` | noun | SHA-256 of a full measurement artifact before portable export. |
| `original data` | noun | Full measurement data before portable redaction or export. |
| `exact division` | noun | FP32 division with correct round-to-nearest output. |
| `exact scale division` | noun | Correct round-to-nearest FP32 division of a maximum by the FP8 range. |
| `FP32 normal value` | noun | Finite FP32 value with an encoded exponent from 1 to 254. Zero and subnormal values are not in this class. |
| `DSpark` | noun | Optional seven-token draft mode with five BF16 GQA layers, target features, and Markov/confidence heads. |
| `target feature` | noun | BF16 target layer output at one target input position, before the next normalization. Temporary verification rows also have these outputs. |
| `Markov head` | noun | Draft projection with checkpoint weights that adds scores from the previous draft token to each row's base logits. |
