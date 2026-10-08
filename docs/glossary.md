# Project technical glossary

Use these technical terms only in the defined project meaning.
ASD-STE100 rules still govern grammar and general words.
A technical entry does not authorize the same word in another meaning or part of speech.
The machine source is [terms.json](terms.json).
Proper product names, source identifiers, and protocol text keep their initial forms.

| Term | Part of speech | Project meaning |
|---|---|---|
| `inference` | noun | Model computation that supplies output tokens. |
| `token` | noun | One vocabulary ID in input or output. |
| `checkpoint` | noun | Model weights/configuration, or a saved recurrent state at a token boundary. |
| `prefill` | noun | Model computation over prompt or recompute history. |
| `decode` | verb | Compute the next target output from an accepted prefix. |
| `scheduler` | noun | Rust component that selects token work and logical slots. |
| `scheduling` | noun | Selection of request work with token and sequence budgets. |
| `cache` | noun | Stored KV or state for continued computation or prefix reuse. |
| `cache` | verb | Keep computed data for subsequent reuse. |
| `KV` | noun | Attention keys and values. |
| `FA` | noun | Full attention over the applicable token prefix. |
| `GQA` | noun | Grouped-query attention with fewer KV heads than query heads. |
| `GDN` | noun | GatedDeltaNet recurrent computation and its state. |
| `MTP` | noun | Multi-token prediction with draft proposals and target verification. |
| `MTP4` | noun | MTP mode with four proposed draft tokens. |
| `draft` | noun | An unverified token proposal or the model that supplies it. |
| `bonus` | noun | Target token after the accepted speculative prefix. |
| `slot` | noun | A group-local tensor location for a logical block or state. |
| `page` | noun | Persistent attention storage for 784 target tokens. |
| `subpage` | noun | Native attention view of sixteen tokens in a target page. |
| `batch` | noun | Requests or token rows computed together. |
| `batching` | noun | Combination of active request work in one step. |
| `logits` | noun | Model scores for the output vocabulary before sampling. |
| `sampling` | noun | Selection of a token from model scores and configured constraints. |
| `sample` | verb | Select a token according to the configured distribution. |
| `grammar` | noun | Token-level constraints for JSON answers or function arguments. |
| `mask` | noun | Allowed-token information used before sampling. |
| `tensor` | noun | A typed multidimensional array with device and layout metadata. |
| `dtype` | noun | Tensor element representation, such as BF16 or FP32. |
| `stride` | noun | Storage offset between adjacent entries of a tensor dimension. |
| `metadata` | noun | Configuration and shape/state descriptors different from tensor values. |
| `provider` | noun | Registered implementation of a semantic operation. |
| `IR` | noun | Intermediate representation of semantic GPU operations. |
| `DSL` | noun | Domain-specific language for project semantic operations. |
| `schema` | noun | Explicit contract for data fields or operator mutations. |
| `mutation` | noun | An operation effect that changes persistent tensor storage. |
| `donation` | noun | Reuse of storage from a proven temporary as another output. |
| `graph` | noun | CUDA launch sequence with persistent inputs for replay, or compiler operation representation. |
| `capture` | verb | Record GPU launches in a CUDA Graph. |
| `replay` | verb | Start a previously captured CUDA Graph with current inputs. |
| `lower` | verb | Replace semantic IR nodes with selected provider operations. |
| `compile` | verb | Translate model or kernel code into an executable GPU program. |
| `compilation` | noun | Translation of model or kernel code into executable programs. |
| `allocate` | verb | Supply logical slots or tensor storage for computation. |
| `allocation` | noun | Reservation of logical slots or tensor storage. |
| `preempt` | verb | Release the logical blocks of a request and queue its accepted history for recompute. |
| `preemption` | noun | Request displacement with logical cache pressure. |
| `evict` | verb | Remove a reusable cache or graph entry to release capacity. |
| `rollback` | noun | Restoration of computed progress or speculative state after rejection. |
| `validate` | verb | Compare input with a data, schema, or operation contract and reject invalid input. |
| `validation` | noun | Comparison of input with data, schema, or operation contracts and rejection of invalid input. |
| `verify` | verb | Accept or reject speculative tokens, or compare output with an independent reference. |
| `verification` | noun | Speculative acceptance decisions or comparisons with an independent reference. |
| `configure` | verb | Set software runtime parameters or environment selection. |
| `configuration` | noun | Explicit software parameters and environment selection. |
| `tokenize` | verb | Convert text into vocabulary IDs. |
| `detokenization` | noun | Conversion of vocabulary IDs into output text. |
| `serialize` | verb | Encode data as msgpack or JSON bytes. |
| `dispatch` | noun | Host selection and launch of a kernel implementation. |
| `quantization` | noun | Conversion to FP8 values with explicit scales and rounding. |
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

## Maintenance

Add a technical noun or verb only when the approved dictionary cannot express the same technical meaning accurately.
Supply one definition, part of speech, Chinese translation, and explicit forms.
Review the term against project source and Rule 1.5 or Rule 1.12.
Use general approved wording, such as keep, examine, or make sure, when it expresses the same action.
Code identifiers can use library-defined names independently of prose terminology.
