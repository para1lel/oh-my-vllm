# Term reference A

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
| `grammar` | noun | Token-level constraints for reasoning control, JSON answers, or function arguments. |
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
