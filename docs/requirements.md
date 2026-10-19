# Runtime contracts

Build the Qwen3.8-27B-FP8 inference service for one NVIDIA B200.
Use Rust for the service, the scheduler, and the logical KV cache.

Use Python for GPU computation. Use ZMQ DEALER and msgpack between the processes.

Keep the block size at 784, with 16 FA layers, 48 GDN layers, and `mamba_cache_mode="align"`.
Accept an input and output total of at most 262144 tokens.

Keep the offline and HTTP functions and the full-model numerical checks.
Use ordinary decode and MTP4 only.

Accept text, image, and video input.
Use CUDA C++ or CuTe DSL for all model operators.
Native CUDA libraries can supply infrastructure and operator computation.
Do not use other external operator libraries, such as DeepGEMM or FlashInfer.
Do not add vLLM dependencies, environments, or caches.

Use different dependency files for development tools and inference packages.

Use the [model specification](operators/model.md) for mathematical and state behavior.
Use the [GPU execution rules](execution.md) for operator scheduling and CUDA Graph resources.
Use Eager execution for each fragment of an Execute that contains prefill.
Use CUDA Graph recording and replay for decode-only Execute messages.

Intermediate changes can report failed or unmeasured performance configurations.

The final implementation must pass all 24 [HTTP configurations](performance.md#configurations) at the specified nominal-resource threshold.
Operator timing, public-operator comparisons, and measured peak calibration are not acceptance requirements.
Use the short-input, single-iteration logits comparison in the [evaluation procedure](operators/evaluation.md).
Keep applicable service, state-ownership, and graph resource checks.
Update the two README tables before each delivery.

Use these archived contracts for the detailed criteria of the functions and modes in this document:

- [Core](https://github.com/para1lel/oh-my-vllm/blob/461e15d39654784b2351c4e72d49103c79e0ade6/docs/requirements/core.md)
- [Performance](https://github.com/para1lel/oh-my-vllm/blob/461e15d39654784b2351c4e72d49103c79e0ade6/docs/requirements/performance.md)
- [Service](https://github.com/para1lel/oh-my-vllm/blob/461e15d39654784b2351c4e72d49103c79e0ade6/docs/requirements/service.md)
- [Operators](https://github.com/para1lel/oh-my-vllm/blob/461e15d39654784b2351c4e72d49103c79e0ade6/docs/requirements/kernels.md)
- [Tutorial](https://github.com/para1lel/oh-my-vllm/blob/461e15d39654784b2351c4e72d49103c79e0ade6/docs/requirements/learning.md)

The [contribution procedure](../CONTRIBUTING.md) controls the workflow, document scope, and test reporting.
The archived records are not acceptance evidence for new code.
