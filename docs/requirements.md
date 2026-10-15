# Runtime contracts

Build single-B200 Qwen3.8-27B-FP8 inference.
Rust owns service, scheduling, and logical KV. Python owns GPU computation.
Use ZMQ DEALER and msgpack between processes.

Keep block size 784, 16 FA layers, 48 GDN layers, and `mamba_cache_mode="align"`.
Support input plus output up to 262144 tokens.
Keep offline and HTTP features, ordinary decoding, MTP4, DSpark, and existing numerical tolerances.
Do not alter DSpark checkpoint bytes.

Use Torch operations and repository CUDA. Native CUDA libraries are permitted.
Do not add vLLM dependencies, environments, or caches.
Keep development tools separate from inference dependencies.

Detailed functional and acceptance criteria stay available in the archived
[core](https://github.com/para1lel/oh-my-vllm/blob/461e15d39654784b2351c4e72d49103c79e0ade6/docs/requirements/core.md),
[performance](https://github.com/para1lel/oh-my-vllm/blob/461e15d39654784b2351c4e72d49103c79e0ade6/docs/requirements/performance.md),
[service](https://github.com/para1lel/oh-my-vllm/blob/461e15d39654784b2351c4e72d49103c79e0ade6/docs/requirements/service.md),
[operator](https://github.com/para1lel/oh-my-vllm/blob/461e15d39654784b2351c4e72d49103c79e0ade6/docs/requirements/kernels.md), and
[tutorial](https://github.com/para1lel/oh-my-vllm/blob/461e15d39654784b2351c4e72d49103c79e0ade6/docs/requirements/learning.md) contracts.
The current [contribution procedure](../CONTRIBUTING.md) governs workflow, document scope, and test reporting.
Archived records do not supply acceptance for new code.
