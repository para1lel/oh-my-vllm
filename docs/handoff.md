# Work state

Use `nightly` for development.

## Current paths

Use the [architecture](architecture.md), [runtime contracts](requirements.md), and [execution rules](execution.md) for shared behavior.
Model guides own their implementation details and acceptance scope.

- [Qwen](models/qwen38/README.md) has the latest completed [HTTP and AgentX results](models/qwen38/results.md).
- [DeepSeek](models/deepseek_v41/README.md) has four-rank text target and DSpark5 paths.
  Its [numerical acceptance](models/deepseek_v41/evaluation.md), full capacity, HTTP matrix, and AgentX acceptance are incomplete.
- [Accounting](models/deepseek_v41/accounting.md) defines the current trace contract.
  The [performance guide](models/deepseek_v41/performance.md) defines the reference calculations and approved workloads.

## Next work

- Restore the independent DeepSeek endpoint data before numerical acceptance.
- Complete DeepSeek numerical and state acceptance for the selected parallel strategy.
- Measure cache pages, state endpoints, projection storage, and temporary storage before long-context capacity changes.
- Complete the approved DeepSeek HTTP and AgentX workloads after numerical and capacity acceptance.
- Use the fixed [Qwen operator cases](models/qwen38/operators.md) for applicable operator changes.

Keep model paths and host settings in `LOCAL.md`.
Keep raw records external to the repository.
