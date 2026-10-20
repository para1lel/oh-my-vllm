# Operator validation implementation plan

> Agent instructions: Use `superpowers:executing-plans` or `superpowers:subagent-driven-development` after selection of the execution method. Complete the checkbox steps in sequence.

**Goal:** Give kernel changes fixed PyTorch reference fixtures and theoretical-latency acceptance limits.

**Architecture:** Development adapters call complete production operations. Numerical tests load repository fixtures. Performance tests compare latency ratios with frozen per-case limits.

**Tech Stack:** Python, NumPy, PyTorch FP64 for explicit fixture generation, TVM FFI, CUDA events, and one NVIDIA B200.

**Spec:** [Microbench design](microbench.md) and [operator structure design](design.md).

## Global constraints

- Inference must not import development dependencies.
- Use the [model contracts](model.md), [GPU ownership rules](../execution.md), and [contribution procedure](../../CONTRIBUTING.md).
- Keep the frozen short full-model reference and its SHA256 unchanged.
- Store numerical fixtures in the repository. Keep timing samples, test reports, and host settings external.
- Give the reason for each new behavior test before implementation. Use a small number of parameterized tests.
- Use at most five numerical cases and five performance cases per boundary.
- Missing fixtures, approved limits, or comparable primary measurements prevent acceptance.

## Review focus

| Condition | Necessary behavior | Task |
|---|---|---|
| Missing, changed, or regenerated expected arrays | Fail without running PyTorch in a usual test | V1 |
| Wrong state writes with plausible outputs | Check assigned writes and untouched guards | V2 |
| Sampling and recurrence state | Match the fixed reference and exact counter/slot contracts | V2 |
| Hot weights or inflated work accounting | Keep the theoretical denominator conservative and fixed | V3 |
| Correct local kernels with incorrect composition | Keep full-model and HTTP acceptance | V4 |

## Fixed shape selection

The following tuples are complete case choices, not values for a Cartesian product.
`B` is request count, `T` is total query rows, `Q` is queries per request, and `L` is context length per request.
Use checkpoint widths from the model contract.

| Text case | Numerical `(B,T,Q,L)` | Performance `(B,T,Q,L)` |
|---|---|---|
| Single decode | `(1,1,1,784)` | `(1,1,1,32768)` |
| Batched decode | `(4,4,1,784)` | `(4,4,1,131072)` |
| MTP verification | `(4,20,5,784)` | `(4,20,5,131072)` |
| Checkpoint prefill | `(1,784,784,784)` | `(1,784,784,784)` |
| Large prefill | `(1,1568,1568,1568)` | `(4,32768,8192,32768)` |

FA and GDN use these five cases.
FFN and `output.hidden` use their `T` values without unused attention metadata.
Use consistent starting positions and page/state slots in each fixture.
GDN cases check all assigned checkpoints, with a restored-state decode case.

| Other boundary | Fixed numerical cases | Fixed performance cases |
|---|---|---|
| `output.distribution`, `output.sample` | Rows 1 greedy, 1 stochastic, 4 stochastic, 5 stochastic, 20 stochastic | Same five cases |
| `mtp.context` | `(T,B)=(1,1),(4,1),(8,2),(16,4),(784,1)` | First four cases plus `(32768,4)` |
| One `mtp.draft` | Batch 1 query-only, 1 full, 2 full, 4 query-only, 4 full | Same five cases |
| Four-draft chain | Batch 1, 2, 4 | Same three cases |
| `mtp.accept` | Batch 1 all accepted, 1 first rejection, 2 partial acceptance, 4 mixed rejection, 4 inactive suffixes | Same five cases |
| Each vision boundary | Grid/frame pairs `(16x16,1),(28x28,1),(14x14,3)` | Same three cases |

MTP contexts use length 784 for numerical cases and length 32768 for performance cases when prior context exists.
The 784-row MTP prefill starts at position 1 and ends with context length 785.
Draft cases use those same respective context lengths.
Layer indices start at zero.
Numerical cases use checkpoint layer 3 for FA, layer 0 for GDN/FFN, and vision layers 0/1/26 for input/internal/merge.
Record full weight tensor names in manifests.

Use one fixed data/mask/penalty configuration per case.
Sampling fixtures select these settings to exercise valid masks, penalties, truncation, and bonus behavior.
Do not add seed or layer sweeps to the numerical set.

Each FA/GDN/FFN performance case executes one fixed pass over its applicable target layers, in model order.
An internal vision performance case uses layers 1 through 25 in order.
Other performance boundaries use their one production weight binding.
The case manifest binds the ordered weight/state identities and operation count.
Time the complete pass and calculate its complete resource work.
Do not divide only one side of the ratio by the operation count.

## V1: Build the fixture and adapter framework

**Dependencies:** None. Start on the current runtime before source migration.

**Create:** `development/operators/{__init__,__main__,cases,run,fixtures,test_contracts}.py`, `cases.json`, and `adapters/{fa,gdn,ffn,output,mtp,vision}.py`.
Modify the pre-commit CPU test command.
Adapters initially translate current buffer interfaces into named observations.
Remove that translation as each public operator migrates.

**Interfaces:** `Case` records its ID, boundary, shape, mode, weight names, fixture path, and expected dispatch conditions.
`Fixture` contains named inputs, initial state, expected outputs, exact side effects, and a validated manifest.
`run.py` supplies:

```python
prepare(case: Case, device: Device, fixture: Fixture, *, model: Path) -> PreparedCase
PreparedCase.reset() -> None
PreparedCase.invoke() -> None
PreparedCase.observe() -> dict[str, numpy.ndarray]
PreparedCase.close() -> None
```

- [x] Add `test_fixture_loading_is_strict_and_reference_free` to `test_contracts.py`.
- [x] Check missing shards, changed hashes, incomplete fields, and forbidden PyTorch imports during usual tests.
- [x] Add `test_case_limits_reject_shape_sweeps` for more than five cases or an unlisted shape.
- [x] Run the checks before implementation.
- [x] Implement cases, stable-address fixtures, named observations, and cleanup.
- [x] Add the CPU checks to the existing pre-commit command.
- [x] Get a different agent to examine the framework.
- [x] Commit the framework.

## V2: Generate and commit numerical fixtures

**Dependencies:** V1. Complete before interface migration.

**Create:** `development/operators/reference/{__init__,__main__,fa,gdn,ffn,output,mtp,vision}.py`, `numerics.py`, `limits.json`, and the fixture directories.
Use the existing reference dependency lock.
Modify `test_contracts.py`.

Modify `python/oh_my_vllm/kernels/sample.py` and `python/oh_my_vllm/kernels/cuda_backend/sample.cu` for a necessary preparation step.
The current implementation has no distribution-only path without token selection and RNG changes.
Add `Sample.distribution(history, mask, parameters, *, greedy=False)` after vocabulary projection.
It returns the prepared probability tensor without token output or an RNG argument.
Add the native `sample_distribution` export and update the function list and manifests.
Keep the existing fused sampling path.

**Interfaces:** Each family supplies `reference(case, inputs, weights) -> dict[str, numpy.ndarray]` using direct PyTorch formulas.
`numerics.compare(actual, fixture, limits) -> NumericalResult` returns field metrics and exact-contract failures.
The generator writes a new staging directory and never overwrites tracked fixtures.

- [ ] Add `test_numerics_rejects_wrong_state_and_missing_limits` with a wrong checkpoint, a guard write, and an absent budget.
- [ ] Check reference formulas with zero-weight projections, one-visible-key attention, and analytical recurrence/acceptance examples.
- [ ] Run these checks before implementation.
- [ ] Implement the distribution-only path before numerical acceptance or performance calibration of that boundary.
- [ ] Implement direct PyTorch references and full-field comparison.
- [ ] Generate the fixed inputs and expected arrays in the reference environment.
- [ ] Get a different agent to examine formulas, numerical budgets, checkpoint identity, and complete fixture manifests.
- [ ] Commit reviewed NumPy shards, manifests, source, and numerical limits together.
- [ ] Run numerical acceptance in the inference environment without PyTorch.

## V3: Implement theoretical-latency acceptance

**Dependencies:** V1 and V2. Complete before source migration.

**Create:** `development/operators/{accounting,timing}.py`.
Modify the CLI, limits, and `test_contracts.py`.
Use rate values and formula helpers from `development/serving/accounting.py` without changing HTTP accounting behavior.

**Interfaces:** `accounting.work(case) -> ResourceWork` gives semantic resource totals.
`accounting.lower_bound(case, device_spec) -> float` returns seconds.
`timing.measure(prepared, policy) -> TimingSamples` records raw GPU/host samples.
`timing.check(samples, lower_bound, max_ratio) -> PerformanceResult` gives the ratio, interval, and decision.

- [ ] Add `test_accounting_ignores_implementation_work` with extra scratch and duplicated launches that cannot increase theoretical credit.
- [ ] Add `test_ratio_gate_excludes_reset_and_fails_uncertainty` with fake events and fixed timing samples.
- [ ] Check cache-resident versus streaming accounting, reset exclusion, fixed-limit failure, and unresolved intervals.
- [ ] Run the checks before implementation.
- [ ] Implement accounting and measurement according to the microbench design.
- [ ] Calibrate proposed limits from the current correct implementation using the fixed five-session procedure.
- [ ] Get a different agent to examine resource counts, cache policy, measured ratios, and proposed limits.
- [ ] Commit only approved limits and their provenance hashes, with the measurement tools.
- [ ] Run performance acceptance without an old build or old timing input.

## V4: Keep integration acceptance

**Dependencies:** V2. Complete related endpoint checks before the affected operator migration.

**Modify:** `development/model/{capture,compare,__main__,test_identity,test_logits}.py` and existing worker behavior tests.
Create `development/model/{cases,reference_cases}.py` and repository fixtures for added endpoints.
Add `--case` to the model CLI. Its default keeps the current short-case behavior.
Keep `reference.py`, `reference.npz`, and the existing short-case identity check unchanged.

- [ ] Use named logits and all seven native module identities in numerical capture.
- [ ] Add `mtp-restored`, `image-video`, and `context-32769` full-model cases before changes to their execution paths.
- [ ] Compare complete vocabulary endpoints after target-plus-MTP computation, media-to-language computation, and long-context decode, respectively.
- [ ] Apply the repository fixture and explicit-generation rules to those added full-model references.
- [ ] Keep the existing short-case limits unchanged.
- [ ] Keep execution lifetime, stop truncation, constraints, service, and Rust checks.
- [ ] Run the 24 HTTP configurations with unchanged workload settings.
- [ ] Update the two README performance tables from verified HTTP artifacts.
- [ ] Get a different agent to examine the integration changes.
- [ ] Commit the integration changes.

## Commands and acceptance

These operator commands are target interfaces to implement.
Set local model/environment paths and new external report paths before use.

```bash
scripts/with-env.sh python -m unittest development.operators.test_contracts
OH_MY_VLLM_CONDA_PREFIX="$REFERENCE_ENV" scripts/with-env.sh scripts/with-gpu.sh python -m development.operators.reference --model "$MODEL_ROOT" --output "$FIXTURE_STAGING"
scripts/with-env.sh scripts/with-gpu.sh python -m development.operators numerics --model "$MODEL_ROOT" --report "$NUMERICAL_REPORT"
scripts/with-env.sh scripts/with-gpu.sh python -m development.operators calibrate --model "$MODEL_ROOT" --report "$CALIBRATION_REPORT"
scripts/with-env.sh scripts/with-gpu.sh python -m development.operators performance --model "$MODEL_ROOT" --report "$PERFORMANCE_REPORT"
scripts/with-env.sh pre-commit run --all-files
```

Only explicit fixture generation uses the reference environment.
Calibration proposes limits in its external report. It cannot change tracked limits.
Usual commands load repository case definitions, fixtures, and approved limits without shape overrides.
Numerical failures, failed ratios, missing results, and unresolved results exit with a nonzero status.

Use the existing [short-logit commands](development.md#numerical-commands) and [HTTP command](../performance.md#measurement-command).
Use Compute Sanitizer independently of timing for changed device memory or synchronization.
Stop all processes and release Graph owners after each run, also on failures.
