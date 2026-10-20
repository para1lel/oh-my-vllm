# Operator refactor implementation plan

> Agent instructions: Use `superpowers:executing-plans` or `superpowers:subagent-driven-development` after selection of the execution method. Complete the checkbox steps in sequence.

**Goal:** Put each operator's interface, plan, temporary storage, and CUDA code in one package.

**Architecture:** Use the approved [design](design.md). Keep the seven native modules during source migration. Change interfaces in dependency sequence.

**Tech Stack:** Python, Rust, TVM FFI, CUDA 13.1, CUTLASS 4.8.0, and one NVIDIA B200.

**Spec:** [Operator structure design](design.md).

## Global constraints

- Use `main` and the [contribution procedure](../../CONTRIBUTING.md).
- Keep the [runtime contracts](../requirements.md), [model precision](model.md), and [execution rules](../execution.md).
- Keep inference free of development imports and compilation.
- Use family-specific types. Do not add an operator base class or backend registry.
- Pass weights explicitly. Keep persistent state ownership in the worker and Rust.
- Give named borrowed outputs before recording. Keep their addresses fixed through replay.
- Use temporary storage again across sequential layers. Keep all recorded resource references until Graph release.
- Keep the frozen short reference and its numerical limits. Use the linked [validation plan](validation-plan.md) before interface changes.
- Apply fixed microbench latency-ratio limits. Use all 24 HTTP configurations for integration checks.

## Review focus

| Condition | Necessary behavior | Task |
|---|---|---|
| Includes cross package directories | Source hashes include all transitive dependencies | 1 |
| Equal buffer sizes, different recorded launches | Plan signatures prevent incorrect Graph reuse | 3 |
| Many layer weights, failed recording | Owners survive until ordered Graph release | 3 |
| Stop truncation and three RNG domains | Effective counts control state selection. Counters stay isolated | 5 |
| First/last vision layers, odd frame counts | Full-model endpoints and media order stay correct | 6 |

## File and interface map

Complete validation tasks V1–V3 on the current runtime before task 1.
Their reviewed fixtures and ratio limits control subsequent migration.

Python package paths below are relative to `python/oh_my_vllm`.
Development, test, and tool paths are relative to the repository root.

Use `python/oh_my_vllm/operators` as the new package root.
Create `api.py`, `plan.py`, and `workspace.py` in each of `fa`, `gdn`, `ffn`, `output`, `mtp`, and `vision`.
Put CUDA stages in each family's `cuda` directory.
Keep shared resource code in `runtime`. Put shared computation in `primitives`.

Each family defines its own `Weights`, `Plan`, `Workspace`, inputs, views, and outputs.
Keep tensor shapes and datatypes from the current native contracts.
Use frozen data classes for plans and views.
Use the existing `Device` allocation interface during preparation.
Plan construction must not alfind GPU storage or execute GPU work.

The naming convention does not add a base class:

```python
prepare(plan: Plan, device: Device, weights: tuple[Weights, ...]) -> Workspace
workspace.outputs: Outputs
workspace.graph_lease()  # Existing context-manager lifetime contract.
```

Preparation creates storage and native contexts and does family-specific warmup.
Its weight tuple contains all layer bindings used by that operator workspace.
The worker puts family `plan.reuse_key` values together and checks resource identities independently.
Plans supply storage descriptions for startup sizing. Constructors with counting allocators must no longer select implementations.

## Task 1: Move sources and their tools

**Dependencies:** Validation tasks V1–V3. **Result:** The existing computation runs from the new directories.

**Files:** Move `kernels/native.py` to `operators/runtime/native.py` and `worker/device.py` to `operators/runtime/device.py`.
Move `kernels/specification.py` to `operators/specification.py` and the kernel directory instructions to `operators`.
Move each wrapper to its family's `api.py`. Move `sample.py` to `output/api.py`.
Move `device.cu` to `runtime/cuda/device.cu` and the other six translation units to their family directories.
Put `sample.cu` in `output/cuda`.

**Interfaces:** Keep native module names and exported functions unchanged in this task.
Add `MODULE_SOURCES: dict[str, Path]` to `runtime/native.py`.
Use this mapping in `development/native_build.py` and `source_fingerprint(name)`.

- [ ] Check the pre-change manifests and measurements from validation task V3.
- [ ] Extend `development/model/test_identity.py::test_native_identity_tracks_only_transitive_dependencies` with sibling-directory includes and an include cycle.
- [ ] Check that a transitive edit changes the digest and an unrelated edit does not.
- [ ] Move family-prefixed CUDA headers with their family. Move shared headers according to their existing users and responsibility.
- [ ] Hash dependency paths relative to the operator source root, together with file contents.
- [ ] Change quoted includes, builder include roots, all inference/development/test imports, and source links to the new paths.
- [ ] Update `scripts/format_cuda.py` and `.pre-commit-config.yaml` so the 800-line CUDA limit still applies.
- [ ] Adapt `tests/test_execution.py` to the new Device import without changing its failure assertions.
- [ ] Complete the delivery procedure below with the existing short FP64 endpoints and all seven rebuilt modules.

## Task 2: Add full-model validation inputs

**Dependencies:** Task 1. **Result:** Numerical evidence covers the paths that subsequent tasks change.

**Files and interfaces:** Use validation task V4 in [the validation plan](validation-plan.md).

- [ ] Add MTP and long-context endpoints before FA/MTP changes.
- [ ] Add media endpoints before vision changes.
- [ ] Examine reference generation and artifact identities independently of inference changes.
- [ ] Complete the delivery procedure with unchanged inference behavior.

## Task 3: Apply FA plans and MTP component interfaces

**Dependencies:** Tasks 1 and 2's MTP/long-context checks. **Result:** FA and its MTP consumers use explicit weights and named storage.

**Files:** `operators/fa/{api,plan,workspace,components}.py`, `operators/mtp/{api,plan,workspace}.py`, and `worker/{model,weights,metadata,speculation,draft_chain}.py`.
Modify `tests/test_execution.py` and the draft-chain fixtures in `tests/test_constraints.py`.

**Interfaces:** FA defines these functions:

```python
make_plan(*, rows: int, max_queries: int, max_context: int,
          min_context: int, variant: str = "full", first_token: int = 0) -> Plan
run(weights: Weights, workspace: Workspace, inputs: Inputs,
    batch: BatchView, state: CacheView) -> Outputs
```

`Inputs` contains `value` and `residual`. Full FA outputs have those same named fields.
`BatchView` contains `positions`, `sequence_rows`, and `lengths`.
`CacheView` contains `cache`, `slots`, and `tables`.
Private component types in `components.py` give MTP its projection and attention storage contracts.

- [ ] Add `test_fa_plan_separates_recorded_launches` for variants, rows 32/33 and 128/129, and dispatch boundaries 1024, 12544, and 32768.
- [ ] Check unequal launch conditions with equal storage sizes. Check compatible metadata updates without a new plan.
- [ ] Add `test_operator_workspace_keeps_outputs_and_graph_owners` with a fake device and multiple layer weights.
- [ ] Check fixed output identities, no allocation during recording, shared sequential scratch, and release after Graph destruction.
- [ ] Run the new checks before implementation to show the missing behavior.
- [ ] Move dispatch and preparation decisions from `worker/model.py` into FA plans.
- [ ] Replace MTP buffer-index access and copied FA layouts with the explicit component types.
- [ ] Keep named private adapters for the old FFN/output components until tasks 4 and 5 remove them.
- [ ] Complete the delivery procedure with FA/MTP microbench cases and V4's full-model endpoints.

## Task 4: Apply GDN and FFN contracts

**Dependencies:** Task 3. **Result:** All target layers use explicit prepared weights and reusable operator storage.

**Files:** `operators/{gdn,ffn}/{api,plan,workspace}.py` and `worker/{model,inference,weights,metadata,cache,speculation}.py`.

**Interfaces:** FFN uses `run(weights, workspace, inputs) -> Outputs` with family-specific types and named `value`/`residual` fields.
GDN uses `run(weights, workspace, inputs, batch, state) -> Outputs`.
Its views use the existing `Schedule` fields and recurrent/convolution tensors.
Its inputs also name optional token and media tensors.

- [ ] Give hidden export a worker-owned named write-slot view before removal of `gdn.metadata[2]` access.
- [ ] Move GDN chunk selection, metadata preparation, and FFN dispatch into family plans.
- [ ] Replace startup counting constructors with plan storage descriptions in `worker/model.py` and `worker/inference.py`.
- [ ] Remove mutable weight bindings and FFN private-buffer access from consumers.
- [ ] Apply task 3's lifetime assertions to the GDN/FFN preparation paths.
- [ ] Complete the delivery procedure with GDN/FFN microbench cases and V4's short, chunk-boundary, and long-context endpoints.

## Task 5: Split output and MTP acceptance

**Dependencies:** Task 4. **Result:** Output computation and candidate acceptance have the approved semantic boundaries.

**Files:** `operators/{output,mtp}/{api,plan,workspace}.py`, their CUDA sources, `operators/runtime/{device.py,cuda/device.cu}`, and `operators/primitives/cuda`.
Modify `worker/{model,sampling,speculation,draft_chain,inference}.py`, `development/model/capture.py`, and `tests/test_constraints.py`.

**Interfaces:** Define `output.hidden`, `output.distribution`, `output.sample`, `mtp.context`, `mtp.draft`, and `mtp.accept` in their `api.py` files.
Each takes explicit family weights when applicable, a prepared operator workspace, and named input/view types.
Output fields are `hidden`, `logits`, `probabilities`, `tokens`, and `selected_probability`, as applicable.
Acceptance fields are `tokens`, `counts`, and `candidate_slots`.

- [ ] Add `test_stop_truncation_controls_accepted_slot`: truncate four candidates to one token and check the effective cache commit count.
- [ ] Check that a stop before the first returned token cannot commit the raw candidate slot.
- [ ] Run the test before changes to show the uncovered contract.
- [ ] Move `final_hidden` to the existing sample native module. Keep generic row gather with primitives.
- [ ] Move `mtp_accept` to the existing MTP native module with its vocabulary, CUB, and Philox dependencies.
- [ ] Change function lists, loader calls, and manifests together with these exports.
- [ ] Move V2's distribution-only path into the new API. Keep sample's fused path when compatible.
- [ ] Hidden export must accept all input rows when vocabulary projection uses fewer rows.
- [ ] Remove the temporary component adapters and use named logits in numerical capture.
- [ ] Complete the delivery procedure with output/MTP microbench cases, V4 endpoints, RNG checks, and state-selection checks.

## Task 6: Apply vision contracts

**Dependencies:** Task 5 and V4's media endpoints. **Result:** Vision has family-owned planning and temporary storage with unchanged media semantics.

**Files:** `operators/vision/{api,plan,workspace}.py`, its CUDA sources, and `worker/media.py`.

**Interfaces:** `run(weights: Weights, workspace: Workspace, inputs: Inputs) -> Outputs` returns named `hidden`.
Plans give input, transformer, and merge stage conditions, grid/frame capacities, and native context ownership.

- [ ] Move constructor validation and native-context preparation into the vision plan/preparation path.
- [ ] Keep media decode, request order, and position construction in the worker.
- [ ] Check first/last layers and odd frame counts with the fixed vision microbench cases.
- [ ] Check invalid grids and the 65536-row capacity with CPU plan validation.
- [ ] Complete the delivery procedure with V4's media endpoints and representative media timing.

## Delivery procedure for each task

- [ ] Run the applicable validation commands and full hooks from the validation plan.
- [ ] Update `operators/specification.py` and the bilingual [serving boundaries](fusion.md) when their facts change.
- [ ] Update the two README performance tables from verified measurements.
- [ ] Get a different agent to examine the code and documents.
- [ ] Correct the findings.
- [ ] Update the bilingual handoff without test logs.
- [ ] Commit only the task files with a Conventional Commits title.
- [ ] Push `main` with hooks active.

After task 6, remove unused compatibility files and the old kernel package.
Check imports and direct buffer access across inference, development tools, and tests.
Evaluate additional fusion changes as later changes with their own complete validation.
