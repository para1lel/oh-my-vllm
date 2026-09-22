# Custom kernel development

Production custom kernels use TileLang 0.1.14. Independent libraries may use Triton
internally. TileFoundry is an agent development tool, never a serving dependency.
The existing correctness tests and frozen vLLM acceptance gates remain authoritative;
there is no comparison gate against the previous project Triton kernels.

## Reproduce the tools

Run from the project root in the existing oh-my-vllm environment:

```bash
git submodule update --init --recursive 3rdparty/TileFoundry
uv pip install --python /data0/shared/dongwu.chen/conda-envs/oh-my-vllm/bin/python setuptools-scm==10.2.3 vcs-versioning==2.4.1
uv pip install --python /data0/shared/dongwu.chen/conda-envs/oh-my-vllm/bin/python --no-build-isolation -r requirements/development.txt
uv pip check --python /data0/shared/dongwu.chen/conda-envs/oh-my-vllm/bin/python
scripts/with-env.sh tilefoundry --help
```

Install the build metadata tools first: --no-build-isolation requires them
before editable metadata generation, even though they also appear in the lock.

The gitlink pins our para1lel/TileFoundry fork, based on upstream main
b5aaa442161940694e710c0d00c32b5c884faae2. Its oh-my-vllm-integration branch is
the sole exception to this project's main-only policy. Never open an upstream PR.
Local adaptations remove the Transformers upper bound (retaining its minimum)
and add HIR sin/cos evaluation, printing and cost classification for RoPE.
The latter does not add TileFoundry device code generation.
Keep this fork small: fix simple incompatibilities locally; discuss substantial
changes first. Do not maintain TileLang or OR-Tools forks; select compatible
released versions instead.

Runtime uses apache-tvm-ffi 0.1.12 and protobuf 6.33.6 with the existing Torch2.14,
Transformers5.17 and FlashInfer0.6.18.post1. requirements/runtime.txt pins runtime
dependencies; requirements/development.txt adds the editable fork and its tools.
The source tree must remain available for the editable installation.

## Agent workflow

1. Read the production wrapper and existing tests before choosing an operator.
   Preserve its supported shapes, packed strides, dtypes, returned state,
   aliasing rules and numerical rounding boundaries.
2. Express complete logical behavior in development/kernels/semantics.py.
   development/kernels/twins.py calls production code through thin adapters.
   Snapshots and updated logical cache are outputs, not hidden side effects.
3. Use TileFoundry analysis to inspect arithmetic, memory and roofline hypotheses.
   The supplied single logical CTA is an analysis topology, not the actual
   TileLang placement. HIR cost estimates are not measured GPU latency.
4. Implement and tune the TileLang kernel. Keep public Python interfaces,
   scheduler ownership, persistent 784-token pages and the MTP position-zero
   exclusion contract. Internal tiling, fusion and temporary layouts may change.
   Use dynamic token dimensions when lengths do not determine algorithm/layout.
5. Run temporary operator experiments outside this repository, then the existing
   required tests and actual-model acceptance. Remove task-owned GPU programs
   promptly. Do not commit microbenchmark scripts, generated kernels, reports,
   cache contents or experiment records.
6. Review each milestone, fix findings, then commit on main. Push fork changes
   before advancing the main repository's submodule gitlink.

Examples (analysis outputs deliberately go to /tmp):

```bash
scripts/with-env.sh tilefoundry analyze development/kernels/semantics.py:Operators.silu /tmp/oh-my-vllm-silu-analysis.txt --compute-cost --memory --roofline
PYTHONPATH="$PWD:$PWD/python" scripts/with-gpu.sh scripts/with-env.sh tilefoundry check development/kernels/twins.py:TileLang.silu --inputs random --out output --fn allclose --atol .03 --rtol .03
scripts/with-gpu.sh scripts/with-env.sh pytest -q tests
```

Select explicit predicates for each returned tensor when checking tuple outputs.
Representative HIR shapes help explore semantics; they do not replace the existing
FP64 references, irregular metadata, packed views, graph replay, page isolation,
maximum context or actual model checks. Do not relax existing tolerances to match
either TileLang or the previous Triton implementation.

## Acceptance and performance

Retain all documented service, agentic, maximum-context and correctness checks.
Use the twelve frozen EngineCore comparison rows: throughput at least95%, TTFT at
most1.1x, with existing stability and provenance audits. Warm every measured shape
before capture/measurement. Per-run TileLang and third-party Triton cache roots
prevent concurrent compilation from contaminating the audit; FlashInfer must also
be fully warmed and unchanged during measured repetitions.

If a row fails, profile the actual model, identify the dominant cost, make
targeted improvements and retest. Report remaining failures and measured causes
explicitly. A performance shortfall may remain with evidence; a correctness
failure cannot be waived. Future models/backends can supply additional semantic
modules and TileLang specializations without changing Rust scheduling; multi-GPU
and the local DSpark model remain future architecture work.
