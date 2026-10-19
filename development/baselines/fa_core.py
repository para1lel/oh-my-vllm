import functools
import hashlib
import os
import subprocess
from pathlib import Path

from oh_my_vllm.kernels.native import SOURCES, cache_directory, source_files

from .inputs import Prepared

CUDA_SOURCE = r"""
#define fa_attention_core_launch fa_baseline_attention_core_launch
#define fa_prefill_kernel fa_baseline_prefill_kernel
#define fa_prefill_cluster_kernel fa_baseline_prefill_cluster_kernel
#define fa_q1_warp_partial_kernel fa_baseline_q1_warp_partial_kernel
#define fa_partial_kernel fa_baseline_partial_kernel
#include "fa_core.cuh"
#undef fa_attention_core_launch
#undef fa_prefill_kernel
#undef fa_prefill_cluster_kernel
#undef fa_q1_warp_partial_kernel
#undef fa_partial_kernel

template <int Warps, int Groups = 1>
__global__ void fa_core_merge_kernel(const float *partial, const float *lse,
                                     __nv_bfloat16 *output, int splits) {
  static_assert(Groups == 1 || Warps == 1);
  int local_group = threadIdx.x / (Warps * 32);
  int group = blockIdx.x * Groups + local_group, head_row = group / 2, half = group % 2;
  int lane = threadIdx.x & 31, warp = (threadIdx.x / 32) % Warps;
  __shared__ float group_weights[Groups][128];
  __shared__ float4 group_totals[Groups][Warps][32];
  auto &weights = group_weights[local_group];
  auto &totals = group_totals[local_group];
  float denominator = 0;
  if (warp == 0) {
    float maximum = -INFINITY;
    for (int i = lane; i < splits; i += 32)
      maximum = fmaxf(maximum, lse[int64_t(head_row) * splits + i]);
#pragma unroll
    for (int offset = 16; offset; offset >>= 1)
      maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, offset));
    float safe = maximum == -INFINITY ? 0.f : maximum;
    for (int i = lane; i < splits; i += 32) {
      float weight = fa_exp2(lse[int64_t(head_row) * splits + i] - safe);
      weights[i] = weight;
      denominator += weight;
    }
    denominator = warp_sum(denominator);
  }
  if constexpr (Warps == 1)
    __syncwarp();
  else
    __syncthreads();
  float values[4] = {};
  for (int i = warp; i < splits; i += Warps) {
    float4 part = *reinterpret_cast<const float4 *>(
        partial + (int64_t(head_row) * splits + i) * 256 + half * 128 + lane * 4);
    values[0] += part.x * weights[i];
    values[1] += part.y * weights[i];
    values[2] += part.z * weights[i];
    values[3] += part.w * weights[i];
  }
  if constexpr (Warps > 1) {
    totals[warp][lane] = make_float4(values[0], values[1], values[2], values[3]);
    __syncthreads();
    if (warp != 0)
      return;
    values[0] = values[1] = values[2] = values[3] = 0;
#pragma unroll
    for (int w = 0; w < Warps; ++w) {
      float4 part = totals[w][lane];
      values[0] += part.x;
      values[1] += part.y;
      values[2] += part.z;
      values[3] += part.w;
    }
  }
#pragma unroll
  for (int j = 0; j < 4; ++j)
    output[int64_t(group) * 128 + lane * 4 + j] =
        __float2bfloat16_rn(denominator > 0 ? values[j] / denominator : 0.f);
}

void attention(TensorView query, TensorView cache, TensorView tables,
               TensorView lengths, TensorView starts, TensorView partial,
               TensorView lse, TensorView output, int64_t max_queries,
               int64_t max_context, int64_t min_context) {
  int rows = query.size(0), batch = tables.size(0), splits = lse.size(2);
  tensor_contract(query, query, kDLBfloat, 16, {rows, 24, 256});
  tensor_contract(cache, query, kDLBfloat, 16, {cache.size(0), 2, 784, 4, 256});
  tensor_contract(tables, query, kDLInt, 32, {batch, tables.size(1)});
  tensor_contract(lengths, query, kDLInt, 32, {rows});
  tensor_contract(starts, query, kDLInt, 32, {batch + 1});
  tensor_contract(partial, query, kDLFloat, 32, {rows, 24, splits, 256});
  tensor_contract(lse, query, kDLFloat, 32, {rows, 24, splits});
  tensor_contract(output, query, kDLBfloat, 16, {rows, 24, 256});
  for (auto destination : {partial, lse, output})
    for (auto source : {query, cache, tables, lengths, starts})
      disjoint(destination, source);
  disjoint(partial, lse);
  disjoint(partial, output);
  disjoint(lse, output);
  tvm::ffi::CUDADeviceGuard guard(query.device().device_id);
  auto stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(kDLCUDA, query.device().device_id));
  splits = fa_baseline_attention_core_launch(
      query, cache, tables, lengths, starts, partial, lse, max_queries, 0, stream,
      static_cast<__nv_bfloat16 *>(output.data_ptr()), max_context, min_context);
  auto merge = [&](auto warps) {
    constexpr int Warps = decltype(warps)::value;
    constexpr int Groups = Warps == 1 ? 8 : 1;
    fa_core_merge_kernel<Warps, Groups>
        <<<rows * 48 / Groups, Warps * Groups * 32, 0, stream>>>(
        static_cast<const float *>(partial.data_ptr()),
        static_cast<const float *>(lse.data_ptr()),
        static_cast<__nv_bfloat16 *>(output.data_ptr()), splits);
  };
  if (splits > 4)
    merge(std::integral_constant<int, 8>{});
  else if (splits != 1 || rows <= 32)
    merge(std::integral_constant<int, 1>{});
  check_cuda(cudaGetLastError());
}
"""


@functools.cache
def build_core(fingerprint):
    import tvm_ffi
    from tvm_ffi.cpp import build_inline

    if os.environ.setdefault("TVM_FFI_CUDA_ARCH_LIST", "10.0a") != "10.0a":
        raise ValueError("FA core requires the B200 architecture 10.0a")
    cutlass = Path(os.environ["OH_MY_VLLM_CUTLASS"])
    revision = subprocess.check_output(
        ["git", "-C", str(cutlass), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != "098de2a652cf8f00fd70b2df54051c7eccbb855a":
        raise ValueError("CUTLASS must use the pinned v4.8.0 commit")
    if subprocess.check_output(["git", "-C", str(cutlass), "status", "--porcelain"]):
        raise ValueError("CUTLASS checkout must be clean")
    directory = cache_directory() / "fa_core" / fingerprint
    directory.mkdir(parents=True, exist_ok=True)
    library = build_inline(
        f"oh_my_vllm_fa_core_{fingerprint[:16]}",
        cuda_sources=CUDA_SOURCE,
        functions=("attention",),
        extra_cuda_cflags=(
            "-O3",
            "--generate-code=arch=compute_100a,code=sm_100a",
            "-lineinfo",
            "--expt-relaxed-constexpr",
            "-static-global-template-stub=false",
        ),
        extra_include_paths=[
            str(SOURCES),
            str(cutlass / "include"),
            str(cutlass / "tools/util/include"),
        ],
        build_directory=str(directory),
        backend="cuda",
    )
    return tvm_ffi.load_module(library), str(Path(library).resolve())


def prepare(case, inputs):
    import torch
    import tvm_ffi

    if case.operator != "fa_attention":
        raise ValueError("FA core cannot supply another operator")
    paths = [Path(__file__).resolve(), *source_files(SOURCES / "fa_core.cuh")]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    module, library = build_core(digest.hexdigest())
    pages = tuple((length + 783) // 784 for length in case.contexts)
    table = torch.full(
        (case.batch, max(pages)), -1, device=inputs.q.device, dtype=torch.int32
    )
    first = 0
    for sequence, count in enumerate(pages):
        table[sequence, :count] = inputs.page_indices[first : first + count]
        first += count
    lengths = torch.cat(
        [
            torch.arange(
                context - queries + 1,
                context + 1,
                device=inputs.q.device,
                dtype=torch.int32,
            )
            for queries, context in zip(case.queries, case.contexts, strict=True)
        ]
    )
    if (
        case.rows <= 32
        and max(case.queries) == 1
        and (
            (case.batch >= 4 and max(case.contexts) >= 32768)
            or (case.batch == 2 and min(case.contexts) >= 32768)
        )
    ):
        splits = 128
    else:
        splits = 64 if case.rows <= 32 else 4 if case.rows <= 4096 else 1
    partial = torch.empty(
        (case.rows, 24, splits, 256), device=inputs.q.device, dtype=torch.float32
    )
    lse = torch.empty(
        (case.rows, 24, splits), device=inputs.q.device, dtype=torch.float32
    )
    output = torch.empty_like(inputs.q)
    tensors = (
        inputs.q,
        inputs.cache,
        table,
        lengths,
        inputs.tensors["sequence_offsets"],
        partial,
        lse,
        output,
    )
    arguments = tuple(tvm_ffi.from_dlpack(tensor) for tensor in tensors)

    def run():
        module.attention(
            *arguments, max(case.queries), max(case.contexts), min(case.contexts)
        )

    return Prepared(
        run,
        lambda: None,
        {"output": output},
        {
            "backend": "project_cuda",
            "adaptation": "immutable dense page-table metadata before timing",
            "precision": "BF16 Q/K/V/P/output; FP32 softmax and accumulation",
            "attention_dispatch": (
                "shared with inference; timed split merge and BF16 output"
            ),
            "source_files": [str(path) for path in paths],
            "binaries": [library],
        },
        (module, tensors, arguments),
    )
