#pragma once

#include "common.cuh"
#include <collective/fmha_fusion.hpp>
#include <collective/sm100_fmha_fwd_epilogue_tma_warpspecialized.hpp>
#include <collective/sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp>
#include <device/fmha.hpp>
#include <kernel/fmha_tile_scheduler.hpp>
#include <kernel/sm100_fmha_fwd_kernel_tma_warpspecialized.hpp>

template <bool Normalized>
inline void vision_attention_launch(const __nv_bfloat16 *query, const __nv_bfloat16 *key,
                                    const __nv_bfloat16 *value, __nv_bfloat16 *output,
                                    float *normalizer, int tokens, int groups, int device,
                                    cudaStream_t stream) {
  using namespace cute;
  using Element = cutlass::bfloat16_t;
  using Problem = tuple<int, int, int, tuple<tuple<int, int>, int>>;
  using Stride = tuple<int, _1, tuple<tuple<int, int>, int>>;
  using KeyStride = tuple<int, _1, tuple<tuple<_0, int>, int>>;
  using LogStride = tuple<_1, tuple<tuple<int, int>, int>>;
  using Mainloop = cutlass::fmha::collective::Sm100FmhaFwdMainloopTmaWarpspecialized<
      Element, float, float, Shape<_256, _64, _128>, Stride, KeyStride, KeyStride,
      cutlass::fmha::collective::ResidualMask>;
  using BaseEpilogue = cutlass::fmha::collective::Sm100FmhaFwdEpilogueTmaWarpspecialized<
      cutlass::bfloat16_t, float, typename Mainloop::TileShapePV, Stride, LogStride>;
  using Epilogue = BaseEpilogue;
  using Kernel = cutlass::fmha::kernel::Sm100FmhaFwdKernelTmaWarpspecialized<
      Problem, Mainloop, Epilogue, cutlass::fmha::kernel::IndividualTileScheduler>;
  using Attention = cutlass::fmha::device::FMHA<Kernel>;
  Stride stride =
      make_stride(128, _1{}, make_stride(make_stride(0, tokens * 128), 16 * tokens * 128));
  KeyStride key_stride =
      make_stride(128, _1{}, make_stride(make_stride(_0{}, tokens * 128), 16 * tokens * 128));
  LogStride log_stride = make_stride(_1{}, make_stride(make_stride(0, tokens), 16 * tokens));
  typename Attention::Arguments arguments{};
  arguments.problem_shape = {tokens, tokens, 128, {{1, 16}, groups}};
  arguments.mainloop = {{reinterpret_cast<const Element *>(query), stride,
                         reinterpret_cast<const Element *>(key), key_stride,
                         reinterpret_cast<const Element *>(value), key_stride},
                        1.f / sqrtf(72.f)};
  arguments.epilogue = {reinterpret_cast<cutlass::bfloat16_t *>(output), stride, nullptr,
                        log_stride};
  arguments.hw_info.device_id = device;
  auto checked = [](cutlass::Status status) {
    TVM_FFI_ICHECK(status == cutlass::Status::kSuccess) << cutlassGetStatusString(status);
  };
  TVM_FFI_ICHECK(Attention::get_workspace_size(arguments) == 0);
  Attention attention;
  checked(attention.can_implement(arguments));
  checked(attention.initialize(arguments, nullptr, stream));
  checked(attention.run(stream));
}

__global__ void vision_attention_unpack_kernel(const __nv_bfloat16 *padded, __nv_bfloat16 *output,
                                               int rows, int tokens) {
  int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= int64_t(rows) * 1152)
    return;
  int row = index / 1152, column = index % 1152;
  int head = column / 72, dimension = column % 72;
  int64_t source = ((int64_t(row / tokens) * 16 + head) * tokens + row % tokens) * 128 + dimension;
  output[index] = padded[source];
}
