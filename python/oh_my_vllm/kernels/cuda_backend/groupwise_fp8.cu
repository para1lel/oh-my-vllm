/*
 * Copyright (c) 2025 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef OH_MY_VLLM_GEMM_GROUPWISE_SM100_CUH_
#define OH_MY_VLLM_GEMM_GROUPWISE_SM100_CUH_

#include <type_traits>
#include <utility>

#include <cuda_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/extra/cuda/device_guard.h>
#include <cstdlib>
#include <cstring>
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/util/packed_stride.hpp>
#include "groupwise_accum.cuh"

// Adapted from FlashInfer 0.6.18.post1 gemm_groupwise_sm100.cuh.
// The project owns launch policy, initialization stream, and workspace lifetime.
using tvm::ffi::TensorView;

namespace oh_my_vllm {

namespace gemm {

using namespace cute;

template <int ScaleGranularityM, int ScaleGranularityN, int ScaleGranularityK, bool ScaleMajorK,
          int MmaSM, typename DTypeIn, typename DTypeOut, int PipelineStages = 0,
          int ClusterN = 1, int TileK = 128, bool ConversionOnly = false>
cudaError_t CutlassGroupwiseScaledGEMMSM100(void* float_buffer, size_t float_buffer_size_in_bytes,
                                            DTypeIn* A_ptr, DTypeIn* B_ptr, float* SFA_ptr,
                                            float* SFB_ptr, DTypeOut* D_ptr, int m, int n, int k,
                                            int l, cudaStream_t stream, bool pdl) {
  using ElementA = DTypeIn;                   // Element type for A matrix operand
  using LayoutA = cutlass::layout::RowMajor;  // Layout type for A matrix operand
  constexpr int AlignmentA =
      128 / cutlass::sizeof_bits<ElementA>::value;  // Memory access granularity/alignment of A
                                                    // matrix in units of elements (up to 16 bytes)

  // B matrix configuration
  using ElementB = DTypeIn;                      // Element type for B matrix operand
  using LayoutB = cutlass::layout::ColumnMajor;  // Layout type for B matrix operand
  constexpr int AlignmentB =
      128 / cutlass::sizeof_bits<ElementB>::value;  // Memory access granularity/alignment of A
                                                    // matrix in units of elements (up to 16 bytes)

  // C/D matrix configuration
  using ElementC = DTypeOut;                  // Element type for C and D matrix operands
  using LayoutC = cutlass::layout::RowMajor;  // Layout type for C and D matrix operands
  constexpr int AlignmentC =
      128 / cutlass::sizeof_bits<ElementC>::value;  // Memory access granularity/alignment of A
                                                    // matrix in units of elements (up to 16 bytes)

  using ElementD = ElementC;
  using LayoutD = LayoutC;
  constexpr int AlignmentD = AlignmentC;

  // MMA type
  using ElementAccumulator = float;  // Element Accumulator will also be our scale factor type
  using ElementCompute = float;

  using MmaTileShape_MNK = Shape<cute::Int<MmaSM * 128>, _128, cute::Int<TileK>>;
  using ClusterShape_MNK = Shape<cute::Int<MmaSM>, cute::Int<ClusterN>, _1>;

  // NOTE(Zihao):: UMMA::Major::MN, UMMA::Major::MN is the fastest configuration.

  using ScaleConfig = std::conditional_t<
      ScaleMajorK,
      cutlass::detail::Sm100BlockwiseScaleConfig<ScaleGranularityM, ScaleGranularityN,
                                                 ScaleGranularityK, UMMA::Major::K, UMMA::Major::K>,
      cutlass::detail::Sm100BlockwiseScaleConfig<ScaleGranularityM, ScaleGranularityN,
                                                 ScaleGranularityK, UMMA::Major::MN,
                                                 UMMA::Major::MN>>;

  using LayoutSFA =
      decltype(ScaleConfig::deduce_layoutSFA());  // Layout type for SFA matrix operand
  using LayoutSFB =
      decltype(ScaleConfig::deduce_layoutSFB());  // Layout type for SFB matrix operand
  // The selected medium path only rounds the final FP32 accumulator to BF16.
  // It needs no runtime alpha/beta scalars or reads of the old output.
  using IdentityCallbacks = cutlass::epilogue::fusion::Sm90EVT<
      cutlass::epilogue::fusion::Sm90Compute<cutlass::epilogue::thread::Identity,
          ElementD, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>,
      cutlass::epilogue::fusion::Sm90AccFetch>;
  using LinearCombination = cutlass::epilogue::fusion::LinearCombination<
      ElementD, ElementCompute, ElementC, ElementCompute>;
  using EpilogueCallbacks = std::conditional_t<ConversionOnly,
      IdentityCallbacks, LinearCombination>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, MmaTileShape_MNK, ClusterShape_MNK,
      cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementCompute, ElementC,
      LayoutC, AlignmentC, ElementD, LayoutC, AlignmentD,
      cutlass::epilogue::collective::EpilogueScheduleAuto,
      EpilogueCallbacks>::CollectiveOp;

  using StandardMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, ElementA,
      cute::tuple<LayoutA, LayoutSFA>, AlignmentA, ElementB, cute::tuple<LayoutB, LayoutSFB>,
      AlignmentB, ElementAccumulator, MmaTileShape_MNK, ClusterShape_MNK,
      std::conditional_t<PipelineStages == 0,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          cutlass::gemm::collective::StageCount<PipelineStages>>,
      cutlass::gemm::KernelScheduleSm100Blockwise>::CollectiveOp;
  using CollectiveMainloop = std::conditional_t<ConversionOnly,
      PairedGroupwiseAccum<StandardMainloop>, StandardMainloop>;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue,
      void>;  // Default to ClusterLaunchControl (CLC) based tile scheduler

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, l));
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, l));
  auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(m, n, l));
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(m, n, l));

  auto layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(make_shape(m, n, k, l));
  auto layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(make_shape(m, n, k, l));

  typename Gemm::Arguments arguments{cutlass::gemm::GemmUniversalMode::kGemm,
                                     {m, n, k, l},
                                     {
                                         A_ptr,
                                         stride_A,
                                         B_ptr,
                                         stride_B,
                                         SFA_ptr,
                                         layout_SFA,
                                         SFB_ptr,
                                         layout_SFB,
                                     },
                                     {
                                         {},  // epilogue.thread
                                         D_ptr,
                                         stride_C,
                                         D_ptr,
                                         stride_C,
                                     }};
  if constexpr (!ConversionOnly) {
    auto& fusion_args = arguments.epilogue.thread;
    fusion_args.alpha = 1.0f;
    fusion_args.beta = 0.0f;
  }

  // The wide gate/up projection otherwise sweeps activation rows for each
  // output tile. Shape-selected swizzles keep nearby tiles in the same L2
  // working set; arithmetic, scale granularity, and output rounding are equal.
  const bool full_prefill_gate_up = m >= 32144 && n == 34816 && k == 5120 && MmaSM == 2;
  const bool full_prefill_qkvz = m >= 32144 && n == 16384 && k == 5120;
  const bool full_prefill_down = m >= 32144 && n == 5120 && k == 17408;
  arguments.scheduler.max_swizzle_size = full_prefill_gate_up ? 16 :
                                         m >= 2048 && n >= 32768 ? 8 :
                                         full_prefill_qkvz ? 16 :
                                         full_prefill_down ? 8 : 0;

  Gemm gemm;

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  TVM_FFI_ICHECK(float_buffer_size_in_bytes >= workspace_size)
      << "groupwise FP8 workspace is too small: " << workspace_size;
  void* workspace_ptr = float_buffer;
  auto check = [](cutlass::Status status) {
    TVM_FFI_ICHECK(status == cutlass::Status::kSuccess)
        << "groupwise FP8 CUTLASS error: " << cutlassGetStatusString(status);
  };
  check(gemm.can_implement(arguments));
  // Keep any initialization on the caller stream. This pinned CLC scheduler
  // currently uses zero workspace; future nonzero requirements fail the guard.
  check(gemm.initialize(arguments, workspace_ptr, stream));
  check(gemm.run(stream, nullptr, pdl));
  return cudaSuccess;
}

}  // namespace gemm
}  // namespace oh_my_vllm

// The owned route supports the checkpoint's regular BF16-output FP8 matrices.
// Small M stays on the independently validated TRT path until its replacement
// has numerical and performance evidence.
void groupwise_fp8(TensorView a, TensorView weight, TensorView scale_a,
                   TensorView scale_weight, TensorView output,
                   TensorView workspace, int64_t mma_sm, bool pdl, bool scale_major_k) {
  auto device = a.device();
  TVM_FFI_ICHECK(device.device_type == kDLCUDA)
      << "owned groupwise FP8 requires CUDA tensors";
  tvm::ffi::CUDADeviceGuard device_guard(device.device_id);
  int major = 0, minor = 0;
  TVM_FFI_ICHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                                        device.device_id) == cudaSuccess);
  TVM_FFI_ICHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor,
                                        device.device_id) == cudaSuccess);
  TVM_FFI_ICHECK(major == 10 && minor == 0);
  auto stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(device.device_type, device.device_id));
  TVM_FFI_ICHECK(a.ndim() == 2 && weight.ndim() == 2 && output.ndim() == 2);
  TVM_FFI_ICHECK(a.size(0) > 32 && a.size(1) == weight.size(1));
  TVM_FFI_ICHECK(a.size(1) > 0 && weight.size(0) > 0 &&
                 a.size(1) % 128 == 0 && weight.size(0) % 128 == 0);
  TVM_FFI_ICHECK(a.size(0) <= INT32_MAX && a.size(1) <= INT32_MAX &&
                 weight.size(0) <= INT32_MAX);
  int m = a.size(0), n = weight.size(0), k = a.size(1);
  TVM_FFI_ICHECK(output.size(0) == m && output.size(1) == n);
  TVM_FFI_ICHECK(scale_a.ndim() == 2 && scale_a.size(0) == m &&
                 scale_a.size(1) == k / 128);
  TVM_FFI_ICHECK(scale_weight.ndim() == 2 && scale_weight.size(0) == n / 128 &&
                 scale_weight.size(1) == k / 128);
  TVM_FFI_ICHECK(workspace.ndim() == 1 && workspace.dtype().code == kDLUInt &&
                 workspace.dtype().bits == 8 && workspace.dtype().lanes == 1);
  TVM_FFI_ICHECK(a.dtype().code == kDLFloat8_e4m3fn &&
                 a.dtype().bits == 8 && weight.dtype().bits == 8 &&
                 weight.dtype().code == kDLFloat8_e4m3fn &&
                 output.dtype().code == kDLBfloat && output.dtype().bits == 16 &&
                 scale_a.dtype().code == kDLFloat && scale_a.dtype().bits == 32 &&
                 scale_weight.dtype().code == kDLFloat && scale_weight.dtype().bits == 32);
  for (auto tensor : {a, weight, scale_a, scale_weight, output, workspace}) {
    auto d = tensor.device();
    TVM_FFI_ICHECK(d.device_type == device.device_type && d.device_id == device.device_id);
    TVM_FFI_ICHECK(tensor.dtype().lanes == 1);
    if (!scale_major_k && tensor.ndim() == 2 && tensor.dtype().code == kDLFloat) {
      TVM_FFI_ICHECK(tensor.stride(0) == 1 && tensor.stride(1) == tensor.size(0) &&
                     tensor.stride(1) % 4 == 0)
          << "MN scale layout requires four-float leading alignment";
    } else {
      TVM_FFI_ICHECK(tensor.stride(tensor.ndim() - 1) == 1);
      if (tensor.ndim() == 2)
        TVM_FFI_ICHECK(tensor.stride(0) == tensor.size(1));
    }
    TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % 16 == 0);
  }
  TVM_FFI_ICHECK(scale_major_k || (mma_sm == 2 && m >= 32144 && n == 34816 && k == 5120))
      << "MN scales require the full two-SM gate/up shape";
  auto run = [&](auto sm, auto stages, auto major_k, auto cluster_n, auto tile_k,
                 auto conversion_only) {
    return oh_my_vllm::gemm::CutlassGroupwiseScaledGEMMSM100<
        1, 128, 128, decltype(major_k)::value, decltype(sm)::value, cutlass::float_e4m3_t,
        cutlass::bfloat16_t, decltype(stages)::value, decltype(cluster_n)::value,
        decltype(tile_k)::value, decltype(conversion_only)::value>(
        workspace.data_ptr(), workspace.size(0),
        static_cast<cutlass::float_e4m3_t*>(a.data_ptr()),
        static_cast<cutlass::float_e4m3_t*>(weight.data_ptr()),
        static_cast<float*>(scale_a.data_ptr()),
        static_cast<float*>(scale_weight.data_ptr()),
        static_cast<cutlass::bfloat16_t*>(output.data_ptr()), m, n, k, 1, stream, pdl);
  };
  TVM_FFI_ICHECK(mma_sm == 1 || mma_sm == 2);
  bool debug = [] {
    const char* value = std::getenv("OH_MY_VLLM_CUDA_DEBUG_SYNC");
    return value && std::strcmp(value, "1") == 0;
  }();
  if (debug) {
    cudaStreamCaptureStatus capture{};
    TVM_FFI_ICHECK(cudaStreamIsCapturing(stream, &capture) == cudaSuccess &&
                   capture == cudaStreamCaptureStatusNone)
        << "groupwise FP8 debug sync cannot execute during graph capture";
    auto prior = cudaPeekAtLastError();
    TVM_FFI_ICHECK(prior == cudaSuccess)
        << "groupwise FP8 prior CUDA error: " << cudaGetErrorString(prior);
    prior = cudaStreamSynchronize(stream);
    TVM_FFI_ICHECK(prior == cudaSuccess)
        << "groupwise FP8 prior execution error: " << cudaGetErrorString(prior);
  }
  // Full MN gate/up uses K=256 and three stages to decrease local-memory traffic.
  // K-major gate/up keeps K=128 and five stages as its layout reference.
  // Other shapes keep the template's automatic storage calculation.
  // Measured medium projections multicast activation tiles across two N CTAs.
  // Keep the original scale layout, pipeline stages, swizzle, and arithmetic.
  const bool medium_projection = m >= 624 && m <= 2496 &&
      ((k == 5120 && (n == 34816 || n == 16384 || n == 14336)) ||
       (n == 5120 && (k == 17408 || k == 6144)));
  auto status = mma_sm == 1
                    ? (medium_projection
                         ? run(std::integral_constant<int, 1>{},
                               std::integral_constant<int, 5>{}, std::true_type{},
                               std::integral_constant<int, 2>{},
                               std::integral_constant<int, 128>{}, std::true_type{})
                         : run(std::integral_constant<int, 1>{},
                               std::integral_constant<int, 0>{}, std::true_type{},
                               std::integral_constant<int, 1>{},
                               std::integral_constant<int, 128>{}, std::false_type{}))
                : m >= 32144 && n == 34816 && k == 5120
                    ? (scale_major_k
                         ? run(std::integral_constant<int, 2>{},
                               std::integral_constant<int, 5>{}, std::true_type{},
                               std::integral_constant<int, 1>{},
                               std::integral_constant<int, 128>{}, std::false_type{})
                         : run(std::integral_constant<int, 2>{},
                               std::integral_constant<int, 3>{}, std::false_type{},
                               std::integral_constant<int, 1>{},
                               std::integral_constant<int, 256>{}, std::false_type{}))
                    : run(std::integral_constant<int, 2>{},
                          std::integral_constant<int, 0>{}, std::true_type{},
                          std::integral_constant<int, 1>{},
                          std::integral_constant<int, 128>{}, std::false_type{});
  TVM_FFI_ICHECK(status == cudaSuccess) << cudaGetErrorString(status);
  if (debug) {
    auto result = cudaStreamSynchronize(stream);
    TVM_FFI_ICHECK(result == cudaSuccess)
        << "groupwise FP8 execution error: " << cudaGetErrorString(result);
  }
}
#endif
