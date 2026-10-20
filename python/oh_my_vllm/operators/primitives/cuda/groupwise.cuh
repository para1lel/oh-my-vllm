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
#pragma once

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/util/packed_stride.hpp>

// Derived from the archived project adaptation of FlashInfer 0.6.18.post1.
// Only CUTLASS headers are a build dependency.
template <int TileM, int TileN = 128, int TileK = 128, int ScaleM = 1, int ScaleN = 128,
          bool Transposed = false, int Splits = 1, typename Result = cutlass::bfloat16_t>
void groupwise(const __nv_fp8_e4m3 *a, const __nv_fp8_e4m3 *weight, const float *scale_a,
               const float *scale_weight, void *output, int m, int n, int k, cudaStream_t stream,
               bool dependent = false) {
  using namespace cute;
  using Scale = cutlass::detail::Sm100BlockwiseScaleConfig<ScaleM, ScaleN, 128, UMMA::Major::K,
                                                           UMMA::Major::K>;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using Element = cutlass::float_e4m3_t;
  using LayoutD = std::conditional_t<Transposed, cutlass::layout::ColumnMajor, LayoutA>;
  constexpr int OutputAlignment = 128 / cutlass::sizeof_bits<Result>::value;
  using Tile = Shape<Int<TileM>, Int<TileN>, Int<TileK>>;
  using Cluster = Shape<_1, _1, _1>;
  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, Tile, Cluster,
      cutlass::epilogue::collective::EpilogueTileAuto, float, float, void, LayoutD, OutputAlignment,
      Result, LayoutD, OutputAlignment,
      cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, Element,
      cute::tuple<LayoutA, decltype(Scale::deduce_layoutSFA())>, 16, Element,
      cute::tuple<LayoutB, decltype(Scale::deduce_layoutSFB())>, 16, float, Tile, Cluster,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename Epilogue::SharedStorage))>,
      cutlass::gemm::KernelScheduleSm100Blockwise>::CollectiveOp;
  using Kernel =
      cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Mainloop, Epilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
  auto stride_a = cutlass::make_cute_packed_stride(typename Kernel::StrideA{},
                                                   make_shape(m, k * Splits, Splits));
  auto stride_b = cutlass::make_cute_packed_stride(typename Kernel::StrideB{},
                                                   make_shape(n, k * Splits, Splits));
  auto stride_d =
      cutlass::make_cute_packed_stride(typename Kernel::StrideD{}, make_shape(m, n, Splits));
  auto layout_a = Scale::tile_atom_to_shape_SFA(make_shape(m, n, k * Splits, Splits));
  auto layout_b = Scale::tile_atom_to_shape_SFB(make_shape(m, n, k * Splits, Splits));
  if constexpr (Splits > 1) {
    cute::get<2>(stride_a) = k;
    cute::get<2>(stride_b) = k;
    cute::get<2>(layout_a.stride()) = k / 128;
    cute::get<2>(layout_b.stride()) = k / 128;
  }
  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, Splits},
      {reinterpret_cast<const Element *>(a), stride_a, reinterpret_cast<const Element *>(weight),
       stride_b, scale_a, layout_a, scale_weight, layout_b},
      {{1.f, 0.f}, nullptr, stride_d, reinterpret_cast<Result *>(output), stride_d}};
  auto check = [](cutlass::Status status) {
    TVM_FFI_ICHECK(status == cutlass::Status::kSuccess) << cutlassGetStatusString(status);
  };
  TVM_FFI_ICHECK(Gemm::get_workspace_size(arguments) == 0);
  Gemm gemm;
  check(gemm.can_implement(arguments));
  check(gemm.initialize(arguments, nullptr, stream));
  check(gemm.run(stream, nullptr, dependent));
}

inline void groupwise(const __nv_fp8_e4m3 *a, const __nv_fp8_e4m3 *weight, const float *scale_a,
                      const float *scale_weight, __nv_bfloat16 *output, int m, int n, int k,
                      cudaStream_t stream) {
  if (m <= 64)
    groupwise<64>(a, weight, scale_a, scale_weight, output, m, n, k, stream);
  else
    groupwise<128>(a, weight, scale_a, scale_weight, output, m, n, k, stream);
}
