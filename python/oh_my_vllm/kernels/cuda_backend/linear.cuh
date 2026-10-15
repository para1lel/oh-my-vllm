#pragma once

#include "common.cuh"
#include "groupwise.cuh"

template <int TileM, int TileN, bool Transposed, typename Result>
inline void bf16_gemm(const __nv_bfloat16 *a, const __nv_bfloat16 *weight, void *output, int m,
                      int n, int k, cudaStream_t stream) {
  using namespace cute;
  using Element = cutlass::bfloat16_t;
  using Row = cutlass::layout::RowMajor;
  using Column = cutlass::layout::ColumnMajor;
  using OutputLayout = std::conditional_t<Transposed, Column, Row>;
  using Tile = Shape<Int<TileM>, Int<TileN>, _64>;
  using Cluster = Shape<_1, _1, _1>;
  constexpr int Alignment = 128 / cutlass::sizeof_bits<Result>::value;
  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, Tile, Cluster,
      cutlass::epilogue::collective::EpilogueTileAuto, float, float, Result, OutputLayout,
      Alignment, Result, OutputLayout, Alignment,
      cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, Element, Row, 8, Element, Column, 8,
      float, Tile, Cluster,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename Epilogue::SharedStorage))>,
      cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
  using Kernel =
      cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Mainloop, Epilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
  auto sa = cutlass::make_cute_packed_stride(typename Kernel::StrideA{}, make_shape(m, k, 1));
  auto sb = cutlass::make_cute_packed_stride(typename Kernel::StrideB{}, make_shape(n, k, 1));
  auto sd = cutlass::make_cute_packed_stride(typename Kernel::StrideD{}, make_shape(m, n, 1));
  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      {reinterpret_cast<const Element *>(a), sa, reinterpret_cast<const Element *>(weight), sb},
      {{1.f, 0.f},
       reinterpret_cast<const Result *>(output),
       sd,
       reinterpret_cast<Result *>(output),
       sd}};
  auto check = [](cutlass::Status status) {
    TVM_FFI_ICHECK(status == cutlass::Status::kSuccess) << cutlassGetStatusString(status);
  };
  TVM_FFI_ICHECK(Gemm::get_workspace_size(args) == 0);
  Gemm gemm;
  check(gemm.can_implement(args));
  check(gemm.initialize(args, nullptr, stream));
  check(gemm.run(stream));
}

template <typename Result = cutlass::bfloat16_t>
inline void bf16_linear(const __nv_bfloat16 *a, const __nv_bfloat16 *weight, void *output, int m,
                        int n, int k, cudaStream_t stream) {
  if (m <= 32)
    bf16_gemm<128, 32, true, Result>(weight, a, output, n, m, k, stream);
  else
    bf16_gemm<128, 128, false, Result>(a, weight, output, m, n, k, stream);
}
