#pragma once

#include "../../primitives/cuda/linear.cuh"
#include <cutlass/epilogue/fusion/operations.hpp>

template <class Value> struct VisionRoundedGelu;

template <> struct VisionRoundedGelu<float> {
  static constexpr bool kIsHeavy = true;

  CUTLASS_HOST_DEVICE float operator()(float value) const {
    cutlass::NumericConverter<cutlass::bfloat16_t, float> convert;
    value = float(convert(value));
    return value * .5f *
           (1.f + tanhf(.7978845608028654f * (value + .044715f * value * value * value)));
  }
};

template <int Count> struct VisionRoundedGelu<cutlass::Array<float, Count>> {
  static constexpr bool kIsHeavy = true;

  CUTLASS_HOST_DEVICE cutlass::Array<float, Count>
  operator()(const cutlass::Array<float, Count> &input) const {
    cutlass::Array<float, Count> output;
    CUTLASS_PRAGMA_UNROLL
    for (int index = 0; index < Count; ++index)
      output[index] = VisionRoundedGelu<float>{}(input[index]);
    return output;
  }
};

#include "vision_lookup.cuh"

template <bool Lookup>
static void vision_mlp_up_impl(const __nv_bfloat16 *input, const __nv_bfloat16 *weight,
                               const __nv_bfloat16 *bias, __nv_bfloat16 *output, int rows,
                               const cutlass::bfloat16_t *table, cudaStream_t stream) {
  using namespace cute;
  using Element = cutlass::bfloat16_t;
  using Row = cutlass::layout::RowMajor;
  using Column = cutlass::layout::ColumnMajor;
  using Tile = Shape<_128, _128, _64>;
  using Cluster = Shape<_1, _1, _1>;
  using Fusion = std::conditional_t<
      Lookup,
      cutlass::epilogue::fusion::LinCombPerColBiasEltAct<VisionLookupGelu, Element, float, Element>,
      cutlass::epilogue::fusion::LinCombPerColBiasEltAct<VisionRoundedGelu, Element, float,
                                                         Element>>;
  using EpilogueTile = conditional_t<Lookup, Shape<_128, _64>, Shape<_128, _32>>;
  using EpilogueSchedule = conditional_t<Lookup, cutlass::epilogue::TmaWarpSpecialized1Sm,
                                         cutlass::epilogue::NoSmemWarpSpecialized1Sm>;
  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, Tile, Cluster, EpilogueTile, float,
      float, Element, Row, 8, Element, Row, 8, EpilogueSchedule, Fusion>::CollectiveOp;
  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, Element, Row, 8, Element, Column, 8,
      float, Tile, Cluster,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename Epilogue::SharedStorage))>,
      cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
  using Kernel =
      cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Mainloop, Epilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
  auto sa = cutlass::make_cute_packed_stride(typename Kernel::StrideA{}, make_shape(rows, 1152, 1));
  auto sb = cutlass::make_cute_packed_stride(typename Kernel::StrideB{}, make_shape(4304, 1152, 1));
  auto sd = cutlass::make_cute_packed_stride(typename Kernel::StrideD{}, make_shape(rows, 4304, 1));
  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {rows, 4304, 1152, 1},
      {reinterpret_cast<const Element *>(input), sa, reinterpret_cast<const Element *>(weight), sb},
      {{}, reinterpret_cast<const Element *>(output), sd, reinterpret_cast<Element *>(output), sd}};
  arguments.epilogue.thread.alpha = 1.f;
  arguments.epilogue.thread.beta = 0.f;
  if constexpr (Lookup)
    arguments.epilogue.thread.activation.table = table;
  arguments.epilogue.thread.bias_ptr = reinterpret_cast<const Element *>(bias);
  auto check = [](cutlass::Status status) {
    TVM_FFI_ICHECK(status == cutlass::Status::kSuccess) << cutlassGetStatusString(status);
  };
  TVM_FFI_ICHECK(Gemm::get_workspace_size(arguments) == 0);
  Gemm operation;
  check(operation.can_implement(arguments));
  check(operation.initialize(arguments, nullptr, stream));
  check(operation.run(stream));
}

static void vision_mlp_up(const __nv_bfloat16 *input, const __nv_bfloat16 *weight,
                          const __nv_bfloat16 *bias, __nv_bfloat16 *output, int rows,
                          const cutlass::bfloat16_t *table, cudaStream_t stream) {
  if (rows > 32)
    vision_mlp_up_impl<true>(input, weight, bias, output, rows, table, stream);
  else
    vision_mlp_up_impl<false>(input, weight, bias, output, rows, nullptr, stream);
}
