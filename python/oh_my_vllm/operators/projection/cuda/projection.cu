#include <cute/tensor.hpp>
#include <cutlass/detail/sm100_blockscaled_layout.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/group_array_problem_shape.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/util/packed_stride.hpp>
#include <vector>

#include "../../runtime/cuda/contracts.cuh"

namespace encoded_projection {

using namespace cute;
using ScaleConfig = cutlass::detail::Sm1xxBlockScaledConfig<32>;

int round128(int value) { return (value + 127) / 128 * 128; }

int scale_bytes(int rows, int width) { return round128(rows) * (round128(width) / 32); }

void dimensions(int64_t rows, int64_t outputs, int64_t width, int64_t bits) {
  TVM_FFI_ICHECK(rows >= 0 && rows <= 32768);
  TVM_FFI_ICHECK(outputs >= 32 && outputs <= 32768 && outputs % 32 == 0);
  TVM_FFI_ICHECK(width >= 32 && width <= 16384 && width % 32 == 0);
  TVM_FFI_ICHECK(bits == 4 || bits == 8);
}

void separate(TensorView left, TensorView right) {
  if (tvm::ffi::GetDataSize(left) && tvm::ffi::GetDataSize(right))
    disjoint(left, right);
}

void check_cutlass(cutlass::Status status) {
  TVM_FFI_ICHECK(status == cutlass::Status::kSuccess) << cutlassGetStatusString(status);
}

using Tile = Shape<_128, _128, _128>;
using Cluster = Shape<_1, _1, _1>;
using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassBlockScaledTensorOp, Tile, Cluster,
    cutlass::epilogue::collective::EpilogueTileAuto, float, float, cutlass::bfloat16_t,
    cutlass::layout::RowMajor, 8, cutlass::bfloat16_t, cutlass::layout::RowMajor, 8,
    cutlass::epilogue::TmaWarpSpecialized1Sm>::CollectiveOp;

template <int Bits> struct Projection {
  using ElementA = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
  using ElementB = conditional_t<Bits == 4, cutlass::mx_float4_t<cutlass::float_e2m1_t>, ElementA>;
  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassBlockScaledTensorOp, ElementA,
      cutlass::layout::RowMajor, 16, ElementB, cutlass::layout::ColumnMajor, Bits == 4 ? 128 : 16,
      float, Tile, Cluster,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename Epilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecialized1SmMxf8f6f4Sm100>::CollectiveOp;
  using Kernel =
      cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Mainloop, Epilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

  static typename Gemm::Arguments
  arguments(int rows, int outputs, int width, const void *input = nullptr,
            const void *weight = nullptr, const void *input_scales = nullptr,
            const void *weight_scales = nullptr, void *output = nullptr) {
    auto shape = make_shape(rows, outputs, width, 1);
    return {
        cutlass::gemm::GemmUniversalMode::kGemm,
        shape,
        {static_cast<const typename ElementA::DataType *>(input),
         cutlass::make_cute_packed_stride(typename Kernel::StrideA{}, make_shape(rows, width, 1)),
         static_cast<const typename ElementB::DataType *>(weight),
         cutlass::make_cute_packed_stride(typename Kernel::StrideB{},
                                          make_shape(outputs, width, 1)),
         static_cast<const typename ElementA::ScaleFactorType *>(input_scales),
         ScaleConfig::tile_atom_to_shape_SFA(shape),
         static_cast<const typename ElementB::ScaleFactorType *>(weight_scales),
         ScaleConfig::tile_atom_to_shape_SFB(shape)},
        {{1.f, 0.f},
         nullptr,
         cutlass::make_cute_packed_stride(typename Kernel::StrideC{}, make_shape(rows, outputs, 1)),
         static_cast<cutlass::bfloat16_t *>(output),
         cutlass::make_cute_packed_stride(typename Kernel::StrideD{},
                                          make_shape(rows, outputs, 1))}};
  }

  static void configure() {
    if constexpr (Kernel::SharedStorageSize >= (48 << 10))
      check_cuda(cudaFuncSetAttribute(cutlass::device_kernel<Kernel>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      Kernel::SharedStorageSize));
  }
};

__global__ void pad_bytes(const uint8_t *input, uint8_t *output, int rows, int width, int padded) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < rows * padded) {
    int row = index / padded, column = index % padded;
    output[index] = column < width ? input[row * width + column] : 0;
  }
}

template <class Layout>
__global__ void pack_scales(const uint8_t *input, uint8_t *output, int rows, int groups, bool dense,
                            Layout layout) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < rows * groups) {
    int row = index / groups, group = index % groups;
    output[layout(make_coord(row, group * 32, 0))] =
        input[(dense ? row / 32 : row) * groups + group];
  }
}

template <class Layout>
void launch_scales(TensorView input, TensorView output, int rows, int width, bool dense,
                   Layout layout, cudaStream_t stream) {
  check_cuda(cudaMemsetAsync(output.data_ptr(), 127, tvm::ffi::GetDataSize(output), stream));
  int count = rows * (width / 32);
  pack_scales<<<(count + 255) / 256, 256, 0, stream>>>(
      static_cast<const uint8_t *>(input.data_ptr()), static_cast<uint8_t *>(output.data_ptr()),
      rows, width / 32, dense, layout);
  check_cuda(cudaGetLastError());
}

void launch_padding(TensorView input, TensorView output, int rows, int width, int padded,
                    cudaStream_t stream) {
  pad_bytes<<<(rows * padded + 255) / 256, 256, 0, stream>>>(
      static_cast<const uint8_t *>(input.data_ptr()), static_cast<uint8_t *>(output.data_ptr()),
      rows, width, padded);
  check_cuda(cudaGetLastError());
}

template <int Bits>
void launch_projection(TensorView encoded, TensorView scales, TensorView weight,
                       TensorView weight_scales, TensorView output, TensorView padded_input,
                       TensorView packed_scales, TensorView scratch, int rows, int outputs,
                       int width, cudaStream_t stream) {
  using Operator = Projection<Bits>;
  using Kernel = typename Operator::Kernel;
  using Gemm = typename Operator::Gemm;
  int padded = Bits == 4 ? round128(width) : width;
  auto input = padded == width ? encoded : padded_input;
  auto args =
      Operator::arguments(rows, outputs, padded, input.data_ptr(), weight.data_ptr(),
                          packed_scales.data_ptr(), weight_scales.data_ptr(), output.data_ptr());
  check_cutlass(Gemm::can_implement(args));
  if (padded != width)
    launch_padding(encoded, padded_input, rows, width, padded, stream);
  auto layout = ScaleConfig::tile_atom_to_shape_SFA(make_shape(rows, outputs, padded, 1));
  launch_scales(scales, packed_scales, rows, width, false, layout, stream);
  check_cutlass(Kernel::initialize_workspace(args, scratch.data_ptr(), stream));
  auto params = Kernel::to_underlying_arguments(args, scratch.data_ptr());
  check_cutlass(Gemm::run(params, stream));
}

} // namespace encoded_projection

using namespace encoded_projection;

namespace grouped_projection {

using namespace cute;
using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;
using ElementA = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
using ElementB = cutlass::mx_float4_t<cutlass::float_e2m1_t>;
using ElementC = cutlass::bfloat16_t;
using ElementD = cutlass::bfloat16_t;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using Tile = Shape<_128, _64, _128>;
using Cluster = Shape<_1, _1, _1>;

using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassBlockScaledTensorOp, Tile, Cluster,
    cutlass::epilogue::collective::EpilogueTileAuto, float, float, ElementC, LayoutC *, 8, ElementD,
    LayoutC *, 8, cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm>::CollectiveOp;
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassBlockScaledTensorOp, ElementA, LayoutA *, 16,
    ElementB, LayoutB *, 128, float, Tile, Cluster,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
        sizeof(typename Epilogue::SharedStorage))>,
    cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmMxf8f6f4Sm100>::CollectiveOp;
using Kernel = cutlass::gemm::kernel::GemmUniversal<ProblemShape, Mainloop, Epilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
using StrideA = typename Gemm::GemmKernel::InternalStrideA;
using StrideB = typename Gemm::GemmKernel::InternalStrideB;
using StrideC = typename Gemm::GemmKernel::InternalStrideC;
using StrideD = typename Gemm::GemmKernel::InternalStrideD;
using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFA;
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFB;
using ScaleConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
using ElementSF = typename Gemm::GemmKernel::ElementSF;
using UnderlyingProblemShape = ProblemShape::UnderlyingProblemShape;
using DataA = typename ElementA::DataType;
using DataB = typename ElementB::DataType;

struct WorkspaceLayout {
  size_t padded_input;
  size_t packed_scales;
  size_t shapes;
  size_t offsets;
  size_t ptr_a;
  size_t ptr_sfa;
  size_t ptr_d;
  size_t strides_a;
  size_t strides_b;
  size_t strides_c;
  size_t strides_d;
  size_t layouts_sfa;
  size_t layouts_sfb;
  size_t cutlass;
  size_t total;
};

size_t align_up(size_t value, size_t alignment = 128) {
  return (value + alignment - 1) / alignment * alignment;
}

WorkspaceLayout layout_workspace(int rows, int groups, int max_rows, int outputs, int width,
                                 size_t cutlass_bytes) {
  auto padded = round128(width);
  auto layout_sfa = ScaleConfig::tile_atom_to_shape_SFA(make_shape(max_rows, outputs, padded, 1));
  auto scale_bytes = size(filter_zeros(layout_sfa)) * sizeof(ElementSF);
  WorkspaceLayout result{};
  size_t offset = 0;
  result.padded_input = offset = align_up(offset);
  if (padded != width)
    offset += size_t(rows) * padded;
  result.packed_scales = (offset = align_up(offset), offset);
  offset += size_t(groups) * scale_bytes;
  result.shapes = (offset = align_up(offset), offset);
  offset += size_t(groups) * sizeof(UnderlyingProblemShape);
  result.offsets = (offset = align_up(offset), offset);
  offset += size_t(groups + 1) * sizeof(int32_t);
  result.ptr_a = (offset = align_up(offset), offset);
  offset += size_t(groups) * sizeof(DataA const *);
  result.ptr_sfa = (offset = align_up(offset), offset);
  offset += size_t(groups) * sizeof(ElementSF const *);
  result.ptr_d = (offset = align_up(offset), offset);
  offset += size_t(groups) * sizeof(ElementD *);
  result.strides_a = (offset = align_up(offset), offset);
  offset += size_t(groups) * sizeof(StrideA);
  result.strides_b = (offset = align_up(offset), offset);
  offset += size_t(groups) * sizeof(StrideB);
  result.strides_c = (offset = align_up(offset), offset);
  offset += size_t(groups) * sizeof(StrideC);
  result.strides_d = (offset = align_up(offset), offset);
  offset += size_t(groups) * sizeof(StrideD);
  result.layouts_sfa = (offset = align_up(offset), offset);
  offset += size_t(groups) * sizeof(LayoutSFA);
  result.layouts_sfb = (offset = align_up(offset), offset);
  offset += size_t(groups) * sizeof(LayoutSFB);
  result.cutlass = (offset = align_up(offset), offset);
  result.total = align_up(offset + cutlass_bytes);
  return result;
}

template <class Layout>
__global__ void pack_group_scales(const uint8_t *input, const int32_t *counts,
                                  const int32_t *offsets, uint8_t *output, int groups, int max_rows,
                                  int width_groups, int scale_bytes, Layout layout) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int per_group = max_rows * width_groups;
  int group = index / per_group;
  if (group >= groups)
    return;
  int local = index - group * per_group;
  int row = local / width_groups;
  int column = local % width_groups;
  if (row >= counts[group])
    return;
  output[group * scale_bytes + layout(make_coord(row, column * 32, 0))] =
      input[(offsets[group] + row) * width_groups + column];
}

template <class DataA_, class Scale_, class DataD_>
__global__ void build_metadata(const int32_t *counts, int groups, int outputs, int width,
                               int padded, const uint8_t *input, const uint8_t *padded_input,
                               uint8_t *packed_scales, uint8_t *output, int scale_bytes,
                               int32_t *offsets, UnderlyingProblemShape *shapes,
                               DataA_ const **ptr_a, Scale_ const **ptr_sfa, DataD_ **ptr_d) {
  if (blockIdx.x != 0 || threadIdx.x != 0)
    return;
  int offset = 0;
  const uint8_t *input_base = padded == width ? input : padded_input;
  for (int group = 0; group < groups; ++group) {
    int count = counts[group];
    offsets[group] = offset;
    shapes[group] = {count, outputs, padded};
    ptr_a[group] = reinterpret_cast<DataA_ const *>(input_base + int64_t(offset) * padded);
    ptr_sfa[group] = reinterpret_cast<Scale_ const *>(packed_scales + int64_t(group) * scale_bytes);
    ptr_d[group] = reinterpret_cast<DataD_ *>(output + int64_t(offset) * outputs * sizeof(DataD_));
    offset += count;
  }
  offsets[groups] = offset;
}

template <class T> T *workspace_pointer(TensorView workspace, size_t offset) {
  return reinterpret_cast<T *>(static_cast<uint8_t *>(workspace.data_ptr()) + offset);
}

typename Gemm::Arguments make_arguments(int groups, int max_rows, int outputs, int padded,
                                        void *shapes, void *ptr_a, void *ptr_sfa, void *ptr_d,
                                        void *strides_a, void *strides_b, void *strides_c,
                                        void *strides_d, void *layouts_sfa, void *layouts_sfb,
                                        void *ptr_b, void *ptr_sfb,
                                        cutlass::KernelHardwareInfo hw_info,
                                        std::vector<ProblemShape::UnderlyingProblemShape> &host) {
  host.assign(groups, {max_rows, outputs, padded});
  typename Gemm::GemmKernel::TileSchedulerArguments scheduler;
  return {
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {groups, reinterpret_cast<UnderlyingProblemShape *>(shapes), host.data()},
      {reinterpret_cast<DataA const **>(ptr_a), reinterpret_cast<StrideA *>(strides_a),
       reinterpret_cast<DataB const **>(ptr_b), reinterpret_cast<StrideB *>(strides_b),
       reinterpret_cast<ElementSF const **>(ptr_sfa), reinterpret_cast<LayoutSFA *>(layouts_sfa),
       reinterpret_cast<ElementSF const **>(ptr_sfb), reinterpret_cast<LayoutSFB *>(layouts_sfb)},
      {{1.f, 0.f},
       reinterpret_cast<ElementC const **>(ptr_d),
       reinterpret_cast<StrideC *>(strides_c),
       reinterpret_cast<ElementD **>(ptr_d),
       reinterpret_cast<StrideD *>(strides_d)},
      hw_info,
      scheduler};
}

} // namespace grouped_projection

using namespace grouped_projection;

void pack_weights(TensorView raw_weight, TensorView raw_scales, TensorView packed_weight,
                  TensorView packed_scales, int64_t bits) {
  TVM_FFI_ICHECK(bits == 4 || bits == 8);
  TVM_FFI_ICHECK(raw_weight.ndim() == 2);
  TVM_FFI_ICHECK(raw_weight.size(1) >= 1 && raw_weight.size(1) <= 16384);
  int64_t outputs = raw_weight.size(0), width = raw_weight.size(1) * (8 / bits);
  dimensions(0, outputs, width, bits);
  int padded = bits == 4 ? round128(width) : width;
  tensor_contract(raw_weight, raw_weight, kDLUInt, 8, {outputs, width * bits / 8});
  tensor_contract(raw_scales, raw_weight, kDLUInt, 8,
                  {bits == 4 ? outputs : outputs / 32, width / 32});
  tensor_contract(packed_weight, raw_weight, kDLUInt, 8, {outputs, padded * bits / 8});
  tensor_contract(packed_scales, raw_weight, kDLUInt, 8, {scale_bytes(outputs, width)});
  for (auto input : {raw_weight, raw_scales, packed_weight})
    separate(packed_scales, input);
  separate(packed_weight, raw_scales);
  bool alias = raw_weight.data_ptr() == packed_weight.data_ptr() &&
               tvm::ffi::GetDataSize(raw_weight) == tvm::ffi::GetDataSize(packed_weight);
  if (alias) {
    TVM_FFI_ICHECK(width == padded);
  } else {
    separate(packed_weight, raw_weight);
  }
  tvm::ffi::CUDADeviceGuard guard(raw_weight.device().device_id);
  auto stream =
      static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, raw_weight.device().device_id));
  cudaStreamCaptureStatus capture;
  check_cuda(cudaStreamIsCapturing(stream, &capture));
  TVM_FFI_ICHECK(capture == cudaStreamCaptureStatusNone) << "prepare weights outside capture";
  if (bits == 4)
    Projection<4>::configure();
  else
    Projection<8>::configure();
  if (!alias)
    launch_padding(raw_weight, packed_weight, outputs, width * bits / 8, padded * bits / 8, stream);
  auto layout = ScaleConfig::tile_atom_to_shape_SFB(make_shape(1, int(outputs), padded, 1));
  launch_scales(raw_scales, packed_scales, outputs, width, bits == 8, layout, stream);
}

int64_t grouped_workspace_size(int64_t rows, int64_t groups, int64_t max_rows, int64_t outputs,
                               int64_t width) {
  TVM_FFI_ICHECK(rows >= 0 && rows <= 1 << 20);
  TVM_FFI_ICHECK(groups > 0 && groups <= 384);
  TVM_FFI_ICHECK(max_rows > 0 && max_rows <= 1 << 20);
  TVM_FFI_ICHECK(outputs >= 32 && outputs <= 32768 && outputs % 32 == 0);
  TVM_FFI_ICHECK(width >= 32 && width <= 16384 && width % 32 == 0);
  int device = 0;
  check_cuda(cudaGetDevice(&device));
  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = device;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device);
  std::vector<ProblemShape::UnderlyingProblemShape> host;
  auto args = make_arguments(groups, max_rows, outputs, round128(width),
                             nullptr, // shapes
                             nullptr, // ptr_a
                             nullptr, // ptr_sfa
                             nullptr, // ptr_d
                             nullptr, // strides_a
                             nullptr, // strides_b
                             nullptr, // strides_c
                             nullptr, // strides_d
                             nullptr, // layouts_sfa
                             nullptr, // layouts_sfb
                             nullptr, // ptr_b
                             nullptr, // ptr_sfb
                             hw_info, host);
  auto cutlass_bytes = Gemm::get_workspace_size(args);
  return static_cast<int64_t>(
      layout_workspace(rows, groups, max_rows, outputs, width, cutlass_bytes).total);
}

void grouped_project(TensorView encoded, TensorView scales, TensorView weight_ptrs,
                     TensorView weight_scale_ptrs, TensorView counts, TensorView output,
                     TensorView scratch, int64_t max_rows) {
  TVM_FFI_ICHECK(encoded.ndim() == 2 && scales.ndim() == 2 && counts.ndim() == 1);
  TVM_FFI_ICHECK(output.ndim() == 2 && scratch.ndim() == 1);
  int64_t rows = encoded.size(0), width = encoded.size(1), groups = counts.size(0);
  int64_t outputs = output.size(1);
  TVM_FFI_ICHECK(scales.size(0) == rows && scales.size(1) == width / 32);
  TVM_FFI_ICHECK(output.size(0) == rows);
  TVM_FFI_ICHECK(weight_ptrs.ndim() == 1 && weight_scale_ptrs.ndim() == 1);
  TVM_FFI_ICHECK(weight_ptrs.size(0) == groups && weight_scale_ptrs.size(0) == groups);
  TVM_FFI_ICHECK(max_rows > 0 && max_rows <= 1 << 20);
  TVM_FFI_ICHECK(rows <= 1 << 20 && groups > 0 && groups <= 384);
  TVM_FFI_ICHECK(width >= 32 && width <= 16384 && width % 32 == 0);
  TVM_FFI_ICHECK(outputs >= 32 && outputs <= 32768 && outputs % 32 == 0);
  tensor_contract(encoded, encoded, kDLUInt, 8, {rows, width});
  tensor_contract(scales, encoded, kDLUInt, 8, {rows, width / 32});
  tensor_contract(weight_ptrs, encoded, kDLInt, 64, {groups});
  tensor_contract(weight_scale_ptrs, encoded, kDLInt, 64, {groups});
  tensor_contract(counts, encoded, kDLInt, 32, {groups});
  tensor_contract(output, encoded, kDLBfloat, 16, {rows, outputs});
  tensor_contract(scratch, encoded, kDLUInt, 8, {scratch.size(0)});

  tvm::ffi::CUDADeviceGuard guard(encoded.device().device_id);
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, encoded.device().device_id));
  int padded = round128(static_cast<int>(width));
  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = encoded.device().device_id;
  hw_info.sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);
  std::vector<ProblemShape::UnderlyingProblemShape> host;
  auto size_args = make_arguments(groups, max_rows, outputs, padded,
                                  nullptr, // shapes
                                  nullptr, // ptr_a
                                  nullptr, // ptr_sfa
                                  nullptr, // ptr_d
                                  nullptr, // strides_a
                                  nullptr, // strides_b
                                  nullptr, // strides_c
                                  nullptr, // strides_d
                                  nullptr, // layouts_sfa
                                  nullptr, // layouts_sfb
                                  nullptr, // ptr_b
                                  nullptr, // ptr_sfb
                                  hw_info, host);
  auto cutlass_bytes = Gemm::get_workspace_size(size_args);
  auto layout = layout_workspace(rows, groups, max_rows, outputs, width, cutlass_bytes);
  TVM_FFI_ICHECK(scratch.size(0) >= static_cast<int64_t>(layout.total));

  auto *shapes = workspace_pointer<UnderlyingProblemShape>(scratch, layout.shapes);
  auto *offsets = workspace_pointer<int32_t>(scratch, layout.offsets);
  auto *ptr_a = workspace_pointer<DataA const *>(scratch, layout.ptr_a);
  auto *ptr_sfa = workspace_pointer<ElementSF const *>(scratch, layout.ptr_sfa);
  auto *ptr_d = workspace_pointer<ElementD *>(scratch, layout.ptr_d);
  auto *packed_scales = workspace_pointer<uint8_t>(scratch, layout.packed_scales);
  auto *padded_input =
      padded != width ? workspace_pointer<uint8_t>(scratch, layout.padded_input) : nullptr;
  int scale_bytes =
      static_cast<int>(size(filter_zeros(ScaleConfig::tile_atom_to_shape_SFA(make_shape(
                           static_cast<int>(max_rows), static_cast<int>(outputs), padded, 1)))) *
                       sizeof(ElementSF));

  build_metadata<<<1, 1, 0, stream>>>(
      static_cast<const int32_t *>(counts.data_ptr()), static_cast<int>(groups),
      static_cast<int>(outputs), static_cast<int>(width), padded,
      static_cast<const uint8_t *>(encoded.data_ptr()), padded_input, packed_scales,
      static_cast<uint8_t *>(output.data_ptr()), scale_bytes, offsets, shapes, ptr_a, ptr_sfa,
      ptr_d);
  check_cuda(cudaGetLastError());
  check_cuda(cudaMemsetAsync(packed_scales, 127, size_t(groups) * scale_bytes, stream));
  auto packed_layout = ScaleConfig::tile_atom_to_shape_SFA(
      make_shape(static_cast<int>(max_rows), static_cast<int>(outputs), padded, 1));
  int scale_count = static_cast<int>(max_rows) * static_cast<int>(width) / 32 * groups;
  pack_group_scales<<<(scale_count + 255) / 256, 256, 0, stream>>>(
      static_cast<const uint8_t *>(scales.data_ptr()),
      static_cast<const int32_t *>(counts.data_ptr()), offsets, packed_scales,
      static_cast<int>(groups), static_cast<int>(max_rows), static_cast<int>(width) / 32,
      scale_bytes, packed_layout);
  check_cuda(cudaGetLastError());
  if (padded != width) {
    pad_bytes<<<(rows * padded + 255) / 256, 256, 0, stream>>>(
        static_cast<const uint8_t *>(encoded.data_ptr()), padded_input, static_cast<int>(rows),
        static_cast<int>(width), padded);
    check_cuda(cudaGetLastError());
  }

  auto *strides_a = workspace_pointer<StrideA>(scratch, layout.strides_a);
  auto *strides_b = workspace_pointer<StrideB>(scratch, layout.strides_b);
  auto *strides_c = workspace_pointer<StrideC>(scratch, layout.strides_c);
  auto *strides_d = workspace_pointer<StrideD>(scratch, layout.strides_d);
  auto *layouts_sfa = workspace_pointer<LayoutSFA>(scratch, layout.layouts_sfa);
  auto *layouts_sfb = workspace_pointer<LayoutSFB>(scratch, layout.layouts_sfb);
  auto stride_a = cutlass::make_cute_packed_stride(
      StrideA{}, make_shape(static_cast<int>(max_rows), padded, 1));
  auto stride_b =
      cutlass::make_cute_packed_stride(StrideB{}, make_shape(static_cast<int>(outputs), padded, 1));
  auto stride_c = cutlass::make_cute_packed_stride(
      StrideC{}, make_shape(static_cast<int>(max_rows), static_cast<int>(outputs), 1));
  auto stride_d = cutlass::make_cute_packed_stride(
      StrideD{}, make_shape(static_cast<int>(max_rows), static_cast<int>(outputs), 1));
  auto layout_sfa = ScaleConfig::tile_atom_to_shape_SFA(
      make_shape(static_cast<int>(max_rows), static_cast<int>(outputs), padded, 1));
  auto layout_sfb = ScaleConfig::tile_atom_to_shape_SFB(
      make_shape(static_cast<int>(max_rows), static_cast<int>(outputs), padded, 1));
  std::vector<StrideA> host_strides_a(groups, stride_a);
  std::vector<StrideB> host_strides_b(groups, stride_b);
  std::vector<StrideC> host_strides_c(groups, stride_c);
  std::vector<StrideD> host_strides_d(groups, stride_d);
  std::vector<LayoutSFA> host_layouts_sfa(groups, layout_sfa);
  std::vector<LayoutSFB> host_layouts_sfb(groups, layout_sfb);
  check_cuda(cudaMemcpyAsync(strides_a, host_strides_a.data(),
                             host_strides_a.size() * sizeof(StrideA), cudaMemcpyHostToDevice,
                             stream));
  check_cuda(cudaMemcpyAsync(strides_b, host_strides_b.data(),
                             host_strides_b.size() * sizeof(StrideB), cudaMemcpyHostToDevice,
                             stream));
  check_cuda(cudaMemcpyAsync(strides_c, host_strides_c.data(),
                             host_strides_c.size() * sizeof(StrideC), cudaMemcpyHostToDevice,
                             stream));
  check_cuda(cudaMemcpyAsync(strides_d, host_strides_d.data(),
                             host_strides_d.size() * sizeof(StrideD), cudaMemcpyHostToDevice,
                             stream));
  check_cuda(cudaMemcpyAsync(layouts_sfa, host_layouts_sfa.data(),
                             host_layouts_sfa.size() * sizeof(LayoutSFA), cudaMemcpyHostToDevice,
                             stream));
  check_cuda(cudaMemcpyAsync(layouts_sfb, host_layouts_sfb.data(),
                             host_layouts_sfb.size() * sizeof(LayoutSFB), cudaMemcpyHostToDevice,
                             stream));

  auto args = make_arguments(groups, max_rows, outputs, padded, shapes, ptr_a, ptr_sfa, ptr_d,
                             strides_a, strides_b, strides_c, strides_d, layouts_sfa, layouts_sfb,
                             weight_ptrs.data_ptr(), weight_scale_ptrs.data_ptr(), hw_info, host);
  check_cutlass(Gemm::can_implement(args));
  if constexpr (Kernel::SharedStorageSize >= (48 << 10))
    check_cuda(cudaFuncSetAttribute(cutlass::device_kernel<Kernel>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    Kernel::SharedStorageSize));
  check_cutlass(Kernel::initialize_workspace(
      args, workspace_pointer<uint8_t>(scratch, layout.cutlass), stream));
  auto params =
      Kernel::to_underlying_arguments(args, workspace_pointer<uint8_t>(scratch, layout.cutlass));
  check_cutlass(Gemm::run(params, stream));
}

int64_t workspace_size(int64_t rows, int64_t outputs, int64_t width, int64_t bits) {
  dimensions(rows, outputs, width, bits);
  if (!rows)
    return 0;
  if (bits == 4)
    return Projection<4>::Gemm::get_workspace_size(
        Projection<4>::arguments(rows, outputs, round128(width)));
  return Projection<8>::Gemm::get_workspace_size(Projection<8>::arguments(rows, outputs, width));
}

void project(TensorView encoded, TensorView scales, TensorView packed_weight,
             TensorView weight_scales, TensorView output, TensorView padded_input,
             TensorView packed_input_scales, TensorView scratch, int64_t width, int64_t bits) {
  TVM_FFI_ICHECK(encoded.ndim() == 2 && output.ndim() == 2);
  int64_t rows = encoded.size(0), outputs = output.size(1);
  dimensions(rows, outputs, width, bits);
  int padded = bits == 4 ? round128(width) : width;
  tensor_contract(encoded, encoded, kDLUInt, 8, {rows, width});
  tensor_contract(scales, encoded, kDLUInt, 8, {rows, width / 32});
  tensor_contract(packed_weight, encoded, kDLUInt, 8, {outputs, padded * bits / 8});
  tensor_contract(weight_scales, encoded, kDLUInt, 8, {scale_bytes(outputs, width)});
  tensor_contract(output, encoded, kDLBfloat, 16, {rows, outputs});
  if (padded != width)
    tensor_contract(padded_input, encoded, kDLUInt, 8, {rows, padded});
  tensor_contract(packed_input_scales, encoded, kDLUInt, 8, {scale_bytes(rows, width)});
  tensor_contract(scratch, encoded, kDLUInt, 8, {workspace_size(rows, outputs, width, bits)});
  for (auto writable : {output, padded_input, packed_input_scales, scratch})
    for (auto input : {encoded, scales, packed_weight, weight_scales})
      separate(writable, input);
  separate(output, padded_input);
  separate(output, packed_input_scales);
  separate(output, scratch);
  separate(padded_input, packed_input_scales);
  separate(padded_input, scratch);
  separate(packed_input_scales, scratch);
  if (!rows)
    return;
  tvm::ffi::CUDADeviceGuard guard(encoded.device().device_id);
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, encoded.device().device_id));
  if (bits == 4)
    launch_projection<4>(encoded, scales, packed_weight, weight_scales, output, padded_input,
                         packed_input_scales, scratch, rows, outputs, width, stream);
  else
    launch_projection<8>(encoded, scales, packed_weight, weight_scales, output, padded_input,
                         packed_input_scales, scratch, rows, outputs, width, stream);
}
