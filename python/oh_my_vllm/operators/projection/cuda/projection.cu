#include <cute/tensor.hpp>
#include <cutlass/detail/sm100_blockscaled_layout.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/util/packed_stride.hpp>

#include "../../runtime/cuda/contracts.cuh"

namespace encoded_projection {

using namespace cute;
using ScaleConfig = cutlass::detail::Sm1xxBlockScaledConfig<32>;

int round128(int value) { return (value + 127) / 128 * 128; }

int scale_bytes(int rows, int width) { return round128(rows) * (round128(width) / 32); }

void dimensions(int64_t rows, int64_t outputs, int64_t width, int64_t bits) {
  TVM_FFI_ICHECK(rows >= 0 && rows <= 384);
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
