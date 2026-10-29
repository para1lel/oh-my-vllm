#include <cute/tensor.hpp>
#include <cutlass/detail/sm100_blockscaled_layout.hpp>

#include "../../runtime/cuda/contracts.cuh"
#include "fp4_k32.cuh"

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

template <class Output>
void launch_fp4_projection(TensorView encoded, TensorView scales, TensorView weight,
                           TensorView weight_scales, TensorView output, int rows, int outputs,
                           int width, cudaStream_t stream) {
  auto layout = ScaleConfig::tile_atom_to_shape_SFB(make_shape(1, outputs, round128(width), 1));
  launch_fp4_k32(static_cast<const uint8_t *>(encoded.data_ptr()),
                 static_cast<const uint8_t *>(weight.data_ptr()),
                 static_cast<const uint8_t *>(scales.data_ptr()),
                 static_cast<const uint8_t *>(weight_scales.data_ptr()),
                 static_cast<Output *>(output.data_ptr()), rows, outputs, width,
                 round128(width) / 2, layout, stream);
  check_cuda(cudaGetLastError());
}

template <class Output>
void launch_grouped_fp4_projection(TensorView encoded, TensorView scales, TensorView weight_ptrs,
                                   TensorView weight_scale_ptrs, TensorView counts,
                                   TensorView output, TensorView scratch, int groups, int rows,
                                   int outputs, int width, int max_rows, cudaStream_t stream) {
  int padded = round128(width);
  auto layout = ScaleConfig::tile_atom_to_shape_SFB(make_shape(1, outputs, padded, 1));
  launch_grouped_fp4_k32(static_cast<const uint8_t *>(encoded.data_ptr()),
                         static_cast<const uint8_t *>(scales.data_ptr()),
                         static_cast<const int64_t *>(weight_ptrs.data_ptr()),
                         static_cast<const int64_t *>(weight_scale_ptrs.data_ptr()),
                         static_cast<const int32_t *>(counts.data_ptr()),
                         static_cast<int32_t *>(scratch.data_ptr()),
                         static_cast<Output *>(output.data_ptr()), groups, rows, outputs, width,
                         max_rows, padded / 2, layout, stream);
  check_cuda(cudaGetLastError());
}

template <class Output>
void launch_fp8_projection(TensorView encoded, TensorView scales, TensorView weight,
                           TensorView weight_scales, TensorView output, int rows, int outputs,
                           int width, cudaStream_t stream) {
  auto layout = ScaleConfig::tile_atom_to_shape_SFB(make_shape(1, outputs, width, 1));
  launch_fp8_k32(static_cast<const uint8_t *>(encoded.data_ptr()),
                 static_cast<const uint8_t *>(weight.data_ptr()),
                 static_cast<const uint8_t *>(scales.data_ptr()),
                 static_cast<const uint8_t *>(weight_scales.data_ptr()),
                 static_cast<Output *>(output.data_ptr()), rows, outputs, width, layout, stream);
  check_cuda(cudaGetLastError());
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
  TVM_FFI_ICHECK(rows <= groups * max_rows);
  TVM_FFI_ICHECK(outputs >= 32 && outputs <= 32768 && outputs % 32 == 0);
  TVM_FFI_ICHECK(width >= 32 && width <= 16384 && width % 32 == 0);
  return ((groups + 1) * sizeof(int32_t) + 15) / 16 * 16;
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
  bool fp32 = output.dtype().code == kDLFloat && output.dtype().bits == 32;
  tensor_contract(output, encoded, fp32 ? kDLFloat : kDLBfloat, fp32 ? 32 : 16, {rows, outputs});
  tensor_contract(scratch, encoded, kDLUInt, 8, {scratch.size(0)});

  auto bytes = grouped_workspace_size(rows, groups, max_rows, outputs, width);
  TVM_FFI_ICHECK(scratch.size(0) >= bytes);
  for (auto writable : {output, scratch})
    for (auto input : {encoded, scales, weight_ptrs, weight_scale_ptrs, counts})
      separate(writable, input);
  separate(output, scratch);
  if (!rows)
    return;
  tvm::ffi::CUDADeviceGuard guard(encoded.device().device_id);
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, encoded.device().device_id));
  if (fp32)
    launch_grouped_fp4_projection<float>(encoded, scales, weight_ptrs, weight_scale_ptrs, counts,
                                         output, scratch, groups, rows, outputs, width, max_rows,
                                         stream);
  else
    launch_grouped_fp4_projection<__nv_bfloat16>(encoded, scales, weight_ptrs, weight_scale_ptrs,
                                                 counts, output, scratch, groups, rows, outputs,
                                                 width, max_rows, stream);
}

int64_t workspace_size(int64_t rows, int64_t outputs, int64_t width, int64_t bits) {
  dimensions(rows, outputs, width, bits);
  return 0;
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
  bool fp32 = output.dtype().code == kDLFloat && output.dtype().bits == 32;
  tensor_contract(output, encoded, fp32 ? kDLFloat : kDLBfloat, fp32 ? 32 : 16, {rows, outputs});
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
  if (bits == 4) {
    if (fp32)
      launch_fp4_projection<float>(encoded, scales, packed_weight, weight_scales, output, rows,
                                   outputs, width, stream);
    else
      launch_fp4_projection<__nv_bfloat16>(encoded, scales, packed_weight, weight_scales, output,
                                           rows, outputs, width, stream);
  } else if (fp32) {
    launch_fp8_projection<float>(encoded, scales, packed_weight, weight_scales, output, rows,
                                 outputs, width, stream);
  } else {
    launch_fp8_projection<__nv_bfloat16>(encoded, scales, packed_weight, weight_scales, output,
                                         rows, outputs, width, stream);
  }
}
