#include "linear.cuh"
#include "vision_mlp.cuh"
#include "vision_attention.cuh"
#include "vision_patch.cuh"
#include <algorithm>
#include "vision_classic.cuh"

__global__ void vision_norm_kernel(const __nv_bfloat16 *value, const __nv_bfloat16 *weight,
                                   const __nv_bfloat16 *bias, __nv_bfloat16 *output) {
  int row = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x / 32;
  __shared__ float partial[8];
  float sum = 0;
  for (int column = threadIdx.x; column < 1152; column += 256)
    sum += __bfloat162float(value[int64_t(row) * 1152 + column]);
  sum = warp_sum(sum);
  if (lane == 0)
    partial[warp] = sum;
  __syncthreads();
  float mean = warp_sum(lane < 8 ? partial[lane] : 0) / 1152.f;
  float variance = 0;
  for (int column = threadIdx.x; column < 1152; column += 256) {
    float centered = __bfloat162float(value[int64_t(row) * 1152 + column]) - mean;
    variance += centered * centered;
  }
  variance = warp_sum(variance);
  __syncthreads();
  if (lane == 0)
    partial[warp] = variance;
  __syncthreads();
  float inverse = rsqrtf(warp_sum(lane < 8 ? partial[lane] : 0) / 1152.f + 1e-6f);
  for (int column = threadIdx.x; column < 1152; column += 256) {
    int64_t index = int64_t(row) * 1152 + column;
    float normalized = (__bfloat162float(value[index]) - mean) * inverse;
    output[index] = __float2bfloat16_rn(normalized * __bfloat162float(weight[column]) +
                                        __bfloat162float(bias[column]));
  }
}

__device__ inline float vision_biased(const float *projected, const __nv_bfloat16 *bias,
                                      int64_t index, int column) {
  return __bfloat162float(__float2bfloat16_rn(projected[index] + __bfloat162float(bias[column])));
}

#include "vision_residual_norm.cuh"

__global__ void vision_qkv_kernel(const float *projected, const __nv_bfloat16 *bias,
                                  __nv_bfloat16 *query, __nv_bfloat16 *key, __nv_bfloat16 *value,
                                  int rows, int grid_height, int grid_width) {
  __shared__ float cosines[72], sines[72];
  int row = blockIdx.x;
  int tokens = grid_height * grid_width, group = row / tokens, local = row % tokens;
  if (threadIdx.x < 72) {
    int dimension = threadIdx.x;
    int block = local / 4;
    int py = (block / (grid_width / 2)) * 2 + (local % 4) / 2;
    int px = (block % (grid_width / 2)) * 2 + local % 2;
    int frequency = dimension % 36;
    float angle = (frequency < 18 ? py : px) * powf(10000.f, -float(frequency % 18) / 18.f);
    cosines[dimension] = __bfloat162float(__float2bfloat16_rn(cosf(angle)));
    sines[dimension] =
        __bfloat162float(__float2bfloat16_rn(sinf(angle))) * (dimension < 36 ? -1.f : 1.f);
  }
  __syncthreads();
  for (int index = threadIdx.x; index < 16 * 128; index += 256) {
    int dimension = index % 128, head = index / 128;
    int64_t output = ((int64_t(group) * 16 + head) * tokens + local) * 128 + dimension;
    if (dimension >= 72) {
      query[output] = key[output] = value[output] = __float2bfloat16_rn(0.f);
      continue;
    }
    int column = head * 72 + dimension, paired = head * 72 + (dimension + 36) % 72;
    float cosine = cosines[dimension], sine = sines[dimension];
    float q = vision_biased(projected, bias, int64_t(row) * 3456 + column, column);
    float qp = vision_biased(projected, bias, int64_t(row) * 3456 + paired, paired);
    float k = vision_biased(projected, bias, int64_t(row) * 3456 + 1152 + column, 1152 + column);
    float kp = vision_biased(projected, bias, int64_t(row) * 3456 + 1152 + paired, 1152 + paired);
    query[output] = __float2bfloat16_rn(q * cosine + qp * sine);
    key[output] = __float2bfloat16_rn(k * cosine + kp * sine);
    value[output] = __float2bfloat16_rn(
        vision_biased(projected, bias, int64_t(row) * 3456 + 2304 + column, 2304 + column));
  }
}

__global__ void vision_residual_kernel(const float *projected, const __nv_bfloat16 *bias,
                                       const __nv_bfloat16 *residual, __nv_bfloat16 *output,
                                       int64_t elements) {
  int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements)
    output[index] = __float2bfloat16_rn(vision_biased(projected, bias, index, index % 1152) +
                                        __bfloat162float(residual[index]));
}

__global__ void vision_activation_kernel(const float *projected, const __nv_bfloat16 *bias,
                                         __nv_bfloat16 *output, int64_t elements, int columns,
                                         bool approximate) {
  int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    float value = vision_biased(projected, bias, index, index % columns);
    float activated =
        approximate
            ? value * .5f *
                  (1.f + tanhf(.7978845608028654f * (value + .044715f * value * value * value)))
            : value * .5f * (1.f + erff(value * .7071067811865475f));
    output[index] = __float2bfloat16_rn(activated);
  }
}

__global__ void vision_bias_kernel(const float *projected, const __nv_bfloat16 *bias,
                                   __nv_bfloat16 *output, int64_t elements, int columns) {
  int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements)
    output[index] = __float2bfloat16_rn(projected[index] + __bfloat162float(bias[index % columns]));
}

inline const __nv_bfloat16 *vision_read(TensorView tensor) {
  return static_cast<const __nv_bfloat16 *>(tensor.data_ptr());
}

inline __nv_bfloat16 *vision_write(TensorView tensor) {
  return static_cast<__nv_bfloat16 *>(tensor.data_ptr());
}

inline void vision_disjoint(std::initializer_list<TensorView> reads,
                            std::initializer_list<TensorView> writes) {
  for (auto output = writes.begin(); output != writes.end(); ++output) {
    for (auto input : reads)
      disjoint(*output, input);
    for (auto other = writes.begin(); other != output; ++other)
      disjoint(*output, *other);
  }
}

inline void vision_grid(int64_t height, int64_t width) {
  TVM_FFI_ICHECK(height > 0 && width > 0 && height % 2 == 0 && width % 2 == 0);
  TVM_FFI_ICHECK(height <= 65536 && width <= 65536 && height * width >= 4 &&
                 height * width <= 65536);
}

inline int vision_rows(TensorView value) {
  TVM_FFI_ICHECK(value.ndim() == 2 && value.size(0) > 0 && value.size(0) <= 65536);
  tensor_contract(value, value, kDLBfloat, 16, {value.size(0), 1152});
  return value.size(0);
}

inline cudaStream_t vision_stream(TensorView value) {
  return static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, value.device().device_id));
}

void vision_patch(TensorView images, TensorView weight, TensorView bias, TensorView position,
                  TensorView patches, TensorView projected, TensorView output, int64_t grid_height,
                  int64_t grid_width) {
  vision_grid(grid_height, grid_width);
  TVM_FFI_ICHECK(images.ndim() == 5 && images.size(0) > 0 && images.size(1) > 0 &&
                 images.size(2) > 0 && images.size(3) > 0 && images.size(4) == 3);
  TVM_FFI_ICHECK(images.size(0) <= 64 && images.size(1) <= 1024 && images.size(2) <= 65536 &&
                 images.size(3) <= 65536);
  TVM_FFI_ICHECK(double(std::max(images.size(2), images.size(3))) /
                     std::min(images.size(2), images.size(3)) <=
                 200.);
  int64_t rows = images.size(0) * ((images.size(1) + 1) / 2) * grid_height * grid_width;
  TVM_FFI_ICHECK(rows > 0 && rows <= 65536);
  tensor_contract(images, images, kDLUInt, 8,
                  {images.size(0), images.size(1), images.size(2), images.size(3), 3});
  tensor_contract(weight, images, kDLBfloat, 16, {1152, 1536});
  tensor_contract(bias, images, kDLBfloat, 16, {1152});
  tensor_contract(position, images, kDLBfloat, 16, {2304, 1152});
  tensor_contract(patches, images, kDLBfloat, 16, {rows, 1536});
  tensor_contract(projected, images, kDLFloat, 32, {rows * 4304});
  tensor_contract(output, images, kDLBfloat, 16, {rows, 1152});
  vision_disjoint({images, weight, bias, position}, {patches, projected, output});
  tvm::ffi::CUDADeviceGuard guard(images.device().device_id);
  auto stream = vision_stream(images);
  vision_patch_kernel<<<(rows * 1536 + 255) / 256, 256, 0, stream>>>(
      static_cast<const uint8_t *>(images.data_ptr()), vision_write(patches), rows, images.size(1),
      images.size(2), images.size(3), grid_height, grid_width);
  bf16_linear<float>(vision_read(patches), vision_read(weight), projected.data_ptr(), rows, 1152,
                     1536, stream);
  vision_position_kernel<<<(rows * 1152 + 255) / 256, 256, 0, stream>>>(
      static_cast<const float *>(projected.data_ptr()), vision_read(bias), vision_read(position),
      vision_write(output), rows, grid_height, grid_width);
  check_cuda(cudaGetLastError());
}

void vision_block(TensorView value, TensorView norm1_weight, TensorView norm1_bias,
                  TensorView qkv_weight, TensorView qkv_bias, TensorView attention_weight,
                  TensorView attention_bias, TensorView norm2_weight, TensorView norm2_bias,
                  TensorView up_weight, TensorView up_bias, TensorView down_weight,
                  TensorView down_bias, TensorView output, TensorView normalized,
                  TensorView projected, TensorView query, TensorView key, TensorView values,
                  TensorView padded_attention, TensorView normalizer, TensorView attended,
                  TensorView activated, int64_t grid_height, int64_t grid_width,
                  tvm::ffi::Optional<VisionClassicOwner> classic) {
  vision_grid(grid_height, grid_width);
  int rows = vision_rows(value), tokens = grid_height * grid_width;
  TVM_FFI_ICHECK(rows % tokens == 0 && rows / tokens <= 4095);
  int heads = rows / tokens * 16;
  for (auto tensor :
       {norm1_weight, norm1_bias, attention_bias, norm2_weight, norm2_bias, down_bias})
    tensor_contract(tensor, value, kDLBfloat, 16, {1152});
  tensor_contract(qkv_weight, value, kDLBfloat, 16, {3456, 1152});
  tensor_contract(qkv_bias, value, kDLBfloat, 16, {3456});
  tensor_contract(attention_weight, value, kDLBfloat, 16, {1152, 1152});
  tensor_contract(up_weight, value, kDLBfloat, 16, {4304, 1152});
  tensor_contract(up_bias, value, kDLBfloat, 16, {4304});
  tensor_contract(down_weight, value, kDLBfloat, 16, {1152, 4304});
  for (auto tensor : {output, normalized, attended})
    tensor_contract(tensor, value, kDLBfloat, 16, {rows, 1152});
  tensor_contract(projected, value, kDLFloat, 32, {int64_t(rows) * 4304});
  tensor_contract(activated, value, kDLBfloat, 16, {int64_t(rows) * 4304});
  for (auto tensor : {query, key, values})
    tensor_contract(tensor, value, kDLBfloat, 16, {heads, tokens, 128});
  tensor_contract(padded_attention, value, kDLBfloat, 16, {heads, tokens, 128});
  tensor_contract(normalizer, value, kDLFloat, 32, {heads, tokens});
  vision_disjoint({value, norm1_weight, norm1_bias, qkv_weight, qkv_bias, attention_weight,
                   attention_bias, norm2_weight, norm2_bias, up_weight, up_bias, down_weight,
                   down_bias},
                  {output, normalized, projected, query, key, values, padded_attention, normalizer,
                   attended, activated});
  tvm::ffi::CUDADeviceGuard guard(value.device().device_id);
  if (rows > 32) {
    TVM_FFI_ICHECK(classic.has_value());
    vision_classic_context(classic.value(), value.device().device_id);
  }
  auto stream = vision_stream(value);
  auto norm = vision_write(normalized);
  auto matrix = static_cast<float *>(projected.data_ptr());
  vision_norm_kernel<<<rows, 256, 0, stream>>>(vision_read(value), vision_read(norm1_weight),
                                               vision_read(norm1_bias), norm);
  bf16_linear<float>(norm, vision_read(qkv_weight), matrix, rows, 3456, 1152, stream);
  vision_qkv_kernel<<<rows, 256, 0, stream>>>(matrix, vision_read(qkv_bias), vision_write(query),
                                              vision_write(key), vision_write(values), rows,
                                              grid_height, grid_width);
  vision_attention_launch<true, 80>(vision_read(query), vision_read(key), vision_read(values),
                                    vision_write(padded_attention),
                                    static_cast<float *>(normalizer.data_ptr()), tokens,
                                    rows / tokens, value.device().device_id, stream);
  vision_attention_unpack_kernel<<<(int64_t(rows) * 1152 + 255) / 256, 256, 0, stream>>>(
      vision_read(padded_attention), vision_write(attended), rows, tokens);
  bf16_linear<float>(vision_read(attended), vision_read(attention_weight), matrix, rows, 1152, 1152,
                     stream);
  vision_residual_norm_kernel<<<rows, 256, 0, stream>>>(
      matrix, vision_read(attention_bias), vision_read(value), vision_read(norm2_weight),
      vision_read(norm2_bias), vision_write(output), norm);
  vision_mlp_up(norm, vision_read(up_weight), vision_read(up_bias), vision_write(activated), rows,
                stream);
  if (rows > 32) {
    TVM_FFI_ICHECK(classic.has_value());
    vision_classic_down(classic.value(), vision_read(activated), vision_read(down_weight), matrix,
                        rows, value.device().device_id, stream);
  } else
    bf16_linear<float>(vision_read(activated), vision_read(down_weight), matrix, rows, 1152, 4304,
                       stream);
  vision_residual_kernel<<<(int64_t(rows) * 1152 + 255) / 256, 256, 0, stream>>>(
      matrix, vision_read(down_bias), vision_read(output), vision_write(output),
      int64_t(rows) * 1152);
  check_cuda(cudaGetLastError());
}

void vision_merge(TensorView value, TensorView norm_weight, TensorView norm_bias,
                  TensorView up_weight, TensorView up_bias, TensorView down_weight,
                  TensorView down_bias, TensorView output, TensorView normalized,
                  TensorView projected, TensorView activated) {
  int rows = vision_rows(value);
  TVM_FFI_ICHECK(rows % 4 == 0);
  tensor_contract(norm_weight, value, kDLBfloat, 16, {1152});
  tensor_contract(norm_bias, value, kDLBfloat, 16, {1152});
  tensor_contract(up_weight, value, kDLBfloat, 16, {4608, 4608});
  tensor_contract(up_bias, value, kDLBfloat, 16, {4608});
  tensor_contract(down_weight, value, kDLBfloat, 16, {5120, 4608});
  tensor_contract(down_bias, value, kDLBfloat, 16, {5120});
  tensor_contract(output, value, kDLBfloat, 16, {rows / 4, 5120});
  tensor_contract(normalized, value, kDLBfloat, 16, {rows, 1152});
  tensor_contract(projected, value, kDLFloat, 32, {int64_t(rows) * 4304});
  tensor_contract(activated, value, kDLBfloat, 16, {int64_t(rows) * 4304});
  vision_disjoint({value, norm_weight, norm_bias, up_weight, up_bias, down_weight, down_bias},
                  {output, normalized, projected, activated});
  tvm::ffi::CUDADeviceGuard guard(value.device().device_id);
  auto stream = vision_stream(value);
  auto matrix = static_cast<float *>(projected.data_ptr());
  vision_norm_kernel<<<rows, 256, 0, stream>>>(vision_read(value), vision_read(norm_weight),
                                               vision_read(norm_bias), vision_write(normalized));
  bf16_linear<float>(vision_read(normalized), vision_read(up_weight), matrix, rows / 4, 4608, 4608,
                     stream);
  vision_activation_kernel<<<(int64_t(rows) * 1152 + 255) / 256, 256, 0, stream>>>(
      matrix, vision_read(up_bias), vision_write(activated), int64_t(rows) * 1152, 4608, false);
  bf16_linear<float>(vision_read(activated), vision_read(down_weight), matrix, rows / 4, 5120, 4608,
                     stream);
  vision_bias_kernel<<<(int64_t(rows / 4) * 5120 + 255) / 256, 256, 0, stream>>>(
      matrix, vision_read(down_bias), vision_write(output), int64_t(rows / 4) * 5120, 5120);
  check_cuda(cudaGetLastError());
}
