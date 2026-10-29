#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <limits>
#include "../../runtime/cuda/contracts.cuh"

namespace {

__device__ float ceil_power_of_two(float value) {
  uint32_t bits = __float_as_uint(value);
  return __uint_as_float((bits & 0x7f800000u) + ((bits & 0x007fffffu) ? 0x00800000u : 0));
}

__device__ float quantization_scale(float maximum, int bits, int format, bool round_scale,
                                    bool invalid, uint8_t &code) {
  float scale;
  if (bits == 8) {
    scale = __fmul_rn(fmaxf(maximum, 1e-4f), 1.f / 448.f);
    if (round_scale)
      scale = ceil_power_of_two(scale);
  } else if (format == 1) {
    scale = ceil_power_of_two(__fmul_rn(fmaxf(maximum, 6.f * 0x1p-126f), 1.f / 6.f));
  } else {
    float unrounded = __fdiv_rn(fmaxf(maximum, 6.f * 0x1p-9f), 6.f);
    // The hardware cast saturates; reject scales beyond E4M3's nearest-even overflow midpoint.
    invalid |= !isfinite(unrounded) || unrounded > 464.f;
    code = __nv_cvt_float_to_fp8(unrounded, __NV_SATFINITE, __NV_E4M3);
    __nv_fp8_e4m3 stored;
    stored.__x = code;
    scale = static_cast<float>(stored);
  }
  invalid |= !isfinite(scale) || scale <= 0.f;
  if (format == 1) {
    float stored = ceil_power_of_two(scale);
    invalid |= !isfinite(stored);
    code = static_cast<uint8_t>(__float_as_uint(stored) >> 23);
  }
  if (invalid) {
    code = format == 2 ? 0x7f : 0xff;
    return __uint_as_float(0x7fc00000u);
  }
  return scale;
}

__global__ void quantize_kernel(const __nv_bfloat162 *values, void *output, void *scales,
                                int64_t groups, int bits, int group_size, int format,
                                bool round_scale, bool inplace) {
  int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  int pairs = group_size / 2;
  for (int64_t group = int64_t(blockIdx.x) * 4 + warp; group < groups;
       group += int64_t(gridDim.x) * 4) {
    float2 input[2];
    float maximum = 0.f;
    bool invalid = false;
    // One warp owns a group and reads it entirely before any exact-alias output write.
    for (int i = 0; i < 2; ++i) {
      int pair = lane + i * 32;
      input[i] =
          pair < pairs ? __bfloat1622float2(values[group * pairs + pair]) : make_float2(0.f, 0.f);
      invalid |= !isfinite(input[i].x) || !isfinite(input[i].y);
      maximum = fmaxf(maximum, fmaxf(fabsf(input[i].x), fabsf(input[i].y)));
    }
    for (int offset = 16; offset; offset >>= 1)
      maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, offset));
    invalid = __any_sync(0xffffffff, invalid);
    uint8_t scale_code = 0;
    float scale = 0.f;
    if (lane == 0) {
      scale = quantization_scale(maximum, bits, format, round_scale, invalid, scale_code);
      if (format == 0)
        static_cast<float *>(scales)[group] = scale;
      else
        static_cast<uint8_t *>(scales)[group] = scale_code;
    }
    scale = __shfl_sync(0xffffffff, scale, 0);
    invalid = !isfinite(scale);
    for (int i = 0; i < 2; ++i) {
      int pair = lane + i * 32;
      if (pair >= pairs)
        continue;
      float limit = bits == 4 ? 6.f : 448.f;
      float2 normalized = make_float2(fminf(fmaxf(__fdiv_rn(input[i].x, scale), -limit), limit),
                                      fminf(fmaxf(__fdiv_rn(input[i].y, scale), -limit), limit));
      float2 decoded;
      if (bits == 4) {
        uint8_t code =
            invalid ? 0 : __nv_cvt_float2_to_fp4x2(normalized, __NV_E2M1, cudaRoundNearest);
        if (!inplace) {
          static_cast<uint8_t *>(output)[group * pairs + pair] = code;
          continue;
        }
        decoded = __half22float2(static_cast<__half2>(__nv_cvt_fp4x2_to_halfraw2(code, __NV_E2M1)));
      } else {
        uint16_t code =
            invalid ? 0x7f7f : __nv_cvt_float2_to_fp8x2(normalized, __NV_SATFINITE, __NV_E4M3);
        if (!inplace) {
          static_cast<uint16_t *>(output)[group * pairs + pair] = code;
          continue;
        }
        __nv_fp8x2_e4m3 stored;
        stored.__x = code;
        decoded = static_cast<float2>(stored);
      }
      static_cast<__nv_bfloat162 *>(output)[group * pairs + pair] =
          __floats2bfloat162_rn(__fmul_rn(decoded.x, scale), __fmul_rn(decoded.y, scale));
    }
  }
}

void quantization_span(TensorView tensor, int64_t bytes) {
  auto address = reinterpret_cast<uintptr_t>(tensor.data_ptr());
  TVM_FFI_ICHECK(bytes >= 0 && (bytes == 0 || address != 0));
  TVM_FFI_ICHECK(static_cast<uint64_t>(bytes) <= std::numeric_limits<uintptr_t>::max() - address)
      << "quantization tensor address overflow";
}

template <int Threads>
__global__ void rms_kernel(const __nv_bfloat162 *values, const __nv_bfloat162 *weight,
                           __nv_bfloat162 *output, int pairs, float epsilon) {
  int row = blockIdx.x, thread = threadIdx.x;
  float sums[4] = {};
  for (int vector = thread; vector < pairs / 2; vector += Threads) {
    float2 first = __bfloat1622float2(values[row * pairs + vector * 2]);
    float2 second = __bfloat1622float2(values[row * pairs + vector * 2 + 1]);
    sums[0] = __fadd_rn(sums[0], __fmul_rn(first.x, first.x));
    sums[1] = __fadd_rn(sums[1], __fmul_rn(first.y, first.y));
    sums[2] = __fadd_rn(sums[2], __fmul_rn(second.x, second.x));
    sums[3] = __fadd_rn(sums[3], __fmul_rn(second.y, second.y));
  }
  float sum = sums[0];
  for (int i = 1; i < 4; ++i)
    sum = __fadd_rn(sum, sums[i]);
  __shared__ float partials[Threads], inverse;
  if constexpr (Threads > 32) {
    partials[thread] = sum;
    for (int offset = Threads / 2; offset >= 32; offset >>= 1) {
      __syncthreads();
      if (thread < offset) {
        sum = __fadd_rn(sum, partials[thread + offset]);
        partials[thread] = sum;
      }
    }
  }
  __syncthreads();
  for (int offset = 16; offset; offset >>= 1)
    sum = __fadd_rn(sum, __shfl_down_sync(0xffffffff, sum, offset));
  if (thread == 0) {
    float mean = __fmul_rn(sum, 1.f / (2 * pairs));
    inverse = rsqrtf(__fadd_rn(mean, epsilon));
  }
  __syncthreads();
  for (int column = thread; column < pairs; column += Threads) {
    float2 value = __bfloat1622float2(values[row * pairs + column]);
    float2 scale = __bfloat1622float2(weight[column]);
    output[row * pairs + column] =
        __floats2bfloat162_rn(__fmul_rn(__fmul_rn(value.x, inverse), scale.x),
                              __fmul_rn(__fmul_rn(value.y, inverse), scale.y));
  }
}

int rms_threads(int rows, int width) {
  int height = 1, threads = 32;
  while (height < 16 && height * 2 <= rows)
    height *= 2;
  while (threads < 512 && threads * 2 <= width / 4)
    threads *= 2;
  // Source mean assigns y to different rows, not to a second reduction.
  return std::min(threads, 512 / height);
}

template <int Threads>
void launch_rms(TensorView values, TensorView weight, TensorView output, int rows, int pairs,
                float epsilon, cudaStream_t stream) {
  rms_kernel<Threads><<<rows, Threads, 0, stream>>>(
      static_cast<const __nv_bfloat162 *>(values.data_ptr()),
      static_cast<const __nv_bfloat162 *>(weight.data_ptr()),
      static_cast<__nv_bfloat162 *>(output.data_ptr()), pairs, epsilon);
}

__global__ void rope_kernel(const __nv_bfloat162 *values, const float2 *coefficients,
                            __nv_bfloat162 *output, int pairs, int heads, bool inverse) {
  int row = blockIdx.x;
  for (int column = threadIdx.x; column < pairs; column += blockDim.x) {
    auto encoded = values[row * pairs + column];
    if (column < pairs - 32) {
      output[row * pairs + column] = encoded;
      continue;
    }
    float2 value = __bfloat1622float2(encoded);
    float2 coefficient = coefficients[(row / heads) * 32 + column - (pairs - 32)];
    float sine = inverse ? -coefficient.y : coefficient.y;
    output[row * pairs + column] = __floats2bfloat162_rn(value.x * coefficient.x - value.y * sine,
                                                         value.x * sine + value.y * coefficient.x);
  }
}

void values_contract(TensorView values, TensorView output) {
  TVM_FFI_ICHECK(values.ndim() == 2 && values.size(0) >= 0);
  tensor_contract(values, values, kDLBfloat, 16, {values.size(0), values.size(1)});
  tensor_contract(output, values, kDLBfloat, 16, {values.size(0), values.size(1)});
  disjoint(values, output);
}

} // namespace

void quantize(TensorView values, TensorView output, TensorView scales, int64_t bits,
              int64_t group_size, int64_t scale_format, bool round_scale, bool inplace) {
  TVM_FFI_ICHECK(values.ndim() == 2 && values.size(0) >= 0 && values.size(1) > 0);
  TVM_FFI_ICHECK(values.device().device_id >= 0);
  TVM_FFI_ICHECK(group_size == 16 || group_size == 32 || group_size == 128);
  int64_t rows = values.size(0), width = values.size(1);
  TVM_FFI_ICHECK(width % group_size == 0);
  TVM_FFI_ICHECK(bits == 4 || bits == 8);
  TVM_FFI_ICHECK((bits == 8 && (scale_format == 0 || scale_format == 1)) ||
                 (bits == 4 && (scale_format == 1 || scale_format == 2)));
  TVM_FFI_ICHECK(width <= std::numeric_limits<int64_t>::max() / 2);
  TVM_FFI_ICHECK(rows <= std::numeric_limits<int64_t>::max() / 2 / width)
      << "quantization tensor size overflow";
  int64_t elements = rows * width, groups = elements / group_size;
  tensor_contract(values, values, kDLBfloat, 16, {rows, width});
  tensor_contract(output, values, inplace ? kDLBfloat : kDLUInt, inplace ? 16 : 8,
                  {rows, inplace ? width : width / (8 / bits)});
  tensor_contract(scales, values, scale_format == 0 ? kDLFloat : kDLUInt,
                  scale_format == 0 ? 32 : 8, {rows, width / group_size});
  quantization_span(values, elements * 2);
  quantization_span(output, inplace ? elements * 2 : elements / (8 / bits));
  quantization_span(scales, groups * (scale_format == 0 ? 4 : 1));
  if (!inplace || values.data_ptr() != output.data_ptr())
    disjoint(values, output);
  disjoint(values, scales);
  disjoint(output, scales);
  if (!rows)
    return;
  tvm::ffi::CUDADeviceGuard guard(values.device().device_id);
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, values.device().device_id));
  int64_t blocks = (groups + 3) / 4;
  if (blocks > 65535)
    blocks = 65535;
  quantize_kernel<<<static_cast<unsigned>(blocks), 128, 0, stream>>>(
      static_cast<const __nv_bfloat162 *>(values.data_ptr()), output.data_ptr(), scales.data_ptr(),
      groups, static_cast<int>(bits), static_cast<int>(group_size), static_cast<int>(scale_format),
      round_scale, inplace);
  check_cuda(cudaGetLastError());
}

void rms(TensorView values, TensorView weight, TensorView output, double epsilon) {
  values_contract(values, output);
  TVM_FFI_ICHECK(values.size(1) == 512 || values.size(1) == 1024 || values.size(1) == 1280 ||
                 values.size(1) == 5120);
  TVM_FFI_ICHECK(epsilon == 1e-20 || epsilon == 1e-6);
  int rows = values.size(0), width = values.size(1);
  tensor_contract(weight, values, kDLBfloat, 16, {width});
  disjoint(weight, output);
  if (!rows)
    return;
  tvm::ffi::CUDADeviceGuard guard(values.device().device_id);
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, values.device().device_id));
  switch (rms_threads(rows, width)) {
  case 32:
    launch_rms<32>(values, weight, output, rows, width / 2, epsilon, stream);
    break;
  case 64:
    launch_rms<64>(values, weight, output, rows, width / 2, epsilon, stream);
    break;
  case 128:
    launch_rms<128>(values, weight, output, rows, width / 2, epsilon, stream);
    break;
  case 256:
    launch_rms<256>(values, weight, output, rows, width / 2, epsilon, stream);
    break;
  case 512:
    launch_rms<512>(values, weight, output, rows, width / 2, epsilon, stream);
    break;
  }
  check_cuda(cudaGetLastError());
}

void rope(TensorView values, TensorView coefficients, TensorView output, bool inverse) {
  TVM_FFI_ICHECK(values.ndim() == 3 && values.size(0) >= 0);
  TVM_FFI_ICHECK(values.size(1) == 1 || values.size(1) == 8 || values.size(1) == 16 ||
                 values.size(1) == 32 || values.size(1) == 64);
  TVM_FFI_ICHECK(values.size(2) == 128 || values.size(2) == 512);
  int rows = values.size(0), heads = values.size(1), width = values.size(2);
  tensor_contract(values, values, kDLBfloat, 16, {rows, heads, width});
  tensor_contract(output, values, kDLBfloat, 16, {rows, heads, width});
  tensor_contract(coefficients, values, kDLFloat, 32, {rows, 32, 2});
  disjoint(values, output);
  disjoint(coefficients, output);
  if (!rows)
    return;
  tvm::ffi::CUDADeviceGuard guard(values.device().device_id);
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, values.device().device_id));
  rope_kernel<<<rows * heads, 128, 0, stream>>>(
      static_cast<const __nv_bfloat162 *>(values.data_ptr()),
      static_cast<const float2 *>(coefficients.data_ptr()),
      static_cast<__nv_bfloat162 *>(output.data_ptr()), width / 2, heads, inverse);
  check_cuda(cudaGetLastError());
}
