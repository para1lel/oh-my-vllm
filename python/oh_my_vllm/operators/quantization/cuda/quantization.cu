#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include <climits>
#include <cstdint>
#include <type_traits>

#include "../../runtime/cuda/contracts.cuh"

namespace {

__device__ float decode_e4m3(uint8_t code) {
  return __half2float(static_cast<__half>(__nv_cvt_fp8_to_halfraw(code, __NV_E4M3)));
}

template <int Mode>
__device__ void reject_group(int group, uint8_t error, uint8_t *encoded, uint8_t *scales,
                             __nv_bfloat16 *dequantized, uint8_t *status) {
  constexpr int width = Mode == 2 ? 16 : 32;
  constexpr int bytes = Mode == 0 ? width : width / 2;
  for (int j = 0; j < bytes; ++j)
    encoded[group * bytes + j] = 0;
  for (int j = 0; j < width; ++j)
    dequantized[group * width + j] = __float2bfloat16_rn(0.f);
  scales[group] = 0;
  status[group] = error;
}

template <int Mode>
__global__ void quantize_kernel(const __nv_bfloat16 *input, uint8_t *encoded, uint8_t *scales,
                                __nv_bfloat16 *dequantized, uint8_t *status, int groups) {
  int group = blockIdx.x * blockDim.x + threadIdx.x;
  if (group >= groups)
    return;
  constexpr int width = Mode == 2 ? 16 : 32;
  constexpr float maximum = Mode == 0 ? 448.f : 6.f;
  float values[width];
  float amax = 0.f;
  for (int j = 0; j < width; ++j) {
    values[j] = __bfloat162float(input[group * width + j]);
    if (!isfinite(values[j])) {
      reject_group<Mode>(group, 1, encoded, scales, dequantized, status);
      return;
    }
    amax = fmaxf(amax, fabsf(values[j]));
  }

  uint8_t scale_code;
  float scale;
  if constexpr (Mode == 2) {
    amax = fmaxf(amax, 6.f * 0x1p-9f);
    scale_code = __nv_cvt_float_to_fp8(__fdiv_rn(amax, 6.f), __NV_SATFINITE, __NV_E4M3);
    scale = decode_e4m3(scale_code);
  } else {
    amax = fmaxf(amax, Mode == 0 ? 1e-4f : 6.f * 0x1p-126f);
    // Released fast_round_scale multiplies by the FP32 reciprocal before ceil(log2).
    uint32_t bits = __float_as_uint(__fmul_rn(amax, 1.f / maximum));
    uint32_t exponent = (bits >> 23) + ((bits & 0x7fffffU) != 0);
    if (exponent < 1 || exponent > 254) {
      reject_group<Mode>(group, 2, encoded, scales, dequantized, status);
      return;
    }
    scale_code = static_cast<uint8_t>(exponent);
    scale = __uint_as_float(exponent << 23);
  }

  uint8_t codes[width];
  bool overflow = false;
  for (int j = 0; j < width; ++j) {
    float normalized = __fdiv_rn(values[j], scale);
    normalized = fminf(fmaxf(normalized, -maximum), maximum);
    float decoded;
    if constexpr (Mode == 0) {
      codes[j] = __nv_cvt_float_to_fp8(normalized, __NV_SATFINITE, __NV_E4M3);
      decoded = decode_e4m3(codes[j]);
    } else {
      codes[j] = __nv_cvt_float_to_fp4(normalized, __NV_E2M1, cudaRoundNearest);
      decoded = __half2float(static_cast<__half>(__nv_cvt_fp4_to_halfraw(codes[j], __NV_E2M1)));
    }
    float restored = __fmul_rn(decoded, scale);
    auto rounded = __float2bfloat16_rn(restored);
    overflow |= !isfinite(restored) || !isfinite(__bfloat162float(rounded));
    dequantized[group * width + j] = rounded;
  }
  if (overflow) {
    reject_group<Mode>(group, 3, encoded, scales, dequantized, status);
    return;
  }
  if constexpr (Mode == 0) {
    for (int j = 0; j < width; ++j)
      encoded[group * width + j] = codes[j];
  } else {
    for (int j = 0; j < width; j += 2)
      encoded[group * (width / 2) + j / 2] = codes[j] | (codes[j + 1] << 4);
  }
  scales[group] = scale_code;
  status[group] = 0;
}

} // namespace

void quantize(TensorView value, TensorView encoded, TensorView scales, TensorView dequantized,
              TensorView status, int64_t mode) {
  TVM_FFI_ICHECK(mode >= 0 && mode <= 2) << "unsupported quantization mode";
  TVM_FFI_ICHECK(value.ndim() == 2);
  int64_t rows = value.size(0), width = value.size(1);
  int group_size = mode == 2 ? 16 : 32;
  TVM_FFI_ICHECK(rows >= 0 && rows <= INT_MAX && width > 0 && width <= INT_MAX &&
                 rows <= INT_MAX / width && width % group_size == 0);
  tensor_contract(value, value, kDLBfloat, 16, {rows, width});
  tensor_contract(encoded, value, kDLUInt, 8, {rows, mode == 0 ? width : width / 2});
  tensor_contract(scales, value, kDLUInt, 8, {rows, width / group_size});
  tensor_contract(dequantized, value, kDLBfloat, 16, {rows, width});
  tensor_contract(status, value, kDLUInt, 8, {rows, width / group_size});
  auto tensors = {value, encoded, scales, dequantized, status};
  int left = 0;
  for (auto a : tensors) {
    int right = 0;
    for (auto b : tensors)
      if (right++ > left)
        disjoint(a, b);
    ++left;
  }
  if (rows == 0)
    return;
  tvm::ffi::CUDADeviceGuard guard(value.device().device_id);
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, value.device().device_id));
  int groups = rows * width / group_size;
  auto launch = [&](auto tag) {
    quantize_kernel<decltype(tag)::value><<<(groups + 127) / 128, 128, 0, stream>>>(
        static_cast<const __nv_bfloat16 *>(value.data_ptr()),
        static_cast<uint8_t *>(encoded.data_ptr()), static_cast<uint8_t *>(scales.data_ptr()),
        static_cast<__nv_bfloat16 *>(dequantized.data_ptr()),
        static_cast<uint8_t *>(status.data_ptr()), groups);
  };
  if (mode == 0)
    launch(std::integral_constant<int, 0>{});
  else if (mode == 1)
    launch(std::integral_constant<int, 1>{});
  else
    launch(std::integral_constant<int, 2>{});
  check_cuda(cudaGetLastError());
}
