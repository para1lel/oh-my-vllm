#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>

using tvm::ffi::TensorView;

// One warp owns one complete 128-value scaling group. Independent warps share
// a CTA; reductions do not require shared memory or block-wide synchronization.
template <typename Input, bool Silu, bool Column>
__global__ void quantize_kernel(const Input* x, __nv_fp8_e4m3* out, float* scales,
                                int rows, int width) {
  const int lane = threadIdx.x & 31;
  const int group = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  const int groups = width / 128;
  if (group >= rows * groups) return;
  const int row = group / groups;
  const int col = (group % groups) * 128;
  float value[4];
  float maximum = 0;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    int offset = row * width * (Silu ? 2 : 1) + col + lane + j * 32;
    float v = static_cast<float>(x[offset]);
    if constexpr (Silu) {
      float activated = __bfloat162float(__float2bfloat16_rn(v / (1.f + expf(-v))));
      v = __bfloat162float(__float2bfloat16_rn(
          activated * static_cast<float>(x[offset + width])));
    }
    value[j] = v;
    maximum = fmaxf(maximum, fabsf(v));
  }
#pragma unroll
  for (int delta = 16; delta; delta /= 2)
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, delta));
  float scale = __fdiv_rn(fmaxf(maximum, 1e-10f), 448.f);
  if (lane == 0) scales[Column ? (group % groups) * rows + row : group] = scale;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    float v = fminf(448.f, fmaxf(-448.f, __fdiv_rn(value[j], scale)));
    out[row * width + col + lane + j * 32] = __nv_fp8_e4m3(v);
  }
}

template <typename Input>
void launch_quantize(TensorView x, TensorView out, TensorView scales,
                     bool column, bool silu, cudaStream_t stream) {
  int rows = out.size(0), width = out.size(1);
  int blocks = (rows * (width / 128) + 3) / 4;
#define LAUNCH(S, C) quantize_kernel<Input, S, C><<<blocks, 128, 0, stream>>>( \
  static_cast<const Input*>(x.data_ptr()), \
  static_cast<__nv_fp8_e4m3*>(out.data_ptr()), \
  static_cast<float*>(scales.data_ptr()), rows, width)
  if (silu) {
    if (column) { LAUNCH(true, true); } else { LAUNCH(true, false); }
  } else {
    if (column) { LAUNCH(false, true); } else { LAUNCH(false, false); }
  }
#undef LAUNCH
}

void quantize(TensorView x, TensorView out, TensorView scales, bool column, bool silu) {
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, x.device().device_id));
  if (x.dtype().code == kDLBfloat) launch_quantize<__nv_bfloat16>(x, out, scales, column, silu, stream);
  else if (x.dtype().bits == 16) launch_quantize<__half>(x, out, scales, column, silu, stream);
  else launch_quantize<float>(x, out, scales, column, silu, stream);
  TVM_FFI_ICHECK(cudaGetLastError() == cudaSuccess) << "CUDA quantize launch failed";
}

__global__ void silu_kernel(const __nv_bfloat16* x, __nv_bfloat16* out, int rows, int width) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * width) return;
  int offset = (i / width) * width * 2 + i % width;
  float g = __bfloat162float(x[offset]);
  float a = __bfloat162float(__float2bfloat16_rn(g / (1.f + expf(-g))));
  out[i] = __float2bfloat16_rn(a * __bfloat162float(x[offset + width]));
}
void silu_mul(TensorView x, TensorView out) {
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, x.device().device_id));
  int n = out.size(0) * out.size(1);
  if (!n) return;
  silu_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(x.data_ptr()),
      static_cast<__nv_bfloat16*>(out.data_ptr()), out.size(0), out.size(1));
  TVM_FFI_ICHECK(cudaGetLastError() == cudaSuccess) << "CUDA SiLU launch failed";
}
