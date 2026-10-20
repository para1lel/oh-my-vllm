#pragma once

#include "../../primitives/cuda/common.cuh"

template <int Threads, class Residual>
__global__ void final_hidden_kernel(const __nv_bfloat16 *value, const Residual *residual,
                                    const float *weight, __nv_bfloat16 *output,
                                    __nv_bfloat16 *cache, const int32_t *slots, int cache_rows) {
  int row = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x / 32;
  int slot = slots[row];
  if (slot < -1 || slot >= cache_rows) {
    asm volatile("trap;");
    return;
  }
  float values[5120 / Threads], square = 0;
#pragma unroll
  for (int i = 0; i < 5120 / Threads; ++i) {
    int column = threadIdx.x + i * Threads;
    values[i] = __bfloat162float(value[row * 5120 + column]) + float(residual[row * 5120 + column]);
    square += values[i] * values[i];
  }
  __shared__ float partial[Threads / 32];
  square = warp_sum(square);
  if (lane == 0)
    partial[warp] = square;
  __syncthreads();
  float total = warp_sum(lane < Threads / 32 ? partial[lane] : 0.f);
  float inverse = rsqrtf(total / 5120.f + 1e-6f);
#pragma unroll
  for (int i = 0; i < 5120 / Threads; ++i) {
    int column = threadIdx.x + i * Threads;
    auto normalized = __float2bfloat16_rn(values[i] * inverse * (weight[column] + 1.f));
    output[row * 5120 + column] = normalized;
    if (slot >= 0)
      cache[int64_t(slot) * 5120 + column] = normalized;
  }
}

void final_hidden(TensorView value, TensorView residual, TensorView weight, TensorView output,
                  TensorView cache, TensorView slots) {
  TVM_FFI_ICHECK(value.device().device_type == kDLCUDA && value.ndim() == 2 && value.size(0) > 0 &&
                 value.size(0) <= 32768);
  int rows = value.size(0);
  for (auto tensor : {value, output})
    tensor_contract(tensor, value, kDLBfloat, 16, {rows, 5120});
  residual_contract(residual, value, rows);
  tensor_contract(weight, value, kDLFloat, 32, {5120});
  TVM_FFI_ICHECK(cache.ndim() == 2 && cache.size(0) > 0 && cache.size(0) <= 65536);
  tensor_contract(cache, value, kDLBfloat, 16, {cache.size(0), 5120});
  tensor_contract(slots, value, kDLInt, 32, {rows});
  for (auto tensor : {value, residual, weight, slots}) {
    disjoint(tensor, output);
    disjoint(tensor, cache);
  }
  disjoint(output, cache);
  tvm::ffi::CUDADeviceGuard guard(value.device().device_id);
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, value.device().device_id));
  dispatch_residual(residual, [&](auto skip) {
    auto launch = [&](auto block) {
      constexpr int threads = decltype(block)::value;
      final_hidden_kernel<threads><<<rows, threads, 0, stream>>>(
          static_cast<const __nv_bfloat16 *>(value.data_ptr()), skip,
          static_cast<const float *>(weight.data_ptr()),
          static_cast<__nv_bfloat16 *>(output.data_ptr()),
          static_cast<__nv_bfloat16 *>(cache.data_ptr()),
          static_cast<const int32_t *>(slots.data_ptr()), cache.size(0));
    };
    if (rows <= 32)
      launch(std::integral_constant<int, 1024>{});
    else
      launch(std::integral_constant<int, 640>{});
  });
  check_cuda(cudaGetLastError());
}
