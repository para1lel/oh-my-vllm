#pragma once

#include "../../primitives/cuda/common.cuh"

namespace {

__global__ void engram_blend_kernel(const __nv_bfloat162 *residual, const __nv_bfloat162 *projected,
                                    const __nv_bfloat162 *q_weight, const __nv_bfloat162 *k_weight,
                                    const uint8_t *mask, float *gates, __nv_bfloat162 *output) {
  constexpr int pairs = 2560;
  int row = blockIdx.x / 4, copy = blockIdx.x % 4;
  int base = (row * 4 + copy) * pairs;
  if (mask[row] > 1) {
    if (threadIdx.x == 0)
      asm volatile("trap;");
    return;
  }
  if (!mask[row]) {
    if (threadIdx.x == 0)
      gates[row * 4 + copy] = 0.f;
    for (int column = threadIdx.x; column < pairs; column += blockDim.x)
      output[base + column] = residual[base + column];
    return;
  }
  float h2 = 0.f, k2 = 0.f, dot = 0.f;
  for (int column = threadIdx.x; column < pairs; column += blockDim.x) {
    float2 h = __bfloat1622float2(residual[base + column]);
    float2 key = __bfloat1622float2(projected[(row * 5 + copy) * pairs + column]);
    float2 qw = __bfloat1622float2(q_weight[copy * pairs + column]);
    float2 kw = __bfloat1622float2(k_weight[copy * pairs + column]);
    h2 += h.x * h.x + h.y * h.y;
    k2 += key.x * key.x + key.y * key.y;
    // Sum adjacent terms before accumulation to retain small signed residuals.
    dot += (h.x * (qw.x * kw.x)) * key.x + (h.y * (qw.y * kw.y)) * key.y;
  }
  __shared__ float totals[3][8];
  int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  h2 = warp_sum(h2);
  k2 = warp_sum(k2);
  dot = warp_sum(dot);
  if (lane == 0) {
    totals[0][warp] = h2;
    totals[1][warp] = k2;
    totals[2][warp] = dot;
  }
  __syncthreads();
  h2 = warp_sum(lane < 8 ? totals[0][lane] : 0.f);
  k2 = warp_sum(lane < 8 ? totals[1][lane] : 0.f);
  dot = warp_sum(lane < 8 ? totals[2][lane] : 0.f);
  float rstd = rsqrtf(h2 / 5120.f + 1e-20f) * rsqrtf(k2 / 5120.f + 1e-20f);
  dot = (dot * rstd) * 0.013975424859373685f;
  float score = copysignf(sqrtf(fmaxf(fabsf(dot), 1e-6f)), dot);
  float gate = 1.f / (1.f + expf(-score));
  if (threadIdx.x == 0)
    gates[row * 4 + copy] = gate;
  for (int column = threadIdx.x; column < pairs; column += blockDim.x) {
    float2 h = __bfloat1622float2(residual[base + column]);
    float2 value = __bfloat1622float2(projected[(row * 5 + 4) * pairs + column]);
    output[base + column] = __floats2bfloat162_rn(h.x + gate * value.x, h.y + gate * value.y);
  }
}

} // namespace

void engram_blend(TensorView residual, TensorView projected, TensorView q_weight,
                  TensorView k_weight, TensorView mask, TensorView gate, TensorView output) {
  TVM_FFI_ICHECK(residual.ndim() == 3 && residual.size(0) >= 0 && residual.size(0) <= 384);
  int rows = residual.size(0);
  tensor_contract(residual, residual, kDLBfloat, 16, {rows, 4, 5120});
  tensor_contract(projected, residual, kDLBfloat, 16, {rows, 5, 5120});
  tensor_contract(q_weight, residual, kDLBfloat, 16, {4, 5120});
  tensor_contract(k_weight, residual, kDLBfloat, 16, {4, 5120});
  tensor_contract(mask, residual, kDLUInt, 8, {rows});
  tensor_contract(gate, residual, kDLFloat, 32, {rows, 4});
  tensor_contract(output, residual, kDLBfloat, 16, {rows, 4, 5120});
  for (auto result : {gate, output})
    for (auto input : {residual, projected, q_weight, k_weight, mask})
      disjoint(input, result);
  disjoint(gate, output);
  if (!rows)
    return;
  tvm::ffi::CUDADeviceGuard guard(residual.device().device_id);
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, residual.device().device_id));
  engram_blend_kernel<<<rows * 4, 256, 0, stream>>>(
      static_cast<const __nv_bfloat162 *>(residual.data_ptr()),
      static_cast<const __nv_bfloat162 *>(projected.data_ptr()),
      static_cast<const __nv_bfloat162 *>(q_weight.data_ptr()),
      static_cast<const __nv_bfloat162 *>(k_weight.data_ptr()),
      static_cast<const uint8_t *>(mask.data_ptr()), static_cast<float *>(gate.data_ptr()),
      static_cast<__nv_bfloat162 *>(output.data_ptr()));
  check_cuda(cudaGetLastError());
}
