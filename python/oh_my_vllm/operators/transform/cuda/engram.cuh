#pragma once

namespace {

struct EngramSums {
  float query, key, dot;
};

__device__ EngramSums add_engram_sums(EngramSums left, EngramSums right) {
  return {__fadd_rn(left.query, right.query), __fadd_rn(left.key, right.key),
          __fadd_rn(left.dot, right.dot)};
}

template <int Threads>
__global__ void engram_kernel(const __nv_bfloat16 *residual, const __nv_bfloat16 *kv,
                              const __nv_bfloat16 *q_weight, const __nv_bfloat16 *k_weight,
                              const uint8_t *token_mask, __nv_bfloat16 *output) {
  int row = blockIdx.x, token = row / 4, copy = row % 4, thread = threadIdx.x;
  __shared__ EngramSums partials[Threads];
  __shared__ float gate;
  if (!token_mask || token_mask[token]) {
    // Source FP32 means and dot use four independent accumulators, then an X tree.
    EngramSums sums[4] = {};
    for (int vector = thread; vector < 5120 / 4; vector += Threads) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        int column = vector * 4 + i;
        float h = __bfloat162float(residual[row * 5120 + column]);
        float key = __bfloat162float(kv[token * 25600 + copy * 5120 + column]);
        float weight = __fmul_rn(__bfloat162float(q_weight[copy * 5120 + column]),
                                 __bfloat162float(k_weight[copy * 5120 + column]));
        sums[i].query = __fadd_rn(sums[i].query, __fmul_rn(h, h));
        sums[i].key = __fadd_rn(sums[i].key, __fmul_rn(key, key));
        sums[i].dot = __fadd_rn(sums[i].dot, __fmul_rn(__fmul_rn(h, weight), key));
      }
    }
    EngramSums sum = sums[0];
#pragma unroll
    for (int i = 1; i < 4; ++i)
      sum = add_engram_sums(sum, sums[i]);
    if constexpr (Threads > 32) {
      partials[thread] = sum;
      for (int offset = Threads / 2; offset >= 32; offset >>= 1) {
        __syncthreads();
        if (thread < offset) {
          sum = add_engram_sums(sum, partials[thread + offset]);
          partials[thread] = sum;
        }
      }
    }
    __syncthreads();
    for (int offset = 16; offset; offset >>= 1) {
      EngramSums other = {__shfl_down_sync(0xffffffff, sum.query, offset),
                          __shfl_down_sync(0xffffffff, sum.key, offset),
                          __shfl_down_sync(0xffffffff, sum.dot, offset)};
      sum = add_engram_sums(sum, other);
    }
    if (thread == 0) {
      float qmean = __fmul_rn(sum.query, 1.f / 5120);
      float kmean = __fmul_rn(sum.key, 1.f / 5120);
      float rstd = __fmul_rn(rsqrtf(__fadd_rn(qmean, 1e-20f)), rsqrtf(__fadd_rn(kmean, 1e-20f)));
      // Python dim**-0.5 rounds once from FP64 to FP32, before multiplication.
      float dot = __fmul_rn(__fmul_rn(sum.dot, rstd), 0.013975424859373685f);
      float signed_root = copysignf(sqrtf(fmaxf(fabsf(dot), 1e-6f)), dot);
      gate = __fdiv_rn(1.f, __fadd_rn(1.f, expf(-signed_root)));
    }
  } else if (thread == 0) {
    gate = 0.f;
  }
  __syncthreads();
  for (int column = thread; column < 5120; column += Threads) {
    float h = __bfloat162float(residual[row * 5120 + column]);
    float value = __bfloat162float(kv[token * 25600 + 20480 + column]);
    // Keep both roundings, including signed zeros when the mask sets gate to +0.
    output[row * 5120 + column] = __float2bfloat16_rn(__fadd_rn(h, __fmul_rn(gate, value)));
  }
}

template <int Threads>
void launch_engram(TensorView residual, TensorView kv, TensorView q_weight, TensorView k_weight,
                   TensorView token_mask, TensorView output, int rows, cudaStream_t stream) {
  engram_kernel<Threads><<<rows * 4, Threads, 0, stream>>>(
      static_cast<const __nv_bfloat16 *>(residual.data_ptr()),
      static_cast<const __nv_bfloat16 *>(kv.data_ptr()),
      static_cast<const __nv_bfloat16 *>(q_weight.data_ptr()),
      static_cast<const __nv_bfloat16 *>(k_weight.data_ptr()),
      token_mask.size(0) ? static_cast<const uint8_t *>(token_mask.data_ptr()) : nullptr,
      static_cast<__nv_bfloat16 *>(output.data_ptr()));
}

} // namespace
