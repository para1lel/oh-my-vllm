#pragma once

#include "common.cuh"

__device__ float warp_max(float value) {
#pragma unroll
  for (int offset = 16; offset; offset >>= 1)
    value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
  return __shfl_sync(0xffffffff, value, 0);
}

// Softmax uses the same hardware base-two exponential family as the frozen
// TileLang path. Keep approximation local: RoPE and FP8 retain their math rules.
__device__ __forceinline__ float softmax_exp2(float value) {
  float result;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(value));
  return result;
}

// PTX mma.m16n8k16 BF16 fragment mapping follows the NVIDIA ISA guide.
// XOR eight-element sectors across rows to avoid shared-memory bank conflicts.
__device__ __forceinline__ int shared_index(int row, int col, int stride = 256) {
  return row * stride + (col ^ ((row & 7) * 8));
}
__device__ __forceinline__ void load_a(unsigned (&a)[4], const __nv_bfloat16 *data, int inner,
                                       int lane, int stride = 256) {
  unsigned address =
      __cvta_generic_to_shared(data + shared_index(lane % 16, inner + (lane / 16) * 8, stride));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
               : "r"(address));
}
__device__ __forceinline__ void load_k(unsigned (&b)[2], const __nv_bfloat16 *data, int inner,
                                       int warp, int lane, int stride = 256) {
  unsigned address = __cvta_generic_to_shared(
      data + shared_index(warp * 8 + lane % 8, inner + ((lane / 8) % 2) * 8, stride));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
               : "=r"(b[0]), "=r"(b[1])
               : "r"(address));
}
__device__ __forceinline__ void load_v(unsigned (&b)[2], const __nv_bfloat16 *data, int inner,
                                       int col, int lane, int stride = 256) {
  unsigned address = __cvta_generic_to_shared(data + shared_index(inner + lane % 16, col, stride));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
               : "=r"(b[0]), "=r"(b[1])
               : "r"(address));
}
__device__ __forceinline__ void mma_bf16(float (&c)[4], unsigned a0, unsigned a1, unsigned a2,
                                         unsigned a3, unsigned b0, unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
