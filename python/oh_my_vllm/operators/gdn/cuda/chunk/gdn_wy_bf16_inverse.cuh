#pragma once
#include "gdn_wy_native.cuh"
#include "gdn_wy_register_mma.cuh"

namespace gdn_wy_bf16_inverse {
namespace mma = gdn_wy_register_mma;
using Accumulator = mma::fragment<mma::accumulator, 16, 16, 16, float>;
using Left = mma::fragment<mma::matrix_a, 16, 16, 16, __nv_bfloat16, mma::row_major>;
using Right = mma::fragment<mma::matrix_b, 16, 16, 16, __nv_bfloat16, mma::row_major>;
constexpr int Stride = gdn_wy::TokenStride;

__device__ inline Left negative_operand(const Accumulator &value) {
  Left result;
  auto *words = reinterpret_cast<unsigned *>(result.x);
#pragma unroll
  for (int pair = 0; pair < 4; ++pair)
    words[pair] = gdn_wy::native_pair(-value.x[pair * 2], -value.x[pair * 2 + 1]);
  return result;
}

__device__ inline void store_tile(__nv_bfloat16 *destination, const Accumulator &value) {
  int lane = threadIdx.x % 32;
#pragma unroll
  for (int half = 0; half < 2; ++half)
#pragma unroll
    for (int pair = 0; pair < 2; ++pair) {
      int element = half * 4 + pair * 2;
      int row = lane / 4 + pair * 8, column = half * 8 + lane % 4 * 2;
      *reinterpret_cast<unsigned *>(destination + row * Stride + column) =
          gdn_wy::native_pair(value.x[element], value.x[element + 1]);
    }
}

template <bool Transpose> __device__ inline unsigned load_block8(const __nv_bfloat16 *source) {
  unsigned address = mma::shared_address(source + threadIdx.x % 8 * Stride), value;
  if constexpr (Transpose)
    asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];"
                 : "=r"(value)
                 : "r"(address));
  else
    asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];"
                 : "=r"(value)
                 : "r"(address));
  return value;
}

__device__ inline void multiply_block8(unsigned a0, unsigned a1, unsigned b, float (&result)[4]) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 "
               "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%7,%7,%7};"
               : "=f"(result[0]), "=f"(result[1]), "=f"(result[2]), "=f"(result[3])
               : "r"(a0), "r"(a1), "r"(b), "f"(0.f));
}

template <class Storage> __device__ inline void merge_block8(Storage &s, int base) {
  unsigned diagonal = load_block8<false>(s.low.inverse + (base + 8) * Stride + base + 8);
  unsigned raw = load_block8<true>(s.low.raw + (base + 8) * Stride + base);
  float product[4];
  multiply_block8(diagonal, diagonal, raw, product);
  unsigned a0 = gdn_wy::native_pair(-product[0], -product[1]);
  unsigned a1 = gdn_wy::native_pair(-product[2], -product[3]);
  unsigned upper = load_block8<true>(s.low.inverse + base * Stride + base);
  multiply_block8(a0, a1, upper, product);
  int lane = threadIdx.x % 32, row = lane / 4, column = lane % 4 * 2;
  *reinterpret_cast<unsigned *>(s.low.inverse + (base + 8 + row) * Stride + base + column) =
      gdn_wy::native_pair(product[0], product[1]);
}

template <class Storage> __device__ inline void inverse_blocks(Storage &s) {
  for (int index = threadIdx.x; index < 64 * 64; index += 256) {
    int row = index / 64, column = index % 64;
    s.low.inverse[row * Stride + column] = __float2bfloat16_rn(row == column ? 1.f : 0.f);
    s.low.raw[row * Stride + column] =
        __float2bfloat16_rn(row > column ? s.products[row * gdn_wy::ProductStride + column] : 0.f);
  }
  __syncthreads();
  if (threadIdx.x < 64) {
    int base = threadIdx.x / 8 * 8, row = threadIdx.x % 8;
    float solved[8];
#pragma unroll
    for (int column = 0; column < 8; ++column)
      solved[column] =
          row == column ? 1.f : __bfloat162float(s.low.raw[(base + row) * Stride + base + column]);
#pragma unroll
    for (int pivot = 0; pivot < 7; ++pivot) {
      float scale = -solved[pivot];
#pragma unroll
      for (int column = 0; column < pivot; ++column) {
        float source = __shfl_sync(0xffffffffu, solved[column], pivot, 8);
        if (row > pivot)
          solved[column] = fmaf(scale, source, solved[column]);
      }
      if (row > pivot)
        solved[pivot] = scale;
    }
#pragma unroll
    for (int column = 0; column < 8; ++column)
      s.low.inverse[(base + row) * Stride + base + column] = __float2bfloat16_rn(solved[column]);
  }
  __syncthreads();
  int warp = threadIdx.x / 32;
  if (warp < 4)
    merge_block8(s, warp * 16);
  __syncthreads();
  if (warp < 2) {
    int base = warp * 32;
    Left diagonal;
    Right raw;
    mma::load_matrix_sync(diagonal, s.low.inverse + (base + 16) * Stride + base + 16, Stride);
    mma::load_matrix_sync(raw, s.low.raw + (base + 16) * Stride + base, Stride);
    Accumulator product;
    mma::fill_fragment(product, 0.f);
    mma::mma_sync(product, diagonal, raw, product);
    Left intermediate = negative_operand(product);
    Right upper;
    mma::load_matrix_sync(upper, s.low.inverse + base * Stride + base, Stride);
    Accumulator result;
    mma::fill_fragment(result, 0.f);
    mma::mma_sync(result, intermediate, upper, result);
    store_tile(s.low.inverse + (base + 16) * Stride + base, result);
  }
  __syncthreads();
  if (warp < 2) {
    Accumulator products[2];
#pragma unroll
    for (int column = 0; column < 2; ++column)
      mma::fill_fragment(products[column], 0.f);
#pragma unroll
    for (int inner = 0; inner < 2; ++inner) {
      Left diagonal;
      mma::load_matrix_sync(diagonal, s.low.inverse + (32 + warp * 16) * Stride + 32 + inner * 16,
                            Stride);
#pragma unroll
      for (int column = 0; column < 2; ++column) {
        Right raw;
        mma::load_matrix_sync(raw, s.low.raw + (32 + inner * 16) * Stride + column * 16, Stride);
        mma::mma_sync(products[column], diagonal, raw, products[column]);
      }
    }
    Left intermediate[2];
#pragma unroll
    for (int inner = 0; inner < 2; ++inner)
      intermediate[inner] = negative_operand(products[inner]);
    Accumulator results[2];
#pragma unroll
    for (int column = 0; column < 2; ++column)
      mma::fill_fragment(results[column], 0.f);
#pragma unroll
    for (int inner = 0; inner < 2; ++inner)
#pragma unroll
      for (int column = 0; column < 2; ++column) {
        Right upper;
        mma::load_matrix_sync(upper, s.low.inverse + inner * 16 * Stride + column * 16, Stride);
        mma::mma_sync(results[column], intermediate[inner], upper, results[column]);
      }
#pragma unroll
    for (int column = 0; column < 2; ++column)
      store_tile(s.low.inverse + (32 + warp * 16) * Stride + column * 16, results[column]);
  }
}
} // namespace gdn_wy_bf16_inverse
