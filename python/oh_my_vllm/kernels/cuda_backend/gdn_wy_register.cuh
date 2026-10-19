#pragma once
#include "gdn_wy_native.cuh"
#include "gdn_wy_register_mma.cuh"

namespace gdn_wy_register {
constexpr int Rows = 64, Threads = 128, UStride = 72;
using namespace gdn_wy;
using Accumulator =
    gdn_wy_register_mma::fragment<gdn_wy_register_mma::accumulator, 16, 16, 16, float>;
using StateOperand = gdn_wy_register_mma::fragment<gdn_wy_register_mma::matrix_a, 16, 16, 16,
                                                   __nv_bfloat16, gdn_wy_register_mma::row_major>;
using WeightOperand = gdn_wy_register_mma::fragment<gdn_wy_register_mma::matrix_b, 16, 16, 16,
                                                    __nv_bfloat16, gdn_wy_register_mma::col_major>;
using KeyOperand = gdn_wy_register_mma::fragment<gdn_wy_register_mma::matrix_b, 16, 16, 16,
                                                 __nv_bfloat16, gdn_wy_register_mma::row_major>;

struct alignas(128) RegisterCarryStorage {
  __nv_bfloat16 weights[2][64 * 136], keys[2][64 * 136], updates[2][64 * UStride];
  __nv_bfloat16 snapshot[Rows * 136];
  alignas(16) float prefix[2][128];
};

__device__ inline StateOperand state_operand(const Accumulator &c) {
  StateOperand a;
  auto *packed = reinterpret_cast<unsigned *>(a.x);
  // The accumulator and matrix-A fragments use the same paired coordinates.
#pragma unroll
  for (int pair = 0; pair < 4; ++pair)
    packed[pair] = native_pair(c.x[pair * 2], c.x[pair * 2 + 1]);
  return a;
}

__device__ inline void prefetch_chunk(RegisterCarryStorage &s, const __nv_bfloat16 *qkv,
                                      const int *metadata, const __nv_bfloat16 *transforms,
                                      const float *prefixes, int chunk, int head, int value,
                                      int buffer) {
  int first = metadata[chunk * 4], count = metadata[chunk * 4 + 1] - first;
  for (int i = threadIdx.x; i < 32; i += Threads)
    native_copy16(s.prefix[buffer] + i * 4, prefixes + (int64_t(chunk) * 48 + head) * 128 + i * 4);
  for (int i = threadIdx.x; i < 64 * 16; i += Threads) {
    int token = i / 16, key = i % 16 * 8;
    native_copy16(s.weights[buffer] + token * 136 + key,
                  transforms + transform_offset(chunk, head, 0, 0) + i * 8);
    native_copy16(s.keys[buffer] + token * 136 + key,
                  qkv + int64_t(first + (token < count ? token : 0)) * 10240 + 2048 +
                      head / 3 * 128 + key,
                  token < count);
  }
  for (int i = threadIdx.x; i < 64 * (Rows / 8); i += Threads) {
    int token = i / (Rows / 8), column = i % (Rows / 8) * 8;
    native_copy16(s.updates[buffer] + token * UStride + column,
                  transforms + transform_offset(chunk, head, 0, 1) + token * 128 + value + column);
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
}

__device__ inline void store_state_tile(__nv_bfloat16 *destination, int stride,
                                        const Accumulator &value) {
  int lane = threadIdx.x % 32;
#pragma unroll
  for (int half = 0; half < 2; ++half)
#pragma unroll
    for (int pair = 0; pair < 2; ++pair) {
      int index = half * 4 + pair * 2;
      unsigned bits = native_pair(value.x[index], value.x[index + 1]);
      int row = lane / 4 + pair * 8, col = half * 8 + lane % 4 * 2;
      *reinterpret_cast<unsigned *>(destination + row * stride + col) = bits;
    }
}

__device__ inline void load_state_tile(Accumulator &value, const __nv_bfloat16 *source,
                                       int stride) {
  int lane = threadIdx.x % 32;
#pragma unroll
  for (int half = 0; half < 2; ++half)
#pragma unroll
    for (int pair = 0; pair < 2; ++pair) {
      int index = half * 4 + pair * 2;
      int row = lane / 4 + pair * 8, col = half * 8 + lane % 4 * 2;
      unsigned bits = *reinterpret_cast<const unsigned *>(source + row * stride + col);
      value.x[index] = __uint_as_float(bits << 16);
      value.x[index + 1] = __uint_as_float(bits & 0xffff0000u);
    }
}

template <bool EarlyPrefetch>
__global__ __launch_bounds__(Threads, 2) void register_carry_kernel(
    const __nv_bfloat16 *qkv, const int *metadata, const int *reads, __nv_bfloat16 *transforms,
    const float *prefixes, __nv_bfloat16 *pool, __nv_bfloat16 *snapshots, int capacity) {
  int sequence = blockIdx.x, head = blockIdx.y, first_chunk = sequence * capacity;
  int chunks = metadata[first_chunk * 4 + 3];
  if (chunks <= 0)
    return;
  int value_base = blockIdx.z * Rows, warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  int source = reads[sequence];
  extern __shared__ __align__(128) unsigned char memory[];
  auto &s = *reinterpret_cast<RegisterCarryStorage *>(memory);
  Accumulator state[8];
#pragma unroll
  for (int key = 0; key < 8; ++key)
    load_state_tile(state[key],
                    pool + ((int64_t(source) * 48 + head) * 128 + value_base + warp * 16) * 128 +
                        key * 16,
                    128);
  prefetch_chunk(s, qkv, metadata, transforms, prefixes, first_chunk, head, value_base, 0);
  for (int local = 0; local < chunks; ++local) {
    int chunk = first_chunk + local, buffer = local % 2;
    int count = metadata[chunk * 4 + 1] - metadata[chunk * 4];
#pragma unroll
    for (int key = 0; key < 8; ++key)
      store_state_tile(s.snapshot + warp * 16 * 136 + key * 16, 136, state[key]);
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    for (int i = threadIdx.x; i < Rows * 16; i += Threads) {
      int v = i / 16, keys = i % 16 * 8;
      *reinterpret_cast<uint4 *>(snapshots + (int64_t(chunk) * 48 + head) * 128 * 128 +
                                 (value_base + v) * 128 + keys) =
          *reinterpret_cast<const uint4 *>(s.snapshot + v * 136 + keys);
    }
    if constexpr (EarlyPrefetch) {
      if (local + 1 < chunks)
        prefetch_chunk(s, qkv, metadata, transforms, prefixes, chunk + 1, head, value_base,
                       1 - buffer);
    }
    Accumulator projection[4];
#pragma unroll
    for (int token = 0; token < 4; ++token)
      gdn_wy_register_mma::fill_fragment(projection[token], 0.f);
#pragma unroll
    for (int key = 0; key < 8; ++key) {
      StateOperand a = state_operand(state[key]);
#pragma unroll
      for (int token = 0; token < 4; ++token) {
        WeightOperand b;
        gdn_wy_register_mma::load_matrix_sync(b, s.weights[buffer] + token * 16 * 136 + key * 16,
                                              136);
        gdn_wy_register_mma::mma_sync(projection[token], a, b, projection[token]);
      }
    }
    if constexpr (!EarlyPrefetch) {
      if (local + 1 < chunks)
        prefetch_chunk(s, qkv, metadata, transforms, prefixes, chunk + 1, head, value_base,
                       1 - buffer);
    }
    Accumulator updates[8];
#pragma unroll
    for (int key = 0; key < 8; ++key)
      gdn_wy_register_mma::fill_fragment(updates[key], 0.f);
#pragma unroll
    for (int token = 0; token < 4; ++token) {
      Accumulator scaled;
#pragma unroll
      for (int half = 0; half < 2; ++half)
#pragma unroll
        for (int pair = 0; pair < 2; ++pair) {
          int element = half * 4 + pair * 2;
          int v = warp * 16 + lane / 4 + pair * 8;
          int token_row = token * 16 + half * 8 + lane % 4 * 2;
          float2 change = __fadd2_rn(
              make_float2(float(s.updates[buffer][token_row * UStride + v]),
                          float(s.updates[buffer][(token_row + 1) * UStride + v])),
              make_float2(-projection[token].x[element], -projection[token].x[element + 1]));
          s.updates[buffer][token_row * UStride + v] = __float2bfloat16_rn(change.x);
          s.updates[buffer][(token_row + 1) * UStride + v] = __float2bfloat16_rn(change.y);
          float2 result = __fmul2_rn(change, make_float2(s.prefix[buffer][64 + token_row],
                                                         s.prefix[buffer][65 + token_row]));
          scaled.x[element] = result.x;
          scaled.x[element + 1] = result.y;
        }
      StateOperand change_operand = state_operand(scaled);
#pragma unroll
      for (int key = 0; key < 8; ++key) {
        KeyOperand b;
        gdn_wy_register_mma::load_matrix_sync(b, s.keys[buffer] + token * 16 * 136 + key * 16, 136);
        gdn_wy_register_mma::mma_sync(updates[key], change_operand, b, updates[key]);
      }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < 64 * (Rows / 8); i += Threads) {
      int token = i / (Rows / 8), v = i % (Rows / 8) * 8;
      *reinterpret_cast<uint4 *>(transforms + transform_offset(chunk, head, 0, 1) + token * 128 +
                                 value_base + v) =
          *reinterpret_cast<const uint4 *>(s.updates[buffer] + token * UStride + v);
    }
#pragma unroll
    for (int key = 0; key < 8; ++key)
#pragma unroll
      for (int element = 0; element < 8; element += 2) {
        float2 result;
        cute::SM100_1x2x1_F32F32F32F32::fma(
            result, s.prefix[buffer][count - 1],
            make_float2(state[key].x[element], state[key].x[element + 1]),
            make_float2(updates[key].x[element], updates[key].x[element + 1]));
        state[key].x[element] = result.x;
        state[key].x[element + 1] = result.y;
      }
    int target = metadata[chunk * 4 + 2];
    if (target >= 0) {
#pragma unroll
      for (int key = 0; key < 8; ++key)
        store_state_tile(s.snapshot + warp * 16 * 136 + key * 16, 136, state[key]);
      __syncthreads();
      for (int i = threadIdx.x; i < Rows * 16; i += Threads) {
        int v = i / 16, keys = i % 16 * 8;
        *reinterpret_cast<uint4 *>(pool + (int64_t(target) * 48 + head) * 128 * 128 +
                                   (value_base + v) * 128 + keys) =
            *reinterpret_cast<const uint4 *>(s.snapshot + v * 136 + keys);
      }
    }
    __syncthreads();
  }
}

} // namespace gdn_wy_register
