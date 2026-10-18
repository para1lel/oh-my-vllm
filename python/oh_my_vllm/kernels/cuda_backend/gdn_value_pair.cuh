#pragma once

#include "gdn_kernels.cuh"

__device__ __forceinline__ uint64_t gdn_pair_bits(uint32_t low, uint32_t high) {
  uint64_t value;
  asm("mov.b64 %0, {%1,%2};" : "=l"(value) : "r"(low), "r"(high));
  return value;
}

__device__ __forceinline__ uint32_t gdn_pair_word(uint64_t value, int unit) {
  uint32_t low, high;
  asm("mov.b64 {%0,%1}, %2;" : "=r"(low), "=r"(high) : "l"(value));
  return unit ? high : low;
}

__device__ __forceinline__ uint64_t gdn_pair_from_bf16x2(uint32_t value) {
  return gdn_pair_bits(value << 16, value & 0xffff0000u);
}

__device__ __forceinline__ uint32_t gdn_pair_to_bf16x2(uint64_t value) {
  float low = __uint_as_float(gdn_pair_word(value, 0));
  float high = __uint_as_float(gdn_pair_word(value, 1));
  uint32_t packed;
  asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(packed) : "f"(low), "f"(high));
  return packed;
}

__device__ __forceinline__ uint64_t gdn_pair_splat(float value) {
  uint32_t bits = __float_as_uint(value);
  return gdn_pair_bits(bits, bits);
}

__device__ __forceinline__ uint64_t gdn_pair_add(uint64_t a, uint64_t b) {
  uint64_t result;
  asm("add.rn.f32x2 %0,%1,%2;" : "=l"(result) : "l"(a), "l"(b));
  return result;
}

__device__ __forceinline__ uint64_t gdn_pair_mul(uint64_t a, uint64_t b) {
  uint64_t result;
  asm("mul.rn.f32x2 %0,%1,%2;" : "=l"(result) : "l"(a), "l"(b));
  return result;
}

__device__ __forceinline__ uint64_t gdn_pair_fma(uint64_t a, uint64_t b, uint64_t c) {
  uint64_t result;
  asm("fma.rn.f32x2 %0,%1,%2,%3;" : "=l"(result) : "l"(a), "l"(b), "l"(c));
  return result;
}

__device__ __forceinline__ uint64_t gdn_pair_half_sum(uint64_t value) {
#pragma unroll
  for (int distance = 8; distance > 0; distance /= 2) {
    uint32_t low = __shfl_xor_sync(0xffffffffu, gdn_pair_word(value, 0), distance, 16);
    uint32_t high = __shfl_xor_sync(0xffffffffu, gdn_pair_word(value, 1), distance, 16);
    value = gdn_pair_add(value, gdn_pair_bits(low, high));
  }
  return value;
}

template <int ValueTile = 16, int Threads = 128>
__global__ void __launch_bounds__(Threads, 1)
    gdn_value_pair_kernel(const __nv_bfloat16 *qkv, __nv_bfloat16 *pool, const int *starts,
                          const int *reads, const int *writes, __nv_bfloat16 *recurrent,
                          const float *gates, int rows, int slots) {
  static_assert(ValueTile == 16 && Threads == 128 && ValueTile / (Threads / 32) == 4);
  static_assert(GDNPrefetchRows == 16 && GDNStageQ && !GDNStageK);
  static_assert(GDNPrefetchSlotBytes<ValueTile> == 13008);
  int sequence = blockIdx.x, head = blockIdx.y, lane = threadIdx.x & 31;
  int warp = threadIdx.x / 32, half = lane / 16;
  int tile_value_base = blockIdx.z * ValueTile;
  int value_base = tile_value_base + warp * 4 + half * 2;
  int first = starts[sequence], last = starts[sequence + 1], source = reads[sequence];
  gdn_require(first >= 0 && first <= last && last <= rows && source >= 0 && source < slots);
  uint64_t state[2][4];
#pragma unroll
  for (int group = 0; group < 2; ++group)
#pragma unroll
    for (int key = 0; key < 4; ++key) {
      int column = (lane % 16) * 4 + group * 64 + key;
      uint32_t low = __float_as_uint(__bfloat162float(
          pool[((int64_t(source) * 48 + head) * 128 + value_base) * 128 + column]));
      uint32_t high = __float_as_uint(__bfloat162float(
          pool[((int64_t(source) * 48 + head) * 128 + value_base + 1) * 128 + column]));
      state[group][key] = gdn_pair_bits(low, high);
    }
  extern __shared__ __align__(16) unsigned char staged[];
  unsigned checkpoint_mask = 0;
  if (first < last) {
    gdn_prefetch_rows<ValueTile>(staged, qkv, gates, writes, first, last, head, tile_value_base,
                                 source, slots);
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    checkpoint_mask =
        *reinterpret_cast<const unsigned *>(staged + GDNPrefetchSlotBytes<ValueTile> - 16);
    if (first + GDNPrefetchRows < last)
      gdn_prefetch_rows<ValueTile>(staged + GDNPrefetchSlotBytes<ValueTile>, qkv, gates, writes,
                                   first + GDNPrefetchRows, last, head, tile_value_base, source,
                                   slots);
  }
  for (int row = first; row < last; ++row) {
    int local_row = (row - first) % GDNPrefetchRows;
    int slot = ((row - first) / GDNPrefetchRows) & 1;
    unsigned char *storage = staged + slot * GDNPrefetchSlotBytes<ValueTile>;
    int target = -1;
    if (checkpoint_mask & (1u << local_row))
      target = reinterpret_cast<const int *>(
          storage + GDNPrefetchRows * (GDNStageQKBytes + 8 + ValueTile * 2))[local_row];
    float q[2][4], k[2][4];
#pragma unroll
    for (int group = 0; group < 2; ++group) {
      int vector = local_row * 32 + lane % 16 + group * 16;
      gdn_stage_load<GDNStageQ>(storage, vector, q[group]);
      gdn_stage_load<GDNStageK>(storage + GDNPrefetchRows * GDNStageQueryBytes, vector, k[group]);
    }
    auto *coefficients = reinterpret_cast<const float2 *>(
        storage + GDNPrefetchRows * (GDNStageQKBytes + ValueTile * 2));
    float2 coefficient = coefficients[local_row];
    uint64_t decay = gdn_pair_splat(coefficient.y), beta = gdn_pair_splat(coefficient.x);
    uint64_t recalled[2] = {};
#pragma unroll
    for (int group = 0; group < 2; ++group)
#pragma unroll
      for (int key = 0; key < 4; ++key) {
        state[group][key] = gdn_pair_mul(state[group][key], decay);
        recalled[group] =
            gdn_pair_fma(state[group][key], gdn_pair_splat(k[group][key]), recalled[group]);
      }
    uint64_t recall = gdn_pair_half_sum(gdn_pair_add(recalled[0], recalled[1]));
    auto *values = reinterpret_cast<const uint32_t *>(storage + GDNPrefetchRows * GDNStageQKBytes);
    uint32_t packed = values[local_row * (ValueTile / 2) + warp * 2 + half];
    uint64_t input = gdn_pair_from_bf16x2(packed);
    uint64_t delta = gdn_pair_mul(beta, gdn_pair_add(input, recall ^ 0x8000000080000000ull));
    uint64_t sums[2] = {};
#pragma unroll
    for (int group = 0; group < 2; ++group)
#pragma unroll
      for (int key = 0; key < 4; ++key) {
        state[group][key] = gdn_pair_fma(delta, gdn_pair_splat(k[group][key]), state[group][key]);
        sums[group] = gdn_pair_fma(state[group][key], gdn_pair_splat(q[group][key]), sums[group]);
      }
    if (target >= 0) {
#pragma unroll
      for (int group = 0; group < 2; ++group)
#pragma unroll
        for (int key = 0; key < 4; ++key)
#pragma unroll
          for (int unit = 0; unit < 2; ++unit)
            pool[((int64_t(target) * 48 + head) * 128 + value_base + unit) * 128 + (lane % 16) * 4 +
                 group * 64 + key] =
                __float2bfloat16_rn(__uint_as_float(gdn_pair_word(state[group][key], unit)));
    }
    uint64_t sum = gdn_pair_half_sum(gdn_pair_add(sums[0], sums[1]));
    if (lane % 16 == 0)
      *reinterpret_cast<uint32_t *>(recurrent + (int64_t(row) * 48 + head) * 128 + value_base) =
          gdn_pair_to_bf16x2(sum);
    if (local_row + 1 == GDNPrefetchRows && row + 1 < last) {
      asm volatile("cp.async.wait_group 0;" ::: "memory");
      __syncthreads();
      checkpoint_mask = *reinterpret_cast<const unsigned *>(
          staged + (slot ^ 1) * GDNPrefetchSlotBytes<ValueTile> + GDNPrefetchSlotBytes<ValueTile> -
          16);
      if (row + 1 + GDNPrefetchRows < last)
        gdn_prefetch_rows<ValueTile>(storage, qkv, gates, writes, row + 1 + GDNPrefetchRows, last,
                                     head, tile_value_base, source, slots);
    }
  }
}

__device__ __forceinline__ void gdn_pair_two_half_sum(uint64_t (&value)[2]) {
#pragma unroll
  for (int distance = 8; distance > 0; distance /= 2) {
    uint32_t low0 = __shfl_xor_sync(0xffffffffu, gdn_pair_word(value[0], 0), distance, 16);
    uint32_t high0 = __shfl_xor_sync(0xffffffffu, gdn_pair_word(value[0], 1), distance, 16);
    uint32_t low1 = __shfl_xor_sync(0xffffffffu, gdn_pair_word(value[1], 0), distance, 16);
    uint32_t high1 = __shfl_xor_sync(0xffffffffu, gdn_pair_word(value[1], 1), distance, 16);
    value[0] = gdn_pair_add(value[0], gdn_pair_bits(low0, high0));
    value[1] = gdn_pair_add(value[1], gdn_pair_bits(low1, high1));
  }
}

template <int ValueTile = 32, int Threads = 128>
__global__ void __launch_bounds__(Threads, 1)
    gdn_value_two_pair_kernel(const __nv_bfloat16 *qkv, __nv_bfloat16 *pool, const int *starts,
                              const int *reads, const int *writes, __nv_bfloat16 *recurrent,
                              const float *gates, int rows, int slots) {
  static_assert(ValueTile == 32 && Threads == 128 && ValueTile / (Threads / 32) == 8);
  static_assert(GDNPrefetchRows == 16 && GDNStageQ && !GDNStageK);
  static_assert(GDNPrefetchSlotBytes<ValueTile> == 13520);
  int sequence = blockIdx.x, head = blockIdx.y, lane = threadIdx.x & 31;
  int warp = threadIdx.x / 32, half = lane / 16;
  int tile_value_base = blockIdx.z * ValueTile;
  int value_base = tile_value_base + warp * 8 + half * 4;
  int first = starts[sequence], last = starts[sequence + 1], source = reads[sequence];
  gdn_require(first >= 0 && first <= last && last <= rows && source >= 0 && source < slots);
  uint64_t state[2][2][4];
#pragma unroll
  for (int group = 0; group < 2; ++group)
#pragma unroll
    for (int key = 0; key < 4; ++key) {
      int column = (lane % 16) * 4 + group * 64 + key;
#pragma unroll
      for (int pair = 0; pair < 2; ++pair) {
        uint32_t low = __float_as_uint(__bfloat162float(
            pool[((int64_t(source) * 48 + head) * 128 + value_base + pair * 2) * 128 + column]));
        uint32_t high = __float_as_uint(__bfloat162float(
            pool[((int64_t(source) * 48 + head) * 128 + value_base + pair * 2 + 1) * 128 +
                 column]));
        state[pair][group][key] = gdn_pair_bits(low, high);
      }
    }
  extern __shared__ __align__(16) unsigned char staged[];
  unsigned checkpoint_mask = 0;
  if (first < last) {
    gdn_prefetch_rows<ValueTile>(staged, qkv, gates, writes, first, last, head, tile_value_base,
                                 source, slots);
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    checkpoint_mask =
        *reinterpret_cast<const unsigned *>(staged + GDNPrefetchSlotBytes<ValueTile> - 16);
    if (first + GDNPrefetchRows < last)
      gdn_prefetch_rows<ValueTile>(staged + GDNPrefetchSlotBytes<ValueTile>, qkv, gates, writes,
                                   first + GDNPrefetchRows, last, head, tile_value_base, source,
                                   slots);
  }
  for (int row = first; row < last; ++row) {
    int local_row = (row - first) % GDNPrefetchRows;
    int slot = ((row - first) / GDNPrefetchRows) & 1;
    unsigned char *storage = staged + slot * GDNPrefetchSlotBytes<ValueTile>;
    int target = -1;
    if (checkpoint_mask & (1u << local_row))
      target = reinterpret_cast<const int *>(
          storage + GDNPrefetchRows * (GDNStageQKBytes + 8 + ValueTile * 2))[local_row];
    float q[2][4], k[2][4];
#pragma unroll
    for (int group = 0; group < 2; ++group) {
      int vector = local_row * 32 + lane % 16 + group * 16;
      gdn_stage_load<GDNStageQ>(storage, vector, q[group]);
      gdn_stage_load<GDNStageK>(storage + GDNPrefetchRows * GDNStageQueryBytes, vector, k[group]);
    }
    auto *coefficients = reinterpret_cast<const float2 *>(
        storage + GDNPrefetchRows * (GDNStageQKBytes + ValueTile * 2));
    float2 coefficient = coefficients[local_row];
    uint64_t decay = gdn_pair_splat(coefficient.y), beta = gdn_pair_splat(coefficient.x);
    uint64_t recalled[2][2] = {};
#pragma unroll
    for (int group = 0; group < 2; ++group)
#pragma unroll
      for (int key = 0; key < 4; ++key) {
        uint64_t key_pair = gdn_pair_splat(k[group][key]);
#pragma unroll
        for (int pair = 0; pair < 2; ++pair) {
          state[pair][group][key] = gdn_pair_mul(state[pair][group][key], decay);
          recalled[pair][group] =
              gdn_pair_fma(state[pair][group][key], key_pair, recalled[pair][group]);
        }
      }
    uint64_t recall[2];
#pragma unroll
    for (int pair = 0; pair < 2; ++pair)
      recall[pair] = gdn_pair_add(recalled[pair][0], recalled[pair][1]);
    gdn_pair_two_half_sum(recall);
    auto *values = reinterpret_cast<const uint2 *>(storage + GDNPrefetchRows * GDNStageQKBytes);
    uint2 packed = values[local_row * (ValueTile / 4) + warp * 2 + half];
    uint32_t packed_values[2] = {packed.x, packed.y};
    uint64_t delta[2];
#pragma unroll
    for (int pair = 0; pair < 2; ++pair) {
      uint64_t input = gdn_pair_from_bf16x2(packed_values[pair]);
      delta[pair] = gdn_pair_mul(beta, gdn_pair_add(input, recall[pair] ^ 0x8000000080000000ull));
    }
    uint64_t sums[2][2] = {};
#pragma unroll
    for (int group = 0; group < 2; ++group)
#pragma unroll
      for (int key = 0; key < 4; ++key) {
        uint64_t key_pair = gdn_pair_splat(k[group][key]);
        uint64_t query_pair = gdn_pair_splat(q[group][key]);
#pragma unroll
        for (int pair = 0; pair < 2; ++pair) {
          state[pair][group][key] = gdn_pair_fma(delta[pair], key_pair, state[pair][group][key]);
          sums[pair][group] = gdn_pair_fma(state[pair][group][key], query_pair, sums[pair][group]);
        }
      }
    if (target >= 0) {
#pragma unroll
      for (int group = 0; group < 2; ++group)
#pragma unroll
        for (int key = 0; key < 4; ++key)
#pragma unroll
          for (int pair = 0; pair < 2; ++pair)
#pragma unroll
            for (int unit = 0; unit < 2; ++unit)
              pool[((int64_t(target) * 48 + head) * 128 + value_base + pair * 2 + unit) * 128 +
                   (lane % 16) * 4 + group * 64 + key] =
                  __float2bfloat16_rn(
                      __uint_as_float(gdn_pair_word(state[pair][group][key], unit)));
    }
    uint64_t sum[2];
#pragma unroll
    for (int pair = 0; pair < 2; ++pair)
      sum[pair] = gdn_pair_add(sums[pair][0], sums[pair][1]);
    gdn_pair_two_half_sum(sum);
    if (lane % 16 == 0)
#pragma unroll
      for (int pair = 0; pair < 2; ++pair)
        *reinterpret_cast<uint32_t *>(recurrent + (int64_t(row) * 48 + head) * 128 + value_base +
                                      pair * 2) = gdn_pair_to_bf16x2(sum[pair]);
    if (local_row + 1 == GDNPrefetchRows && row + 1 < last) {
      asm volatile("cp.async.wait_group 0;" ::: "memory");
      __syncthreads();
      checkpoint_mask = *reinterpret_cast<const unsigned *>(
          staged + (slot ^ 1) * GDNPrefetchSlotBytes<ValueTile> + GDNPrefetchSlotBytes<ValueTile> -
          16);
      if (row + 1 + GDNPrefetchRows < last)
        gdn_prefetch_rows<ValueTile>(storage, qkv, gates, writes, row + 1 + GDNPrefetchRows, last,
                                     head, tile_value_base, source, slots);
    }
  }
}

inline bool gdn_prefill_two_pairs(int rows, int max_queries, int device) {
  static thread_local int cached_device = -1, multiprocessors = 0;
  if (cached_device != device) {
    check_cuda(cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, device));
    cached_device = device;
  }
  constexpr int BlocksPerSequence = 48 * (128 / 32);
  return int64_t(rows) * BlocksPerSequence >= int64_t(2) * multiprocessors * max_queries;
}
