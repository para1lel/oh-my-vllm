#pragma once

#include "common.cuh"
#include <type_traits>

namespace gdn_recurrent {

__device__ inline void require(bool condition) {
  if (!condition)
    asm volatile("trap;");
}

__device__ inline void load_four(const __nv_bfloat16 *source, float (&value)[4]) {
  uint2 packed = *reinterpret_cast<const uint2 *>(source);
  value[0] = __uint_as_float(packed.x << 16);
  value[1] = __uint_as_float(packed.x & 0xffff0000u);
  value[2] = __uint_as_float(packed.y << 16);
  value[3] = __uint_as_float(packed.y & 0xffff0000u);
}

__device__ inline uint64_t pair_bits(uint32_t low, uint32_t high) {
  uint64_t result;
  asm("mov.b64 %0, {%1,%2};" : "=l"(result) : "r"(low), "r"(high));
  return result;
}

__device__ inline float pair_low(uint64_t value) {
  uint32_t low, high;
  asm("mov.b64 {%0,%1}, %2;" : "=r"(low), "=r"(high) : "l"(value));
  return __uint_as_float(low);
}

__device__ inline float pair_high(uint64_t value) {
  uint32_t low, high;
  asm("mov.b64 {%0,%1}, %2;" : "=r"(low), "=r"(high) : "l"(value));
  return __uint_as_float(high);
}

__device__ inline uint64_t pair(float low, float high) {
  return pair_bits(__float_as_uint(low), __float_as_uint(high));
}

__device__ inline uint64_t from_bf16(uint32_t bits) {
  return pair_bits(bits << 16, bits & 0xffff0000u);
}

__device__ inline uint32_t to_bf16(uint64_t value) {
  float low = pair_low(value), high = pair_high(value);
  uint32_t bits;
  asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(bits) : "f"(low), "f"(high));
  return bits;
}

__device__ inline uint64_t multiply(uint64_t left, uint64_t right) {
  uint64_t result;
  asm("mul.rn.f32x2 %0,%1,%2;" : "=l"(result) : "l"(left), "l"(right));
  return result;
}

__device__ inline uint64_t fma(uint64_t left, uint64_t right, uint64_t addend) {
  uint64_t result;
  asm("fma.rn.f32x2 %0,%1,%2,%3;" : "=l"(result) : "l"(left), "l"(right), "l"(addend));
  return result;
}

__device__ inline void load_eight(const __nv_bfloat16 *source, float (&values)[8]) {
  uint4 packed = *reinterpret_cast<const uint4 *>(source);
  uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    values[i * 2] = __uint_as_float(words[i] << 16);
    values[i * 2 + 1] = __uint_as_float(words[i] & 0xffff0000u);
  }
}

__device__ inline void store_eight(__nv_bfloat16 *destination, uint64_t (&values)[8], bool high) {
  uint32_t packed[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float low = high ? pair_high(values[i * 2]) : pair_low(values[i * 2]);
    float upper = high ? pair_high(values[i * 2 + 1]) : pair_low(values[i * 2 + 1]);
    asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(packed[i]) : "f"(low), "f"(upper));
  }
  *reinterpret_cast<uint4 *>(destination) = make_uint4(packed[0], packed[1], packed[2], packed[3]);
}

__device__ inline float half_sum(float value) {
#pragma unroll
  for (int distance = 8; distance; distance /= 2)
    value += __shfl_xor_sync(0xffffffffu, value, distance, 16);
  return value;
}

__device__ inline uint64_t add(uint64_t a, uint64_t b) {
  uint64_t result;
  asm("add.rn.f32x2 %0,%1,%2;" : "=l"(result) : "l"(a), "l"(b));
  return result;
}

__device__ inline uint64_t half_sum(uint64_t value) {
#pragma unroll
  for (int distance = 8; distance; distance /= 2) {
    float low = __shfl_xor_sync(0xffffffffu, pair_low(value), distance, 16);
    float high = __shfl_xor_sync(0xffffffffu, pair_high(value), distance, 16);
    value = add(value, pair(low, high));
  }
  return value;
}

__device__ inline float log_one_plus(float value) {
  if (value < 0.0625f) {
    float tail = -1.f / 6.f;
    tail = fmaf(tail, value, 1.f / 5.f);
    tail = fmaf(tail, value, -1.f / 4.f);
    tail = fmaf(tail, value, 1.f / 3.f);
    tail = fmaf(tail, value, -1.f / 2.f);
    return fmaf(value * value, tail, value);
  }
  return __logf(1.f + value);
}

__device__ inline float exponential(float value) {
  return value >= -80.f && value <= 80.f ? __expf(value) : expf(value);
}

__device__ inline float sigmoid(float value) {
  float exponent = exponential(-fabsf(value));
  float numerator = value >= 0.f ? 1.f : exponent;
  return exponent >= 1e-35f ? __fdividef(numerator, 1.f + exponent) : numerator / (1.f + exponent);
}

__device__ __noinline__ float2 cold_gate_pair(float alog, float activation, float projected_b) {
  float softplus = activation > 20.f ? activation : log_one_plus(exponential(activation));
  return make_float2(exponential(-exponential(alog) * softplus), sigmoid(projected_b));
}

__device__ inline float2 fast_gate_pair(float alog, float activation, float projected_b) {
  float activated = __expf(activation);
  float softplus = __logf(1.f + activated);
  float exponent = -__expf(alog) * softplus;
  if (alog >= -80.f && alog <= 80.f && activation >= -2.7f && activation <= 20.f &&
      fabsf(projected_b) <= 80.f && exponent >= -80.f && isfinite(exponent))
    return make_float2(__expf(exponent), __fdividef(1.f, 1.f + __expf(-projected_b)));
  return cold_gate_pair(alog, activation, projected_b);
}

template <int ValueTile, bool Precompute, bool ValidateMetadata, bool InterleavedInputs,
          int StaticQueries = 0, int StaticBatch = 0>
__global__ void __launch_bounds__(128, 1)
    kernel(const __nv_bfloat16 *__restrict__ query, const __nv_bfloat16 *__restrict__ key,
           const __nv_bfloat16 *__restrict__ value, const __nv_bfloat16 *__restrict__ a,
           const __nv_bfloat16 *__restrict__ b, const float *__restrict__ a_log,
           const float *__restrict__ dt_bias, const int *__restrict__ starts,
           const int *__restrict__ reads, const int *__restrict__ writes,
           __nv_bfloat16 *__restrict__ pool, __nv_bfloat16 *__restrict__ output, int rows,
           int batch, int slots, int max_queries) {
  static_assert(ValueTile == 16 || ValueTile == 32);
  constexpr int Pairs = ValueTile / 16;
  constexpr int QKStride = InterleavedInputs ? 10240 : 2048;
  constexpr int ValueStride = InterleavedInputs ? 10240 : 6144;
  constexpr int GateStride = InterleavedInputs ? 96 : 48;
  int sequence = blockIdx.x, head = blockIdx.y;
  int lane = threadIdx.x % 16;
  int value_base = blockIdx.z * ValueTile + (threadIdx.x / 16) * Pairs * 2;
  static_assert(StaticQueries >= 0 && StaticQueries <= 5);
  static_assert(StaticBatch == 0 || StaticBatch == 1 || StaticBatch == 2);
  static_assert((StaticQueries == 0) == (StaticBatch == 0));
  static_assert(StaticBatch == 0 || (Precompute && StaticQueries >= 2 &&
                                     ((ValidateMetadata && InterleavedInputs) ||
                                      (!ValidateMetadata && !InterleavedInputs))));
  int first = StaticQueries ? sequence * StaticQueries : starts[sequence];
  int last = StaticQueries ? first + StaticQueries : starts[sequence + 1];
  int source = reads[sequence];
  if constexpr (ValidateMetadata) {
    require(starts[0] == 0 && starts[batch] == rows);
    if constexpr (StaticQueries > 0)
      require(starts[sequence] == first && starts[sequence + 1] == last);
    require(first >= 0 && last >= first && last <= rows && last - first <= max_queries);
    require(source >= 0 && source < slots);
  }
  if (first == last)
    return;
  int read0 = 0, read1 = 0;
  int write0 = 0, write1 = 0, write2 = 0, write3 = 0, write4 = 0;
  if constexpr (StaticBatch > 0) {
    read0 = reads[0];
    if constexpr (StaticBatch > 1)
      read1 = reads[1];
    write0 = writes[first];
    if constexpr (StaticQueries > 1)
      write1 = writes[first + 1];
    if constexpr (StaticQueries > 2)
      write2 = writes[first + 2];
    if constexpr (StaticQueries > 3)
      write3 = writes[first + 3];
    if constexpr (StaticQueries > 4)
      write4 = writes[first + 4];
  }
  uint64_t state[Pairs][8];
#pragma unroll
  for (int unit = 0; unit < Pairs; ++unit) {
    float low[8], high[8];
    int64_t offset = ((int64_t(source) * 48 + head) * 128 + value_base + unit * 2) * 128 + lane * 8;
    load_eight(pool + offset, low);
    load_eight(pool + offset + 128, high);
#pragma unroll
    for (int element = 0; element < 8; ++element)
      state[unit][element] = pair(low[element], high[element]);
  }
  __shared__ float operands[Precompute ? (StaticQueries ? StaticQueries : 5) : 1][264];
  if constexpr (Precompute) {
    int warp = threadIdx.x / 32, warp_lane = threadIdx.x % 32;
    for (int token = warp; token < last - first; token += 4) {
      int row = first + token;
      float q[4], k[4];
      int64_t offset = int64_t(row) * QKStride + head / 3 * 128 + warp_lane * 4;
      load_four(query + offset, q);
      load_four(key + offset, k);
      float qs = 0.f, ks = 0.f;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        qs += q[i] * q[i];
        ks += k[i] * k[i];
      }
      float q_scale = 0.08838834764831844f * rsqrtf(warp_sum(qs) + 1e-6f);
      float k_scale = rsqrtf(warp_sum(ks) + 1e-6f);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        operands[token][warp_lane * 4 + i] = q[i] * q_scale;
        operands[token][128 + warp_lane * 4 + i] = k[i] * k_scale;
      }
      if constexpr (StaticBatch > 0) {
        float2 gates = fast_gate_pair(
            a_log[head], __bfloat162float(a[int64_t(row) * GateStride + head]) + dt_bias[head],
            __bfloat162float(b[int64_t(row) * GateStride + head]));
        if (warp_lane == 0) {
          operands[token][256] = gates.x;
          operands[token][257] = gates.y;
        }
      } else {
        if (warp_lane == 0) {
          float2 gates = fast_gate_pair(
              a_log[head], __bfloat162float(a[int64_t(row) * GateStride + head]) + dt_bias[head],
              __bfloat162float(b[int64_t(row) * GateStride + head]));
          operands[token][256] = gates.x;
          operands[token][257] = gates.y;
        }
      }
    }
    __syncthreads();
  }
  for (int token = 0; token < (Precompute ? last - first : 1); ++token) {
    int row = first + token;
    int target;
    if constexpr (StaticBatch > 0) {
      int token = row - first;
      target = write0;
      if constexpr (StaticQueries > 1)
        target = token == 1 ? write1 : target;
      if constexpr (StaticQueries > 2)
        target = token == 2 ? write2 : target;
      if constexpr (StaticQueries > 3)
        target = token == 3 ? write3 : target;
      if constexpr (StaticQueries > 4)
        target = token == 4 ? write4 : target;
    } else {
      target = writes[row];
    }
    if constexpr (ValidateMetadata) {
      require(target >= 0 && target < slots);
      if constexpr (StaticBatch > 0) {
        require(target != read0);
        if constexpr (StaticBatch > 1)
          require(target != read1);
      } else {
        for (int request = 0; request < batch; ++request)
          require(target != reads[request]);
      }
    }
    float q[8], k[8], decay = 0.f, beta = 0.f;
    if constexpr (Precompute) {
#pragma unroll
      for (int element = 0; element < 8; ++element) {
        q[element] = operands[row - first][lane * 8 + element];
        k[element] = operands[row - first][128 + lane * 8 + element];
      }
      decay = operands[row - first][256];
      beta = operands[row - first][257];
    } else {
      int64_t offset = int64_t(row) * QKStride + head / 3 * 128 + lane * 8;
      load_eight(query + offset, q);
      load_eight(key + offset, k);
      float q_square = 0.f, k_square = 0.f;
#pragma unroll
      for (int element = 0; element < 8; ++element) {
        q_square += q[element] * q[element];
        k_square += k[element] * k[element];
      }
      float q_scale = 0.08838834764831844f * rsqrtf(half_sum(q_square) + 1e-6f);
      float k_scale = rsqrtf(half_sum(k_square) + 1e-6f);
#pragma unroll
      for (int element = 0; element < 8; ++element) {
        q[element] *= q_scale;
        k[element] *= k_scale;
      }
      float2 gates = fast_gate_pair(
          a_log[head], __bfloat162float(a[int64_t(row) * GateStride + head]) + dt_bias[head],
          __bfloat162float(b[int64_t(row) * GateStride + head]));
      decay = gates.x;
      beta = gates.y;
    }
    uint64_t decay_pair = pair(decay, decay), beta_pair = pair(beta, beta);
#pragma unroll
    for (int unit = 0; unit < Pairs; ++unit) {
      uint64_t recalled = 0;
#pragma unroll
      for (int element = 0; element < 8; ++element) {
        state[unit][element] = multiply(state[unit][element], decay_pair);
        recalled = fma(state[unit][element], pair(k[element], k[element]), recalled);
      }
      recalled = half_sum(recalled);
      uint32_t packed_value = *reinterpret_cast<const uint32_t *>(
          value + int64_t(row) * ValueStride + head * 128 + value_base + unit * 2);
      uint64_t delta =
          multiply(beta_pair, add(from_bf16(packed_value), recalled ^ 0x8000000080000000ull));
      uint64_t sum = 0;
#pragma unroll
      for (int element = 0; element < 8; ++element) {
        state[unit][element] = fma(delta, pair(k[element], k[element]), state[unit][element]);
        sum = fma(state[unit][element], pair(q[element], q[element]), sum);
      }
      int64_t destination =
          ((int64_t(target) * 48 + head) * 128 + value_base + unit * 2) * 128 + lane * 8;
      store_eight(pool + destination, state[unit], false);
      store_eight(pool + destination + 128, state[unit], true);
      sum = half_sum(sum);
      if (lane == 0)
        *reinterpret_cast<uint32_t *>(output + (int64_t(row) * 48 + head) * 128 + value_base +
                                      unit * 2) = to_bf16(sum);
    }
  }
}

template <bool ValidateMetadata = true, bool InterleavedInputs = false>
inline void launch(const __nv_bfloat16 *q, const __nv_bfloat16 *k, const __nv_bfloat16 *v,
                   const __nv_bfloat16 *a, const __nv_bfloat16 *b, const float *a_log,
                   const float *dt_bias, const int *starts, const int *reads, const int *writes,
                   __nv_bfloat16 *pool, __nv_bfloat16 *output, int rows, int batch, int slots,
                   int max_queries, int value_tile, cudaStream_t stream) {
  TVM_FFI_ICHECK(max_queries >= 1 && max_queries <= 5);
  auto run = [&](auto tile, auto precompute, auto query_count, auto batch_count) {
    constexpr int ValueTile = decltype(tile)::value;
    constexpr bool Precompute = decltype(precompute)::value;
    kernel<ValueTile, Precompute, ValidateMetadata, InterleavedInputs, decltype(query_count)::value,
           decltype(batch_count)::value><<<dim3(batch, 48, 128 / ValueTile), 128, 0, stream>>>(
        q, k, v, a, b, a_log, dt_bias, starts, reads, writes, pool, output, rows, batch, slots,
        max_queries);
  };
  auto select = [&](auto tile) {
    if constexpr ((ValidateMetadata && InterleavedInputs) ||
                  (!ValidateMetadata && !InterleavedInputs)) {
      if ((batch == 1 || batch == 2) && max_queries > 1 && rows == int64_t(batch) * max_queries) {
        auto dispatch = [&](auto query_count) {
          if (batch == 1)
            run(tile, std::true_type{}, query_count, std::integral_constant<int, 1>{});
          else
            run(tile, std::true_type{}, query_count, std::integral_constant<int, 2>{});
        };
        switch (max_queries) {
        case 2:
          dispatch(std::integral_constant<int, 2>{});
          break;
        case 3:
          dispatch(std::integral_constant<int, 3>{});
          break;
        case 4:
          dispatch(std::integral_constant<int, 4>{});
          break;
        case 5:
          dispatch(std::integral_constant<int, 5>{});
          break;
        }
        return;
      }
    }
    if (max_queries == 1)
      run(tile, std::false_type{}, std::integral_constant<int, 0>{},
          std::integral_constant<int, 0>{});
    else
      run(tile, std::true_type{}, std::integral_constant<int, 0>{},
          std::integral_constant<int, 0>{});
  };
  TVM_FFI_ICHECK(value_tile == 16 || value_tile == 32);
  if (value_tile == 16)
    select(std::integral_constant<int, 16>{});
  else
    select(std::integral_constant<int, 32>{});
  check_cuda(cudaGetLastError());
}

// The caller must first validate fixed offsets, read slots, and private export slots.
inline constexpr auto launch_validated_metadata = launch<false, false>;

// Q/K/V share row stride 10240. A/B share row stride 96; pass A=BA+48 and B=BA.
inline constexpr auto launch_interleaved = launch<true, true>;

// This entry also requires the fixed metadata validation described above.
inline constexpr auto launch_interleaved_validated_metadata = launch<false, true>;

} // namespace gdn_recurrent
