#pragma once

template <bool ValidateMetadata, int StaticQueries>
__device__ __forceinline__ void validate_recurrent_input(const int *starts, int sequence, int first,
                                                         int last, int source, int rows, int batch,
                                                         int slots, int max_queries) {
  if constexpr (ValidateMetadata) {
    require(starts[0] == 0 && starts[batch] == rows);
    if constexpr (StaticQueries > 0)
      require(starts[sequence] == first && starts[sequence + 1] == last);
    require(first >= 0 && last >= first && last <= rows && last - first <= max_queries);
    require(source >= 0 && source < slots);
  }
}

template <bool Precompute, bool InterleavedInputs, int StaticBatch, int Capacity>
__device__ __forceinline__ void
prepare_recurrent_operands(const __nv_bfloat16 *query, const __nv_bfloat16 *key,
                           const __nv_bfloat16 *a, const __nv_bfloat16 *b, const float *a_log,
                           const float *dt_bias, float (&operands)[Capacity][264], int first,
                           int last, int head) {
  constexpr int QKStride = InterleavedInputs ? 10240 : 2048;
  constexpr int GateStride = InterleavedInputs ? 96 : 48;
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
}

template <bool ValidateMetadata, int StaticQueries, int StaticBatch>
__device__ __forceinline__ int
recurrent_target(const int *reads, const int *writes, int row, int first, int slots, int batch,
                 int read0, int read1, int write0, int write1, int write2, int write3, int write4) {
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
  return target;
}

template <bool Precompute, bool InterleavedInputs, int Capacity>
__device__ __forceinline__ void
load_recurrent_operands(const __nv_bfloat16 *query, const __nv_bfloat16 *key,
                        const __nv_bfloat16 *a, const __nv_bfloat16 *b, const float *a_log,
                        const float *dt_bias, const float (&operands)[Capacity][264], float (&q)[8],
                        float (&k)[8], float &decay, float &beta, int row, int first, int head,
                        int lane) {
  constexpr int QKStride = InterleavedInputs ? 10240 : 2048;
  constexpr int GateStride = InterleavedInputs ? 96 : 48;
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
}

template <bool InterleavedInputs, int Pairs>
__device__ __forceinline__ void
recurrent_update(uint64_t (&state)[Pairs][8], const float (&q)[8], const float (&k)[8],
                 const __nv_bfloat16 *value, float *pool, __nv_bfloat16 *output, float decay,
                 float beta, int row, int target, int head, int value_base, int lane) {
  constexpr int ValueStride = InterleavedInputs ? 10240 : 6144;
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
    uint32_t packed_value = *reinterpret_cast<const uint32_t *>(value + int64_t(row) * ValueStride +
                                                                head * 128 + value_base + unit * 2);
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
