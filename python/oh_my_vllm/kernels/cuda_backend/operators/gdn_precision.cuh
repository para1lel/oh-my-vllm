#pragma once

namespace gdn_chunk {
__device__ void scan_constant_key_chunk(
    const __nv_bfloat16 *queries, const __nv_bfloat16 *keys, const __nv_bfloat16 *values,
    const float *decay, const float *beta, const float *cumulative, int64_t position, int count,
    int hk, int hv, int head, int value_offset, int64_t qs, int64_t ks, int64_t vs, float *state,
    __nv_bfloat16 *rounded_state, __nv_bfloat16 *a, __nv_bfloat16 *b, float *temporary,
    float *projection, float *query_key, float *gamma, float *multipliers, __nv_bfloat16 *output) {
  __shared__ float updates[64];
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, query_head = head / (hv / hk);
  float key[4], squared_norm = 0;
#pragma unroll
  for (int column = 0; column < 4; ++column) {
    key[column] = __bfloat162float(keys[position * ks + query_head * 128 + lane + column * 32]);
    squared_norm += key[column] * key[column];
  }
  squared_norm = warp_sum(squared_norm);
#pragma unroll
  for (int row = 0; row < 8; ++row) {
    float recalled = 0;
#pragma unroll
    for (int column = 0; column < 4; ++column)
      recalled += state[(warp * 8 + row) * key_pitch + lane + column * 32] * key[column];
    recalled = warp_sum(recalled);
    if (lane == 0)
      projection[warp * 8 + row] = recalled;
  }
  for (int i = threadIdx.x; i < 64 * 16; i += blockDim.x) {
    int row = i / 16, column = i % 16 * 8;
    copy_bf16_8(a + row * key_pitch + column,
                queries + (position + min(row, count - 1)) * qs + query_head * 128 + column,
                row < count);
  }
  for (int i = threadIdx.x; i < 64 * (value_tile / 8); i += blockDim.x) {
    int row = i / (value_tile / 8), column = i % (value_tile / 8) * 8;
    copy_bf16_8(b + row * value_pitch + column,
                values + (position + min(row, count - 1)) * vs + head * 128 + value_offset + column,
                row < count);
  }
  if (threadIdx.x < count) {
    const int row = threadIdx.x;
    gamma[row] = expf(cumulative[row]);
    multipliers[row] = expf(decay[(position + row) * hv + head]);
    updates[row] = beta[(position + row) * hv + head];
  }
  wait_copies();
  for (int row = warp; row < count; row += blockDim.x / 32) {
    float dot = 0;
#pragma unroll
    for (int column = 0; column < 4; ++column)
      dot += __bfloat162float(a[row * key_pitch + lane + column * 32]) * key[column];
    dot = warp_sum(dot);
    if (lane == 0)
      query_key[row] = dot;
  }
  multiply<64, value_tile, 128, false, true>(a, key_pitch, rounded_state, key_pitch, temporary,
                                             value_pitch);
  for (int i = threadIdx.x; i < value_tile * 128; i += blockDim.x) {
    const int index = i / 128 * key_pitch + i % 128;
    rounded_state[index] =
        __float2bfloat16_rn(state[index] - __bfloat162float(rounded_state[index]));
  }
  __syncthreads();
  multiply<64, value_tile, 128, false, true>(a, key_pitch, rounded_state, key_pitch, temporary,
                                             value_pitch, true);
  if (threadIdx.x < value_tile) {
    const int value = threadIdx.x;
    const float initial_projection = projection[value];
    float recalled = initial_projection, displacement = 0;
    for (int row = 0; row < count; ++row) {
      const float decayed = multipliers[row] * recalled;
      recalled = decayed + squared_norm * updates[row] *
                               (__bfloat162float(b[row * value_pitch + value]) - decayed);
      displacement = (recalled - gamma[row] * initial_projection) / squared_norm;
      temporary[row * value_pitch + value] =
          (gamma[row] * temporary[row * value_pitch + value] + displacement * query_key[row]) *
          0.08838834764831844f;
    }
    projection[value] = displacement;
  }
  __syncthreads();
  for (int i = threadIdx.x; i < count * value_tile; i += blockDim.x) {
    int row = i / value_tile, value = i % value_tile;
    output[((position + row) * hv + head) * 128 + value_offset + value] =
        __float2bfloat16_rn(temporary[row * value_pitch + value]);
  }
  for (int i = threadIdx.x; i < value_tile * 128; i += blockDim.x) {
    int row = i / 128, column = i % 128, index = row * key_pitch + column;
    const float k = __bfloat162float(keys[position * ks + query_head * 128 + column]);
    state[index] = gamma[count - 1] * state[index] + projection[row] * k;
    rounded_state[index] = __float2bfloat16_rn(state[index]);
  }
  __syncthreads();
}

__device__ void scan_precise_chunk(const __nv_bfloat16 *queries, const __nv_bfloat16 *keys,
                                   const __nv_bfloat16 *values, const float *decay,
                                   const float *beta, int64_t position, int count, int hk, int hv,
                                   int head, int value_offset, int64_t qs, int64_t ks, int64_t vs,
                                   float *shared_state, __nv_bfloat16 *rounded_state,
                                   unsigned char *storage, __nv_bfloat16 *output) {
  constexpr int lanes = precise_subgroup_lanes, columns = 128 / lanes;
  const int lane = threadIdx.x % lanes,
            first_value = threadIdx.x / lanes * precise_values_per_group;
  const int query_head = head / (hv / hk);
  auto shared_queries = reinterpret_cast<__nv_bfloat16 *>(storage + 26112);
  auto shared_keys = reinterpret_cast<__nv_bfloat16 *>(storage + 42496);
  auto shared_values = reinterpret_cast<__nv_bfloat16 *>(storage + 58880);
  auto multipliers = reinterpret_cast<float *>(storage + 62976);
  auto updates = reinterpret_cast<float *>(storage + 63232);
  for (int i = threadIdx.x; i < 64 * 16; i += blockDim.x) {
    const int row = i / 16, column = i % 16 * 8;
    copy_bf16_8(shared_queries + row * 128 + column,
                queries + (position + min(row, count - 1)) * qs + query_head * 128 + column,
                row < count);
    copy_bf16_8(shared_keys + row * 128 + column,
                keys + (position + min(row, count - 1)) * ks + query_head * 128 + column,
                row < count);
  }
  for (int i = threadIdx.x; i < 64 * (value_tile / 8); i += blockDim.x) {
    const int row = i / (value_tile / 8), column = i % (value_tile / 8) * 8;
    copy_bf16_8(shared_values + row * value_tile + column,
                values + (position + min(row, count - 1)) * vs + head * 128 + value_offset + column,
                row < count);
  }
  if (threadIdx.x < count) {
    multipliers[threadIdx.x] = expf(decay[(position + threadIdx.x) * hv + head]);
    updates[threadIdx.x] = beta[(position + threadIdx.x) * hv + head];
  }
  wait_copies();
  float2 state[columns];
#pragma unroll
  for (int column = 0; column < columns; ++column)
    state[column] =
        make_float2(shared_state[first_value * key_pitch + lane + column * lanes],
                    shared_state[(first_value + 1) * key_pitch + lane + column * lanes]);
  for (int row_index = 0; row_index < count; ++row_index) {
    float query[columns], key[columns];
#pragma unroll
    for (int column = 0; column < columns; ++column) {
      query[column] = __bfloat162float(shared_queries[row_index * 128 + lane + column * lanes]) *
                      0.08838834764831844f;
      key[column] = __bfloat162float(shared_keys[row_index * 128 + lane + column * lanes]);
    }
    const float multiplier = multipliers[row_index], update = updates[row_index];
    float2 recalled = make_float2(0, 0);
#pragma unroll
    for (int column = 0; column < columns; ++column) {
      state[column] = __fmul2_rn(state[column], make_float2(multiplier, multiplier));
      recalled = __ffma2_rn(state[column], make_float2(key[column], key[column]), recalled);
    }
#pragma unroll
    for (int offset = lanes / 2; offset; offset >>= 1)
      recalled =
          __fadd2_rn(recalled, make_float2(__shfl_xor_sync(0xffffffff, recalled.x, offset, lanes),
                                           __shfl_xor_sync(0xffffffff, recalled.y, offset, lanes)));
    float2 value = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(
        shared_values + row_index * value_tile + first_value));
    float2 delta = __fmul2_rn(__fadd2_rn(value, make_float2(-recalled.x, -recalled.y)),
                              make_float2(update, update));
    float2 result = make_float2(0, 0);
#pragma unroll
    for (int column = 0; column < columns; ++column) {
      state[column] = __ffma2_rn(delta, make_float2(key[column], key[column]), state[column]);
      result = __ffma2_rn(state[column], make_float2(query[column], query[column]), result);
    }
#pragma unroll
    for (int offset = lanes / 2; offset; offset >>= 1)
      result =
          __fadd2_rn(result, make_float2(__shfl_xor_sync(0xffffffff, result.x, offset, lanes),
                                         __shfl_xor_sync(0xffffffff, result.y, offset, lanes)));
    if (lane == 0) {
      auto destination =
          output + ((position + row_index) * hv + head) * 128 + value_offset + first_value;
      if ((reinterpret_cast<uintptr_t>(destination) & 3) == 0)
        *reinterpret_cast<__nv_bfloat162 *>(destination) = __float22bfloat162_rn(result);
      else {
        destination[0] = __float2bfloat16_rn(result.x);
        destination[1] = __float2bfloat16_rn(result.y);
      }
    }
  }
#pragma unroll
  for (int column = 0; column < columns; ++column) {
    int index = first_value * key_pitch + lane + column * lanes;
    shared_state[index] = state[column].x;
    shared_state[index + key_pitch] = state[column].y;
    rounded_state[index] = __float2bfloat16_rn(state[column].x);
    rounded_state[index + key_pitch] = __float2bfloat16_rn(state[column].y);
  }
  __syncthreads();
}

// Precision-only persistent chunk. The per-token arithmetic and every BF16
// rounding boundary are identical to scan_precise_chunk; only the state
// location changes. The caller keeps the FP32 state live in registers across
// the whole chunk loop of one (sequence, head) and only selects this path when
// every chunk of that sequence/head has conditioning == 1, so the shared
// state<->register roundtrip and the (precision-dead) BF16 state mirror write
// are removed. The mixed-route loop in scan_kernel is unchanged.
__device__ __forceinline__ void
scan_precision_persistent_chunk(const __nv_bfloat16 *queries, const __nv_bfloat16 *keys,
                                const __nv_bfloat16 *values, const float *decay, const float *beta,
                                int64_t position, int count, int hk, int hv, int head,
                                int value_offset, int64_t qs, int64_t ks, int64_t vs,
                                unsigned char *storage, float2 *state, __nv_bfloat16 *output) {
  constexpr int lanes = precise_subgroup_lanes, columns = 128 / lanes;
  const int lane = threadIdx.x % lanes,
            first_value = threadIdx.x / lanes * precise_values_per_group;
  const int query_head = head / (hv / hk);
  auto shared_queries = reinterpret_cast<__nv_bfloat16 *>(storage + 26112);
  auto shared_keys = reinterpret_cast<__nv_bfloat16 *>(storage + 42496);
  auto shared_values = reinterpret_cast<__nv_bfloat16 *>(storage + 58880);
  auto multipliers = reinterpret_cast<float *>(storage + 62976);
  auto updates = reinterpret_cast<float *>(storage + 63232);
  for (int i = threadIdx.x; i < 64 * 16; i += blockDim.x) {
    const int row = i / 16, column = i % 16 * 8;
    copy_bf16_8(shared_queries + row * 128 + column,
                queries + (position + min(row, count - 1)) * qs + query_head * 128 + column,
                row < count);
    copy_bf16_8(shared_keys + row * 128 + column,
                keys + (position + min(row, count - 1)) * ks + query_head * 128 + column,
                row < count);
  }
  for (int i = threadIdx.x; i < 64 * (value_tile / 8); i += blockDim.x) {
    const int row = i / (value_tile / 8), column = i % (value_tile / 8) * 8;
    copy_bf16_8(shared_values + row * value_tile + column,
                values + (position + min(row, count - 1)) * vs + head * 128 + value_offset + column,
                row < count);
  }
  if (threadIdx.x < count) {
    multipliers[threadIdx.x] = expf(decay[(position + threadIdx.x) * hv + head]);
    updates[threadIdx.x] = beta[(position + threadIdx.x) * hv + head];
  }
  wait_copies();
  for (int row_index = 0; row_index < count; ++row_index) {
    float query[columns], key[columns];
#pragma unroll
    for (int column = 0; column < columns; ++column) {
      query[column] = __bfloat162float(shared_queries[row_index * 128 + lane + column * lanes]) *
                      0.08838834764831844f;
      key[column] = __bfloat162float(shared_keys[row_index * 128 + lane + column * lanes]);
    }
    const float multiplier = multipliers[row_index], update = updates[row_index];
    float2 recalled = make_float2(0, 0);
#pragma unroll
    for (int column = 0; column < columns; ++column) {
      state[column] = __fmul2_rn(state[column], make_float2(multiplier, multiplier));
      recalled = __ffma2_rn(state[column], make_float2(key[column], key[column]), recalled);
    }
#pragma unroll
    for (int offset = lanes / 2; offset; offset >>= 1)
      recalled =
          __fadd2_rn(recalled, make_float2(__shfl_xor_sync(0xffffffff, recalled.x, offset, lanes),
                                           __shfl_xor_sync(0xffffffff, recalled.y, offset, lanes)));
    float2 value = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(
        shared_values + row_index * value_tile + first_value));
    float2 delta = __fmul2_rn(__fadd2_rn(value, make_float2(-recalled.x, -recalled.y)),
                              make_float2(update, update));
    float2 result = make_float2(0, 0);
#pragma unroll
    for (int column = 0; column < columns; ++column) {
      state[column] = __ffma2_rn(delta, make_float2(key[column], key[column]), state[column]);
      result = __ffma2_rn(state[column], make_float2(query[column], query[column]), result);
    }
#pragma unroll
    for (int offset = lanes / 2; offset; offset >>= 1)
      result =
          __fadd2_rn(result, make_float2(__shfl_xor_sync(0xffffffff, result.x, offset, lanes),
                                         __shfl_xor_sync(0xffffffff, result.y, offset, lanes)));
    if (lane == 0) {
      auto destination =
          output + ((position + row_index) * hv + head) * 128 + value_offset + first_value;
      if ((reinterpret_cast<uintptr_t>(destination) & 3) == 0)
        *reinterpret_cast<__nv_bfloat162 *>(destination) = __float22bfloat162_rn(result);
      else {
        destination[0] = __float2bfloat16_rn(result.x);
        destination[1] = __float2bfloat16_rn(result.y);
      }
    }
  }
  __syncthreads();
}

} // namespace gdn_chunk
