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
  for (int row = 0; row < 2; ++row) {
    float recalled = 0;
#pragma unroll
    for (int column = 0; column < 4; ++column)
      recalled += state[(warp * 2 + row) * key_pitch + lane + column * 32] * key[column];
    recalled = warp_sum(recalled);
    if (lane == 0)
      projection[warp * 2 + row] = recalled;
  }
  for (int i = threadIdx.x; i < 64 * 16; i += blockDim.x) {
    int row = i / 16, column = i % 16 * 8;
    copy_bf16_8(a + row * key_pitch + column,
                queries + (position + min(row, count - 1)) * qs + query_head * 128 + column,
                row < count);
  }
  for (int i = threadIdx.x; i < 64 * 2; i += blockDim.x) {
    int row = i / 2, column = i % 2 * 8;
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
  for (int row = warp; row < count; row += 8) {
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
  const int lane = threadIdx.x % 32, first_value = threadIdx.x / 32 * 2;
  const int query_head = head / (hv / hk);
  auto shared_queries = reinterpret_cast<__nv_bfloat16 *>(storage + 13056);
  auto shared_keys = reinterpret_cast<__nv_bfloat16 *>(storage + 29440);
  auto shared_values = reinterpret_cast<__nv_bfloat16 *>(storage + 45824);
  auto multipliers = reinterpret_cast<float *>(storage + 47872);
  auto updates = reinterpret_cast<float *>(storage + 48128);
  for (int i = threadIdx.x; i < 64 * 16; i += blockDim.x) {
    const int row = i / 16, column = i % 16 * 8;
    copy_bf16_8(shared_queries + row * 128 + column,
                queries + (position + min(row, count - 1)) * qs + query_head * 128 + column,
                row < count);
    copy_bf16_8(shared_keys + row * 128 + column,
                keys + (position + min(row, count - 1)) * ks + query_head * 128 + column,
                row < count);
  }
  for (int i = threadIdx.x; i < 64 * 2; i += blockDim.x) {
    const int row = i / 2, column = i % 2 * 8;
    copy_bf16_8(shared_values + row * value_tile + column,
                values + (position + min(row, count - 1)) * vs + head * 128 + value_offset + column,
                row < count);
  }
  if (threadIdx.x < count) {
    multipliers[threadIdx.x] = expf(decay[(position + threadIdx.x) * hv + head]);
    updates[threadIdx.x] = beta[(position + threadIdx.x) * hv + head];
  }
  wait_copies();
  float state[2][4];
#pragma unroll
  for (int row = 0; row < 2; ++row)
#pragma unroll
    for (int column = 0; column < 4; ++column)
      state[row][column] = shared_state[(first_value + row) * key_pitch + lane + column * 32];
  for (int row_index = 0; row_index < count; ++row_index) {
    float query[4], key[4];
#pragma unroll
    for (int column = 0; column < 4; ++column) {
      query[column] = __bfloat162float(shared_queries[row_index * 128 + lane + column * 32]) *
                      0.08838834764831844f;
      key[column] = __bfloat162float(shared_keys[row_index * 128 + lane + column * 32]);
    }
    const float multiplier = multipliers[row_index], update = updates[row_index];
#pragma unroll
    for (int row = 0; row < 2; ++row) {
      float recalled = 0;
#pragma unroll
      for (int column = 0; column < 4; ++column) {
        state[row][column] *= multiplier;
        recalled += state[row][column] * key[column];
      }
      recalled = warp_sum(recalled);
      const int value = value_offset + first_value + row;
      const float delta =
          (__bfloat162float(shared_values[row_index * value_tile + first_value + row]) - recalled) *
          update;
      float result = 0;
#pragma unroll
      for (int column = 0; column < 4; ++column) {
        state[row][column] += delta * key[column];
        result += state[row][column] * query[column];
      }
      result = warp_sum(result);
      if (lane == 0)
        output[((position + row_index) * hv + head) * 128 + value] = __float2bfloat16_rn(result);
    }
  }
#pragma unroll
  for (int row = 0; row < 2; ++row)
#pragma unroll
    for (int column = 0; column < 4; ++column) {
      const int index = (first_value + row) * key_pitch + lane + column * 32;
      shared_state[index] = state[row][column];
      rounded_state[index] = __float2bfloat16_rn(state[row][column]);
    }
  __syncthreads();
}

} // namespace gdn_chunk
