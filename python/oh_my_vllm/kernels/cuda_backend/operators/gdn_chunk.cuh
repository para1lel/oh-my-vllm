#pragma once

#include "gdn_contract.cuh"
#include <mma.h>

namespace gdn_chunk {
__device__ void copy16(void *destination, const void *source, bool valid = true) {
  uint32_t dst = __cvta_generic_to_shared(destination);
  int bytes = valid ? 16 : 0;
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(source), "r"(bytes));
}

__device__ void copy_bf16_8(__nv_bfloat16 *destination, const __nv_bfloat16 *source,
                            bool valid = true) {
  if ((reinterpret_cast<uintptr_t>(source) & 15) == 0)
    copy16(destination, source, valid);
  else {
#pragma unroll
    for (int i = 0; i < 8; ++i)
      destination[i] = valid ? source[i] : __float2bfloat16(0);
  }
}

__device__ AlignedVector<__nv_bfloat16, 8> load_bf16_8(const __nv_bfloat16 *source) {
  if ((reinterpret_cast<uintptr_t>(source) & 15) == 0)
    return load_aligned_vector<__nv_bfloat16, 8>(source);
  AlignedVector<__nv_bfloat16, 8> result;
#pragma unroll
  for (int i = 0; i < 8; ++i)
    result.value[i] = source[i];
  return result;
}

__device__ void wait_copies() {
  asm volatile("cp.async.commit_group; cp.async.wait_group 0;" ::: "memory");
  __syncthreads();
}

constexpr int chunk_size = 64;
constexpr int value_tile = 32;
constexpr int precise_subgroup_lanes = 8;
constexpr int precise_values_per_group = 2;
constexpr int key_pitch = 136;
constexpr int value_pitch = 40;
constexpr int transform_pitch = 72;
constexpr int scan_shared_bytes = 77056;

__device__ int chunk_prefix(const void *starts, bool wide, int sequence) {
  int prefix = 0;
  for (int i = 0; i < sequence; ++i)
    prefix += (index_at(starts, wide, i + 1) - index_at(starts, wide, i) + 63) / 64;
  return prefix;
}

template <int Rows, int Columns, int Inner, bool TransposeA = false, bool TransposeB = false>
__device__ void multiply(const __nv_bfloat16 *a, int leading_a, const __nv_bfloat16 *b,
                         int leading_b, float *output, int leading_output,
                         bool accumulate = false) {
  using namespace nvcuda;
  using LayoutA = std::conditional_t<TransposeA, wmma::col_major, wmma::row_major>;
  using LayoutB = std::conditional_t<TransposeB, wmma::col_major, wmma::row_major>;
  const int warp = threadIdx.x / 32;
  for (int tile = warp; tile < Rows / 16 * (Columns / 16); tile += blockDim.x / 32) {
    const int row = tile / (Columns / 16) * 16, column = tile % (Columns / 16) * 16;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> sum;
    if (accumulate)
      wmma::load_matrix_sync(sum, output + row * leading_output + column, leading_output,
                             wmma::mem_row_major);
    else
      wmma::fill_fragment(sum, 0.0f);
    for (int inner = 0; inner < Inner; inner += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, LayoutA> left;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, LayoutB> right;
      const int left_offset = TransposeA ? inner * leading_a + row : row * leading_a + inner;
      const int right_offset = TransposeB ? column * leading_b + inner : inner * leading_b + column;
      wmma::load_matrix_sync(left, a + left_offset, leading_a);
      wmma::load_matrix_sync(right, b + right_offset, leading_b);
      wmma::mma_sync(sum, left, right, sum);
    }
    wmma::store_matrix_sync(output + row * leading_output + column, sum, leading_output,
                            wmma::mem_row_major);
  }
  __syncthreads();
}

__device__ void invert_unit_triangular(const float *lower, float *inverse, float *product) {
  if (threadIdx.x < 64) {
    const int column = threadIdx.x, limit = (column / 8 + 1) * 8;
    for (int row = column + 1; row < limit; ++row) {
      float value = -lower[row * 64 + column];
      for (int inner = column + 1; inner < row; ++inner)
        value -= lower[row * 64 + inner] * inverse[inner * 64 + column];
      inverse[row * 64 + column] = value;
    }
  }
  __syncthreads();
  // Cross blocks use -C^-1 B A^-1 in FP32 before the coefficient's BF16 boundary.
  for (int width = 16; width <= 64; width *= 2) {
    const int half = width / 2, blocks = 64 / width, total = blocks * half * half;
    for (int index = threadIdx.x; index < total; index += blockDim.x) {
      const int block = index / (half * half);
      const int row = block * width + half + index / half % half;
      const int column = block * width + index % half;
      float result = 0;
      for (int inner = block * width + half; inner <= row; ++inner)
        result += inverse[row * 64 + inner] * lower[inner * 64 + column];
      product[row * 64 + column] = result;
    }
    __syncthreads();
    for (int index = threadIdx.x; index < total; index += blockDim.x) {
      const int block = index / (half * half);
      const int row = block * width + half + index / half % half;
      const int column = block * width + index % half;
      float result = 0;
      for (int inner = column; inner < block * width + half; ++inner)
        result -= product[row * 64 + inner] * inverse[inner * 64 + column];
      inverse[row * 64 + column] = result;
    }
    __syncthreads();
  }
}

__global__ void prepare_kernel(const __nv_bfloat16 *__restrict__ keys,
                               const float *__restrict__ decay, const float *__restrict__ beta,
                               const void *__restrict__ starts, bool wide, int sequences,
                               int query_heads, int value_heads, int64_t tokens, int64_t key_stride,
                               __nv_bfloat16 *__restrict__ coefficients,
                               float *__restrict__ cumulative_decay, int32_t *conditioning) {
  pdl_dependency_wait();
  pdl_launch_next();
  if (!valid_gdn_offsets(starts, wide, sequences, tokens)) {
    asm volatile("trap;");
    return;
  }
  const int chunk = blockIdx.x, head = blockIdx.y;
  int sequence = 0, prefix = 0;
  for (; sequence < sequences; ++sequence) {
    int chunks =
        (index_at(starts, wide, sequence + 1) - index_at(starts, wide, sequence) + 63) / 64;
    if (chunk < prefix + chunks)
      break;
    prefix += chunks;
  }
  if (sequence == sequences)
    return;
  const int64_t first = index_at(starts, wide, sequence) + (chunk - prefix) * 64;
  const int64_t end = min(first + 64, index_at(starts, wide, sequence + 1));
  const int count = end - first;
  const int query_head = head / (value_heads / query_heads);
  extern __shared__ __align__(32) unsigned char storage[];
  auto key = reinterpret_cast<__nv_bfloat16 *>(storage);
  auto lower = reinterpret_cast<float *>(storage + 17408);
  auto inverse = reinterpret_cast<float *>(storage + 33792);
  auto log_decay = reinterpret_cast<float *>(storage + 50176);
  auto update = reinterpret_cast<float *>(storage + 50432);
  for (int i = threadIdx.x; i < 64 * 16; i += blockDim.x) {
    int row = i / 16, key_column = i % 16 * 8;
    copy_bf16_8(key + row * 136 + key_column,
                keys + (first + min(row, count - 1)) * key_stride + query_head * 128 + key_column,
                row < count);
  }
  if (threadIdx.x < 64) {
    int row = threadIdx.x;
    log_decay[row] = row < count ? decay[(first + row) * value_heads + head] : 0;
    update[row] = row < count ? beta[(first + row) * value_heads + head] : 0;
  }
  wait_copies();
  if (threadIdx.x == 0) {
    float total = 0;
    for (int row = 0; row < 64; ++row) {
      if (row < count)
        total += log_decay[row];
      log_decay[row] = total;
    }
  }
  __syncthreads();
  bool different_key = false;
  for (int i = threadIdx.x; i < count * 128; i += blockDim.x)
    different_key |= __bfloat16_as_ushort(key[i / 128 * key_pitch + i % 128]) !=
                     __bfloat16_as_ushort(key[i % 128]);
  const bool constant_key = __syncthreads_or(different_key) == 0;
  float squared_norm = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float value = __bfloat162float(key[threadIdx.x % 32 + i * 32]);
    squared_norm += value * value;
  }
  squared_norm = warp_sum(squared_norm);
  const int64_t base = (int64_t(chunk) * value_heads + head) * 64;
  if (constant_key && squared_norm >= 0.5f) {
    if (threadIdx.x == 0)
      conditioning[int64_t(chunk) * value_heads + head] = 2;
    if (threadIdx.x < 64)
      cumulative_decay[base + threadIdx.x] = log_decay[threadIdx.x];
    return;
  }
  multiply<64, 64, 128, false, true>(key, 136, key, 136, lower, 64);
  bool needs_precise_recurrence = false;
  for (int row = threadIdx.x / 32; row < count; row += blockDim.x / 32) {
    float row_squared_norm = 0;
    for (int column = threadIdx.x % 32; column < count; column += 32) {
      const float correlation = column == row ? 0 : lower[row * 64 + column];
      row_squared_norm += correlation * correlation;
      needs_precise_recurrence |= fabsf(correlation) > 0.3f;
    }
    needs_precise_recurrence |= warp_sum(row_squared_norm) > 1.0f;
  }
  const int precise_recurrence = __syncthreads_or(needs_precise_recurrence);
  if (threadIdx.x == 0)
    conditioning[int64_t(chunk) * value_heads + head] = precise_recurrence;
  if (threadIdx.x < 64)
    cumulative_decay[base + threadIdx.x] = log_decay[threadIdx.x];
  if (precise_recurrence)
    return;
  for (int i = threadIdx.x; i < 64 * 64; i += blockDim.x) {
    const int row = i / 64, column = i % 64;
    lower[i] =
        row > column ? lower[i] * update[row] * expf(log_decay[row] - log_decay[column]) : 0.0f;
    inverse[i] = row == column ? 1.0f : 0.0f;
  }
  __syncthreads();
  invert_unit_triangular(lower, inverse, reinterpret_cast<float *>(key));
  for (int i = threadIdx.x; i < 64 * 64; i += blockDim.x)
    coefficients[base * 64 + i] = __float2bfloat16_rn(inverse[i] * update[i % 64]);
}

__global__ void project_kernel(const __nv_bfloat16 *queries, const __nv_bfloat16 *keys,
                               const __nv_bfloat16 *values, const void *starts, bool wide,
                               int sequences, int hk, int hv, int64_t qs, int64_t ks, int64_t vs,
                               const __nv_bfloat16 *coeff, const float *decay,
                               __nv_bfloat16 *transformed_keys, float *transformed_values,
                               __nv_bfloat16 *query_key_scores, __nv_bfloat16 *carried_keys,
                               int64_t tokens, const int32_t *conditioning) {
  pdl_dependency_wait();
  pdl_launch_next();
  if (!valid_gdn_offsets(starts, wide, sequences, tokens)) {
    asm volatile("trap;");
    return;
  }
  int chunk = blockIdx.x, head = blockIdx.y, sequence = 0, prefix = 0;
  for (; sequence < sequences; ++sequence) {
    int chunks =
        (index_at(starts, wide, sequence + 1) - index_at(starts, wide, sequence) + 63) / 64;
    if (chunk < prefix + chunks)
      break;
    prefix += chunks;
  }
  if (sequence == sequences)
    return;
  if (conditioning[int64_t(chunk) * hv + head])
    return;
  int64_t first = index_at(starts, wide, sequence) + (chunk - prefix) * 64;
  int count = min(int64_t(64), index_at(starts, wide, sequence + 1) - first), qh = head / (hv / hk);
  int64_t base = (int64_t(chunk) * hv + head) * 64;
  extern __shared__ __align__(32) unsigned char memory[];
  auto a = reinterpret_cast<__nv_bfloat16 *>(memory);
  auto b = reinterpret_cast<__nv_bfloat16 *>(memory + 17408);
  auto transform = reinterpret_cast<__nv_bfloat16 *>(memory + 34816);
  auto temporary = reinterpret_cast<float *>(memory + 44032);
  auto logs = reinterpret_cast<float *>(memory + 78848);
  if (threadIdx.x < 64)
    logs[threadIdx.x] = decay[base + threadIdx.x];
  for (int i = threadIdx.x; i < 64 * 8; i += blockDim.x) {
    int row = i / 8, col = i % 8 * 8;
    copy16(transform + row * 72 + col, coeff + base * 64 + row * 64 + col);
  }
  wait_copies();
  for (int i = threadIdx.x; i < 64 * 16; i += blockDim.x) {
    int row = i / 16, col = i % 16 * 8;
    auto ka = load_bf16_8(keys + (first + min(row, count - 1)) * ks + qh * 128 + col);
    auto vb = load_bf16_8(values + (first + min(row, count - 1)) * vs + head * 128 + col);
    float factor = expf(logs[row]);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      ka.value[j] = row < count ? __float2bfloat16_rn(__bfloat162float(ka.value[j]) * factor)
                                : __float2bfloat16(0);
      vb.value[j] = row < count ? vb.value[j] : __float2bfloat16(0);
    }
    *reinterpret_cast<AlignedVector<__nv_bfloat16, 8> *>(a + row * 136 + col) = ka;
    *reinterpret_cast<AlignedVector<__nv_bfloat16, 8> *>(b + row * 136 + col) = vb;
  }
  __syncthreads();
  multiply<64, 128, 64>(transform, 72, a, 136, temporary, 136);
  for (int i = threadIdx.x; i < 64 * 16; i += blockDim.x) {
    int row = i / 16, col = i % 16 * 8;
    auto result = load_aligned_vector<float, 8>(temporary + row * 136 + col);
    AlignedVector<__nv_bfloat16, 8> packed;
#pragma unroll
    for (int j = 0; j < 8; ++j)
      packed.value[j] = __float2bfloat16_rn(-result.value[j]);
    *reinterpret_cast<AlignedVector<__nv_bfloat16, 8> *>(transformed_keys + (base + row) * 128 +
                                                         col) = packed;
  }
  __syncthreads();
  multiply<64, 128, 64>(transform, 72, b, 136, temporary, 136);
  for (int i = threadIdx.x; i < 64 * 32; i += blockDim.x) {
    int row = i / 32, col = i % 32 * 4;
    *reinterpret_cast<float4 *>(transformed_values + (base + row) * 128 + col) =
        *reinterpret_cast<float4 *>(temporary + row * 136 + col);
  }
  for (int i = threadIdx.x; i < 64 * 16; i += blockDim.x) {
    int row = i / 16, col = i % 16 * 8;
    copy_bf16_8(a + row * 136 + col, queries + (first + min(row, count - 1)) * qs + qh * 128 + col,
                row < count);
    copy_bf16_8(b + row * 136 + col, keys + (first + min(row, count - 1)) * ks + qh * 128 + col,
                row < count);
  }
  wait_copies();
  multiply<64, 64, 128, false, true>(a, 136, b, 136, temporary, 72);
  for (int i = threadIdx.x; i < 64 * 16; i += blockDim.x) {
    int row = i / 16, col = i % 16 * 4;
    auto result = load_four<float, true>(temporary + row * 72 + col);
    Four<__nv_bfloat16> packed;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float score = row < count && col + j <= row
                        ? result.values[j] * expf(logs[row] - logs[col + j]) * 0.08838834764831844f
                        : 0;
      packed.values[j] = __float2bfloat16_rn(score);
    }
    *reinterpret_cast<Four<__nv_bfloat16> *>(query_key_scores + (base + row) * 64 + col) = packed;
  }
  for (int i = threadIdx.x; i < 64 * 16; i += blockDim.x) {
    int row = i / 16, col = i % 16 * 8;
    auto result = load_bf16_8(keys + (first + min(row, count - 1)) * ks + qh * 128 + col);
    float factor = expf(logs[count - 1] - logs[row]);
#pragma unroll
    for (int j = 0; j < 8; ++j)
      result.value[j] =
          __float2bfloat16_rn((row < count ? __bfloat162float(result.value[j]) : 0) * factor);
    *reinterpret_cast<AlignedVector<__nv_bfloat16, 8> *>(carried_keys + (base + row) * 128 + col) =
        result;
  }
}

} // namespace gdn_chunk

#include "gdn_precision.cuh"

namespace gdn_chunk {
__global__ void
scan_kernel(const __nv_bfloat16 *__restrict__ queries, const __nv_bfloat16 *__restrict__ keys,
            const __nv_bfloat16 *__restrict__ values, const float *__restrict__ initial,
            const void *__restrict__ starts, bool wide, int query_heads, int value_heads,
            int sequences, int64_t tokens, int64_t qs, int64_t ks, int64_t vs,
            const __nv_bfloat16 *__restrict__ coefficients,
            const float *__restrict__ cumulative_decay, const __nv_bfloat16 *transformed_keys,
            const float *transformed_values, const __nv_bfloat16 *query_key_scores,
            const __nv_bfloat16 *carried_keys, __nv_bfloat16 *__restrict__ output,
            float *__restrict__ final, const int32_t *conditioning, const float *decay,
            const float *beta) {
  pdl_dependency_wait();
  pdl_launch_next();
  if (!valid_gdn_offsets(starts, wide, sequences, tokens)) {
    asm volatile("trap;");
    return;
  }
  const int sequence = blockIdx.x, head = blockIdx.y, value_offset = blockIdx.z * value_tile;
  const int query_head = head / (value_heads / query_heads);
  const int64_t first = index_at(starts, wide, sequence);
  const int64_t end = index_at(starts, wide, sequence + 1);
  const int prefix = chunk_prefix(starts, wide, sequence);
  const int64_t state_base =
      (int64_t(sequence) * value_heads + head) * 128 * 128 + value_offset * 128;
  extern __shared__ __align__(32) unsigned char storage[];
  auto state = reinterpret_cast<float *>(storage);
  auto state_bf16 = reinterpret_cast<__nv_bfloat16 *>(storage + 17408);
  auto a = reinterpret_cast<__nv_bfloat16 *>(storage + 26112);
  auto b = reinterpret_cast<__nv_bfloat16 *>(storage + 43520);
  auto temporary = reinterpret_cast<float *>(storage + 48640);
  auto delta = reinterpret_cast<float *>(storage + 66048);
  auto log_decay = reinterpret_cast<float *>(storage + 76288);
  auto decayed_rows = reinterpret_cast<float *>(storage + 76544);
  auto final_rows = reinterpret_cast<float *>(storage + 76800);
  for (int i = threadIdx.x; i < value_tile * 128; i += blockDim.x) {
    int shared_index = i / 128 * key_pitch + i % 128;
    state[shared_index] = initial[state_base + i];
    state_bf16[shared_index] = __float2bfloat16_rn(state[shared_index]);
  }
  __syncthreads();
  for (int64_t position = first; position < end; position += 64) {
    const int count = min(int64_t(64), end - position);
    const int chunk = prefix + (position - first) / 64;
    const int64_t base = (int64_t(chunk) * value_heads + head) * 64;
    if (conditioning[int64_t(chunk) * value_heads + head] == 2) {
      scan_constant_key_chunk(queries, keys, values, decay, beta, cumulative_decay + base, position,
                              count, query_heads, value_heads, head, value_offset, qs, ks, vs,
                              state, state_bf16, a, b, temporary, delta, log_decay, decayed_rows,
                              final_rows, output);
      continue;
    }
    if (conditioning[int64_t(chunk) * value_heads + head]) {
      scan_precise_chunk(queries, keys, values, decay, beta, position, count, query_heads,
                         value_heads, head, value_offset, qs, ks, vs, state, state_bf16, storage,
                         output);
      continue;
    }
    if (threadIdx.x < 64) {
      log_decay[threadIdx.x] = cumulative_decay[base + threadIdx.x];
      decayed_rows[threadIdx.x] = expf(cumulative_decay[base + threadIdx.x]);
      final_rows[threadIdx.x] =
          expf(cumulative_decay[base + count - 1] - cumulative_decay[base + threadIdx.x]);
    }
    for (int i = threadIdx.x; i < 64 * 16; i += blockDim.x) {
      int row = i / 16, key = i % 16 * 8;
      copy16(a + row * key_pitch + key, transformed_keys + (base + row) * 128 + key);
    }
    for (int i = threadIdx.x; i < 64 * (value_tile / 4); i += blockDim.x) {
      int row = i / (value_tile / 4), value = i % (value_tile / 4) * 4;
      copy16(delta + row * value_pitch + value,
             transformed_values + (base + row) * 128 + value_offset + value);
    }
    wait_copies();
    multiply<64, value_tile, 128, false, true>(a, key_pitch, state_bf16, key_pitch, delta,
                                               value_pitch, true);
    for (int i = threadIdx.x; i < 64 * 16; i += blockDim.x) {
      int row = i / 16, key = i % 16 * 8;
      copy_bf16_8(a + row * key_pitch + key,
                  queries + (position + min(row, count - 1)) * qs + query_head * 128 + key,
                  row < count);
    }
    wait_copies();
    multiply<64, value_tile, 128, false, true>(a, key_pitch, state_bf16, key_pitch, temporary,
                                               value_pitch);
    for (int i = threadIdx.x; i < 64 * value_tile; i += blockDim.x)
      temporary[i / value_tile * value_pitch + i % value_tile] *=
          decayed_rows[i / value_tile] * 0.08838834764831844f;
    for (int i = threadIdx.x; i < 64 * 8; i += blockDim.x) {
      int row = i / 8, column = i % 8 * 8;
      copy16(a + row * transform_pitch + column, query_key_scores + (base + row) * 64 + column);
    }
    wait_copies();
    for (int i = threadIdx.x; i < 64 * value_tile; i += blockDim.x)
      b[i / value_tile * value_pitch + i % value_tile] =
          __float2bfloat16_rn(delta[i / value_tile * value_pitch + i % value_tile]);
    __syncthreads();
    multiply<64, value_tile, 64>(a, transform_pitch, b, value_pitch, temporary, value_pitch, true);
    for (int i = threadIdx.x; i < count * value_tile; i += blockDim.x) {
      int row = i / value_tile, column = i % value_tile;
      output[((position + row) * value_heads + head) * 128 + value_offset + column] =
          __float2bfloat16_rn(temporary[row * value_pitch + column]);
    }
    for (int i = threadIdx.x; i < 64 * 16; i += blockDim.x) {
      int row = i / 16, key = i % 16 * 8;
      copy16(a + row * key_pitch + key, carried_keys + (base + row) * 128 + key);
    }
    wait_copies();
    multiply<value_tile, 128, 64, true, false>(b, value_pitch, a, key_pitch, temporary, key_pitch);
    const float carry = decayed_rows[count - 1];
    for (int i = threadIdx.x; i < value_tile * 128; i += blockDim.x) {
      int offset = i / 128 * key_pitch + i % 128;
      state[offset] = state[offset] * carry + temporary[offset];
      state_bf16[offset] = __float2bfloat16_rn(state[offset]);
    }
    __syncthreads();
  }
  for (int i = threadIdx.x; i < value_tile * 128; i += blockDim.x)
    final[state_base + i] = state[i / 128 * key_pitch + i % 128];
}

} // namespace gdn_chunk

void gdn_chunk_prepare(TensorView keys, TensorView decay, TensorView beta, TensorView starts,
                       TensorView coefficients, TensorView cumulative_decay,
                       TensorView conditioning) {
  TVM_FFI_ICHECK(keys.ndim() == 3 && keys.size(0) > 0 && keys.size(0) <= INT32_MAX &&
                 keys.size(1) > 0 && decay.ndim() == 2 && decay.size(1) > 0 &&
                 decay.size(1) <= 65535 && decay.size(1) % keys.size(1) == 0);
  validate_gdn_rows(keys, keys, keys.size(0), keys.size(1));
  validate_gdn_starts(starts, keys);
  for (auto gate : {decay, beta})
    TVM_FFI_ICHECK(same_cuda_device(gate, keys) && gate.ndim() == 2 &&
                   gate.size(0) == keys.size(0) && gate.size(1) == decay.size(1) &&
                   has_dtype(gate, kDLFloat, 32) && gate.stride(1) == 1 &&
                   gate.stride(0) == decay.size(1));
  const int64_t chunks = (keys.size(0) + 63) / 64 + starts.size(0) - 2;
  validate_gdn_scratch(coefficients, cumulative_decay, keys, chunks, decay.size(1));
  TVM_FFI_ICHECK(conditioning.ndim() == 2 && conditioning.size(0) == chunks &&
                 conditioning.size(1) == decay.size(1) && has_dtype(conditioning, kDLInt, 32) &&
                 same_cuda_device(conditioning, keys) && aligned(conditioning, 4) &&
                 conditioning.stride(0) == decay.size(1) && conditioning.stride(1) == 1);
  for (auto source : {coefficients, cumulative_decay})
    TVM_FFI_ICHECK(disjoint_storage(conditioning, source));
  for (auto destination : {coefficients, cumulative_decay, conditioning})
    for (auto source : {keys, decay, beta, starts})
      TVM_FFI_ICHECK(disjoint_storage(destination, source));
  TVM_FFI_ICHECK(cudaFuncSetAttribute(gdn_chunk::prepare_kernel,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      50688) == cudaSuccess);
  auto stream = stream_for(keys, "gdn_chunk_prepare");
  launch_kernel(gdn_chunk::prepare_kernel, dim3(coefficients.size(0), decay.size(1)), 256, 50688,
                stream, static_cast<const __nv_bfloat16 *>(keys.data_ptr()),
                static_cast<const float *>(decay.data_ptr()),
                static_cast<const float *>(beta.data_ptr()), starts.data_ptr(),
                starts.dtype().bits == 64, starts.size(0) - 1, keys.size(1), decay.size(1),
                keys.size(0), keys.stride(0), static_cast<__nv_bfloat16 *>(coefficients.data_ptr()),
                static_cast<float *>(cumulative_decay.data_ptr()),
                static_cast<int32_t *>(conditioning.data_ptr()));
  finish_cuda_launch(stream, "gdn_chunk_prepare");
}

void gdn_chunk_scan(TensorView q, TensorView k, TensorView v, TensorView initial, TensorView starts,
                    TensorView coefficients, TensorView cumulative_decay,
                    TensorView transformed_keys, TensorView transformed_values,
                    TensorView query_key_scores, TensorView carried_keys, TensorView output,
                    TensorView final, TensorView conditioning, TensorView decay, TensorView beta) {
  TVM_FFI_ICHECK(q.ndim() == 3 && q.size(0) > 0 && q.size(0) <= INT32_MAX && q.size(1) > 0 &&
                 v.ndim() == 3 && v.size(1) > 0 && v.size(1) <= 65535 &&
                 v.size(1) % q.size(1) == 0);
  validate_gdn_rows(q, q, q.size(0), q.size(1));
  validate_gdn_rows(k, q, q.size(0), q.size(1));
  validate_gdn_rows(v, q, q.size(0), v.size(1));
  validate_gdn_rows(output, q, q.size(0), v.size(1));
  TVM_FFI_ICHECK(output.stride(0) == v.size(1) * 128);
  validate_gdn_starts(starts, q);
  validate_gdn_states(initial, q, starts.size(0) - 1, v.size(1));
  validate_gdn_states(final, q, starts.size(0) - 1, v.size(1));
  const int64_t chunks = (q.size(0) + 63) / 64 + starts.size(0) - 2;
  validate_gdn_scratch(coefficients, cumulative_decay, q, chunks, v.size(1));
  validate_gdn_intermediates(transformed_keys, transformed_values, query_key_scores, carried_keys,
                             q, chunks, v.size(1));
  TVM_FFI_ICHECK(conditioning.ndim() == 2 && conditioning.size(0) == chunks &&
                 conditioning.size(1) == v.size(1) && has_dtype(conditioning, kDLInt, 32) &&
                 same_cuda_device(conditioning, q) && aligned(conditioning, 4) &&
                 conditioning.stride(0) == v.size(1) && conditioning.stride(1) == 1);
  for (auto gate : {decay, beta})
    TVM_FFI_ICHECK(same_cuda_device(gate, q) && gate.ndim() == 2 && gate.size(0) == q.size(0) &&
                   gate.size(1) == v.size(1) && has_dtype(gate, kDLFloat, 32) && aligned(gate, 4) &&
                   gate.stride(0) == v.size(1) && gate.stride(1) == 1);
  for (auto destination : {output, final})
    for (auto source :
         {q, k, v, initial, starts, coefficients, cumulative_decay, transformed_keys,
          transformed_values, query_key_scores, carried_keys, conditioning, decay, beta})
      TVM_FFI_ICHECK(disjoint_storage(destination, source));
  TVM_FFI_ICHECK(disjoint_storage(output, final));
  TVM_FFI_ICHECK(cudaFuncSetAttribute(gdn_chunk::scan_kernel,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      gdn_chunk::scan_shared_bytes) == cudaSuccess);
  auto stream = stream_for(q, "gdn_chunk_scan");
  launch_kernel(
      gdn_chunk::scan_kernel, dim3(starts.size(0) - 1, v.size(1), 128 / gdn_chunk::value_tile),
      gdn_chunk::value_tile / gdn_chunk::precise_values_per_group *
          gdn_chunk::precise_subgroup_lanes,
      gdn_chunk::scan_shared_bytes, stream, static_cast<const __nv_bfloat16 *>(q.data_ptr()),
      static_cast<const __nv_bfloat16 *>(k.data_ptr()),
      static_cast<const __nv_bfloat16 *>(v.data_ptr()),
      static_cast<const float *>(initial.data_ptr()), starts.data_ptr(), starts.dtype().bits == 64,
      q.size(1), v.size(1), starts.size(0) - 1, q.size(0), q.stride(0), k.stride(0), v.stride(0),
      static_cast<const __nv_bfloat16 *>(coefficients.data_ptr()),
      static_cast<const float *>(cumulative_decay.data_ptr()),
      static_cast<const __nv_bfloat16 *>(transformed_keys.data_ptr()),
      static_cast<const float *>(transformed_values.data_ptr()),
      static_cast<const __nv_bfloat16 *>(query_key_scores.data_ptr()),
      static_cast<const __nv_bfloat16 *>(carried_keys.data_ptr()),
      static_cast<__nv_bfloat16 *>(output.data_ptr()), static_cast<float *>(final.data_ptr()),
      static_cast<const int32_t *>(conditioning.data_ptr()),
      static_cast<const float *>(decay.data_ptr()), static_cast<const float *>(beta.data_ptr()));
  finish_cuda_launch(stream, "gdn_chunk_scan");
}

void gdn_chunk_project(TensorView q, TensorView k, TensorView v, TensorView starts,
                       TensorView coefficients, TensorView cumulative_decay,
                       TensorView transformed_keys, TensorView transformed_values,
                       TensorView query_key_scores, TensorView carried_keys,
                       TensorView conditioning) {
  TVM_FFI_ICHECK(q.ndim() == 3 && q.size(0) > 0 && q.size(0) <= INT32_MAX && q.size(1) > 0 &&
                 v.ndim() == 3 && v.size(1) > 0 && v.size(1) <= 65535 &&
                 v.size(1) % q.size(1) == 0);
  validate_gdn_rows(q, q, q.size(0), q.size(1));
  validate_gdn_rows(k, q, q.size(0), q.size(1));
  validate_gdn_rows(v, q, q.size(0), v.size(1));
  validate_gdn_starts(starts, q);
  const int64_t chunks = (q.size(0) + 63) / 64 + starts.size(0) - 2;
  validate_gdn_scratch(coefficients, cumulative_decay, q, chunks, v.size(1));
  validate_gdn_intermediates(transformed_keys, transformed_values, query_key_scores, carried_keys,
                             q, chunks, v.size(1));
  TVM_FFI_ICHECK(conditioning.ndim() == 2 && conditioning.size(0) == chunks &&
                 conditioning.size(1) == v.size(1) && has_dtype(conditioning, kDLInt, 32) &&
                 same_cuda_device(conditioning, q) && aligned(conditioning, 4) &&
                 conditioning.stride(0) == v.size(1) && conditioning.stride(1) == 1);
  const std::array<TensorView, 4> destinations = {transformed_keys, transformed_values,
                                                  query_key_scores, carried_keys};
  for (size_t i = 0; i < destinations.size(); ++i) {
    for (auto source : {q, k, v, starts, coefficients, cumulative_decay, conditioning})
      TVM_FFI_ICHECK(disjoint_storage(destinations[i], source));
    for (size_t j = 0; j < i; ++j)
      TVM_FFI_ICHECK(disjoint_storage(destinations[i], destinations[j]));
  }
  TVM_FFI_ICHECK(cudaFuncSetAttribute(gdn_chunk::project_kernel,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      79104) == cudaSuccess);
  auto stream = stream_for(q, "gdn_chunk_project");
  launch_kernel(gdn_chunk::project_kernel, dim3(chunks, v.size(1)), 256, 79104, stream,
                static_cast<const __nv_bfloat16 *>(q.data_ptr()),
                static_cast<const __nv_bfloat16 *>(k.data_ptr()),
                static_cast<const __nv_bfloat16 *>(v.data_ptr()), starts.data_ptr(),
                starts.dtype().bits == 64, starts.size(0) - 1, q.size(1), v.size(1), q.stride(0),
                k.stride(0), v.stride(0),
                static_cast<const __nv_bfloat16 *>(coefficients.data_ptr()),
                static_cast<const float *>(cumulative_decay.data_ptr()),
                static_cast<__nv_bfloat16 *>(transformed_keys.data_ptr()),
                static_cast<float *>(transformed_values.data_ptr()),
                static_cast<__nv_bfloat16 *>(query_key_scores.data_ptr()),
                static_cast<__nv_bfloat16 *>(carried_keys.data_ptr()), q.size(0),
                static_cast<const int32_t *>(conditioning.data_ptr()));
  finish_cuda_launch(stream, "gdn_chunk_project");
}
