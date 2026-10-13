#pragma once

#include "gdn_contract.cuh"
#include <mma.h>

namespace gdn_chunk {

constexpr int chunk_size = 64;
constexpr int value_tile = 16;
constexpr int key_pitch = 136;
constexpr int value_pitch = 24;
constexpr int transform_pitch = 72;
constexpr int scan_shared_bytes = 108032;

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

__global__ void prepare_kernel(const __nv_bfloat16 *__restrict__ keys,
                               const float *__restrict__ decay, const float *__restrict__ beta,
                               const void *__restrict__ starts, bool wide, int sequences,
                               int query_heads, int value_heads, int64_t tokens, int64_t key_stride,
                               __nv_bfloat16 *__restrict__ coefficients,
                               float *__restrict__ cumulative_decay) {
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
  auto lower = reinterpret_cast<float *>(storage + 16384);
  auto inverse = reinterpret_cast<float *>(storage + 32768);
  auto log_decay = reinterpret_cast<float *>(storage + 49152);
  auto update = reinterpret_cast<float *>(storage + 49408);
  for (int i = threadIdx.x; i < 64 * 128; i += blockDim.x) {
    int row = i / 128, column = i % 128;
    key[i] = row < count ? keys[(first + row) * key_stride + query_head * 128 + column]
                         : __float2bfloat16(0.0f);
  }
  if (threadIdx.x == 0) {
    float total = 0;
    for (int row = 0; row < 64; ++row) {
      if (row < count)
        total += decay[(first + row) * value_heads + head];
      log_decay[row] = total;
      update[row] = row < count ? beta[(first + row) * value_heads + head] : 0.0f;
    }
  }
  __syncthreads();
  multiply<64, 64, 128, false, true>(key, 128, key, 128, lower, 64);
  for (int i = threadIdx.x; i < 64 * 64; i += blockDim.x) {
    int row = i / 64, column = i % 64;
    lower[i] =
        row > column ? lower[i] * update[row] * expf(log_decay[row] - log_decay[column]) : 0.0f;
    inverse[i] = row == column ? 1.0f : 0.0f;
  }
  __syncthreads();
  // Solve the unit triangular system in FP32 before its BF16 tensor-core boundary.
  for (int row = 1; row < 64; ++row) {
    if (threadIdx.x < row) {
      int column = threadIdx.x;
      float value = -lower[row * 64 + column];
      for (int inner = column + 1; inner < row; ++inner)
        value -= lower[row * 64 + inner] * inverse[inner * 64 + column];
      inverse[row * 64 + column] = value;
    }
    __syncthreads();
  }
  const int64_t base = (int64_t(chunk) * value_heads + head) * 64;
  for (int i = threadIdx.x; i < 64 * 64; i += blockDim.x)
    coefficients[base * 64 + i] = __float2bfloat16_rn(inverse[i] * update[i % 64]);
  if (threadIdx.x < 64)
    cumulative_decay[base + threadIdx.x] = log_decay[threadIdx.x];
}

__global__ void scan_kernel(const __nv_bfloat16 *__restrict__ queries,
                            const __nv_bfloat16 *__restrict__ keys,
                            const __nv_bfloat16 *__restrict__ values,
                            const float *__restrict__ initial, const void *__restrict__ starts,
                            bool wide, int query_heads, int value_heads, int sequences,
                            int64_t tokens, int64_t qs, int64_t ks, int64_t vs,
                            const __nv_bfloat16 *__restrict__ coefficients,
                            const float *__restrict__ cumulative_decay,
                            __nv_bfloat16 *__restrict__ output, float *__restrict__ final) {
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
  auto state_bf16 = reinterpret_cast<__nv_bfloat16 *>(storage + 8704);
  auto a = reinterpret_cast<__nv_bfloat16 *>(storage + 13056);
  auto b = reinterpret_cast<__nv_bfloat16 *>(storage + 30464);
  auto transform = reinterpret_cast<__nv_bfloat16 *>(storage + 47872);
  auto score = reinterpret_cast<float *>(storage + 47872);
  auto temporary = reinterpret_cast<float *>(storage + 66304);
  auto delta = reinterpret_cast<float *>(storage + 101120);
  auto log_decay = reinterpret_cast<float *>(storage + 107264);
  auto decayed_rows = reinterpret_cast<float *>(storage + 107520);
  auto final_rows = reinterpret_cast<float *>(storage + 107776);
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
    if (threadIdx.x < 64) {
      log_decay[threadIdx.x] = cumulative_decay[base + threadIdx.x];
      decayed_rows[threadIdx.x] = expf(cumulative_decay[base + threadIdx.x]);
      final_rows[threadIdx.x] =
          expf(cumulative_decay[base + count - 1] - cumulative_decay[base + threadIdx.x]);
    }
    for (int i = threadIdx.x; i < 64 * 64; i += blockDim.x)
      transform[i / 64 * transform_pitch + i % 64] = coefficients[base * 64 + i];
    __syncthreads();
    for (int i = threadIdx.x; i < 64 * 128; i += blockDim.x) {
      int row = i / 128, column = i % 128, offset = row * key_pitch + column;
      float key = row < count
                      ? __bfloat162float(keys[(position + row) * ks + query_head * 128 + column])
                      : 0.0f;
      a[offset] = __float2bfloat16_rn(key * decayed_rows[row]);
      b[offset] = row < count && column < value_tile
                      ? values[(position + row) * vs + head * 128 + value_offset + column]
                      : __float2bfloat16(0.0f);
    }
    __syncthreads();
    multiply<64, 128, 64>(transform, transform_pitch, a, key_pitch, temporary, key_pitch);
    for (int i = threadIdx.x; i < 64 * 128; i += blockDim.x) {
      int offset = i / 128 * key_pitch + i % 128;
      a[offset] = __float2bfloat16_rn(-temporary[offset]);
    }
    __syncthreads();
    multiply<64, value_tile, 64>(transform, transform_pitch, b, key_pitch, delta, value_pitch);
    multiply<64, value_tile, 128, false, true>(a, key_pitch, state_bf16, key_pitch, delta,
                                               value_pitch, true);
    for (int i = threadIdx.x; i < 64 * 128; i += blockDim.x) {
      int row = i / 128, column = i % 128, offset = row * key_pitch + column;
      a[offset] = row < count ? queries[(position + row) * qs + query_head * 128 + column]
                              : __float2bfloat16(0.0f);
      b[offset] = row < count ? keys[(position + row) * ks + query_head * 128 + column]
                              : __float2bfloat16(0.0f);
    }
    __syncthreads();
    multiply<64, value_tile, 128, false, true>(a, key_pitch, state_bf16, key_pitch, temporary,
                                               value_pitch);
    for (int i = threadIdx.x; i < 64 * value_tile; i += blockDim.x)
      temporary[i / value_tile * value_pitch + i % value_tile] *=
          decayed_rows[i / value_tile] * 0.08838834764831844f;
    multiply<64, 64, 128, false, true>(a, key_pitch, b, key_pitch, score, transform_pitch);
    for (int i = threadIdx.x; i < 64 * 64; i += blockDim.x) {
      int row = i / 64, column = i % 64;
      float attention = row < count && column <= row
                            ? score[row * transform_pitch + column] *
                                  expf(log_decay[row] - log_decay[column]) * 0.08838834764831844f
                            : 0.0f;
      a[row * transform_pitch + column] = __float2bfloat16_rn(attention);
    }
    for (int i = threadIdx.x; i < 64 * value_tile; i += blockDim.x)
      b[i / value_tile * key_pitch + i % value_tile] =
          __float2bfloat16_rn(delta[i / value_tile * value_pitch + i % value_tile]);
    __syncthreads();
    multiply<64, value_tile, 64>(a, transform_pitch, b, key_pitch, temporary, value_pitch, true);
    for (int i = threadIdx.x; i < count * value_tile; i += blockDim.x) {
      int row = i / value_tile, column = i % value_tile;
      output[((position + row) * value_heads + head) * 128 + value_offset + column] =
          __float2bfloat16_rn(temporary[row * value_pitch + column]);
    }
    for (int i = threadIdx.x; i < 64 * 128; i += blockDim.x) {
      int row = i / 128, column = i % 128;
      float key = row < count
                      ? __bfloat162float(keys[(position + row) * ks + query_head * 128 + column])
                      : 0.0f;
      a[row * key_pitch + column] = __float2bfloat16_rn(key * final_rows[row]);
    }
    __syncthreads();
    multiply<value_tile, 128, 64, true, false>(b, key_pitch, a, key_pitch, temporary, key_pitch);
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
                       TensorView coefficients, TensorView cumulative_decay) {
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
  for (auto destination : {coefficients, cumulative_decay})
    for (auto source : {keys, decay, beta, starts})
      TVM_FFI_ICHECK(disjoint_storage(destination, source));
  TVM_FFI_ICHECK(cudaFuncSetAttribute(gdn_chunk::prepare_kernel,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      49664) == cudaSuccess);
  auto stream = stream_for(keys, "gdn_chunk_prepare");
  launch_kernel(gdn_chunk::prepare_kernel, dim3(coefficients.size(0), decay.size(1)), 256, 49664,
                stream, static_cast<const __nv_bfloat16 *>(keys.data_ptr()),
                static_cast<const float *>(decay.data_ptr()),
                static_cast<const float *>(beta.data_ptr()), starts.data_ptr(),
                starts.dtype().bits == 64, starts.size(0) - 1, keys.size(1), decay.size(1),
                keys.size(0), keys.stride(0), static_cast<__nv_bfloat16 *>(coefficients.data_ptr()),
                static_cast<float *>(cumulative_decay.data_ptr()));
  finish_cuda_launch(stream, "gdn_chunk_prepare");
}

void gdn_chunk_scan(TensorView q, TensorView k, TensorView v, TensorView initial, TensorView starts,
                    TensorView coefficients, TensorView cumulative_decay, TensorView output,
                    TensorView final) {
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
  for (auto destination : {output, final})
    for (auto source : {q, k, v, initial, starts, coefficients, cumulative_decay})
      TVM_FFI_ICHECK(disjoint_storage(destination, source));
  TVM_FFI_ICHECK(disjoint_storage(output, final));
  TVM_FFI_ICHECK(cudaFuncSetAttribute(gdn_chunk::scan_kernel,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      gdn_chunk::scan_shared_bytes) == cudaSuccess);
  auto stream = stream_for(q, "gdn_chunk_scan");
  launch_kernel(
      gdn_chunk::scan_kernel, dim3(starts.size(0) - 1, v.size(1), 128 / gdn_chunk::value_tile), 128,
      gdn_chunk::scan_shared_bytes, stream, static_cast<const __nv_bfloat16 *>(q.data_ptr()),
      static_cast<const __nv_bfloat16 *>(k.data_ptr()),
      static_cast<const __nv_bfloat16 *>(v.data_ptr()),
      static_cast<const float *>(initial.data_ptr()), starts.data_ptr(), starts.dtype().bits == 64,
      q.size(1), v.size(1), starts.size(0) - 1, q.size(0), q.stride(0), k.stride(0), v.stride(0),
      static_cast<const __nv_bfloat16 *>(coefficients.data_ptr()),
      static_cast<const float *>(cumulative_decay.data_ptr()),
      static_cast<__nv_bfloat16 *>(output.data_ptr()), static_cast<float *>(final.data_ptr()));
  finish_cuda_launch(stream, "gdn_chunk_scan");
}
