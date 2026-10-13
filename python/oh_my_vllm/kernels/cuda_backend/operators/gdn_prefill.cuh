#pragma once

#include "common.cuh"

__device__ bool valid_gdn_offsets(const void *starts, bool wide, int sequences, int64_t tokens) {
  if (index_at(starts, wide, 0) != 0 || index_at(starts, wide, sequences) != tokens)
    return false;
  int64_t previous = 0;
  for (int sequence = 1; sequence <= sequences; ++sequence) {
    int64_t current = index_at(starts, wide, sequence);
    if (current < previous || current > tokens)
      return false;
    previous = current;
  }
  return true;
}

__global__ void gdn_prefill_kernel(const __nv_bfloat16 *__restrict__ q,
                                   const __nv_bfloat16 *__restrict__ k,
                                   const __nv_bfloat16 *__restrict__ v,
                                   const float *__restrict__ decay, const float *__restrict__ beta,
                                   const float *__restrict__ initial,
                                   const void *__restrict__ starts, bool wide_starts,
                                   __nv_bfloat16 *__restrict__ output, float *__restrict__ final,
                                   int query_heads, int value_heads, int sequences, int64_t tokens,
                                   int64_t qs, int64_t ks, int64_t vs) {
  pdl_dependency_wait();
  pdl_launch_next();
  if (!valid_gdn_offsets(starts, wide_starts, sequences, tokens)) {
    asm volatile("trap;");
    return;
  }
  const int sequence = blockIdx.x, head = blockIdx.y;
  const int lane = threadIdx.x % 32;
  const int first_value = blockIdx.z * 16 + threadIdx.x / 32 * 4;
  const int query_head = head / (value_heads / query_heads);
  const int64_t state_base = (int64_t(sequence) * value_heads + head) * 128 * 128;
  float state[4][4];
#pragma unroll
  for (int row = 0; row < 4; ++row)
#pragma unroll
    for (int column = 0; column < 4; ++column)
      state[row][column] = initial[state_base + (first_value + row) * 128 + lane + column * 32];
  for (int64_t token = index_at(starts, wide_starts, sequence);
       token < index_at(starts, wide_starts, sequence + 1); ++token) {
    float query[4], key[4];
#pragma unroll
    for (int column = 0; column < 4; ++column) {
      query[column] = __bfloat162float(q[token * qs + query_head * 128 + lane + column * 32]) *
                      0.08838834764831844f;
      key[column] = __bfloat162float(k[token * ks + query_head * 128 + lane + column * 32]);
    }
    const float multiplier = expf(decay[token * value_heads + head]);
    const float update = beta[token * value_heads + head];
#pragma unroll
    for (int row = 0; row < 4; ++row) {
      float recalled = 0;
#pragma unroll
      for (int column = 0; column < 4; ++column) {
        state[row][column] *= multiplier;
        recalled += state[row][column] * key[column];
      }
      recalled = warp_sum(recalled);
      float delta =
          (__bfloat162float(v[token * vs + head * 128 + first_value + row]) - recalled) * update;
      float result = 0;
#pragma unroll
      for (int column = 0; column < 4; ++column) {
        state[row][column] += delta * key[column];
        result += state[row][column] * query[column];
      }
      result = warp_sum(result);
      if (lane == 0)
        output[(token * value_heads + head) * 128 + first_value + row] =
            __float2bfloat16_rn(result);
    }
  }
#pragma unroll
  for (int row = 0; row < 4; ++row)
#pragma unroll
    for (int column = 0; column < 4; ++column)
      final[state_base + (first_value + row) * 128 + lane + column * 32] = state[row][column];
}

void gdn_prefill(TensorView q, TensorView k, TensorView v, TensorView decay, TensorView beta,
                 TensorView initial, TensorView starts, TensorView output, TensorView final) {
  for (auto tensor : {q, k, v, decay, beta, initial, starts, output, final})
    TVM_FFI_ICHECK(same_cuda_device(tensor, q));
  TVM_FFI_ICHECK(q.ndim() == 3 && k.ndim() == 3 && v.ndim() == 3 && q.size(0) > 0 &&
                 q.size(1) > 0 && q.size(2) == 128 && k.size(0) == q.size(0) &&
                 k.size(1) == q.size(1) && k.size(2) == 128 && v.size(0) == q.size(0) &&
                 v.size(1) > 0 && v.size(1) % q.size(1) == 0 && v.size(2) == 128);
  TVM_FFI_ICHECK(q.size(0) <= INT32_MAX && v.size(1) <= 65535);
  for (auto tensor : {q, k, v})
    TVM_FFI_ICHECK(has_dtype(tensor, kDLBfloat, 16) && tensor.stride(2) == 1 &&
                   tensor.stride(1) == 128 && tensor.stride(0) >= tensor.size(1) * 128);
  TVM_FFI_ICHECK(starts.ndim() == 1 && starts.size(0) > 1 && starts.size(0) <= INT32_MAX &&
                 is_index_dtype(starts) && starts.stride(0) == 1);
  for (auto tensor : {initial, final})
    TVM_FFI_ICHECK(tensor.ndim() == 4 && tensor.size(0) == starts.size(0) - 1 &&
                   tensor.size(1) == v.size(1) && tensor.size(2) == 128 && tensor.size(3) == 128 &&
                   has_dtype(tensor, kDLFloat, 32) && tensor.stride(3) == 1 &&
                   tensor.stride(2) == 128 && tensor.stride(1) == 128 * 128 &&
                   tensor.stride(0) == v.size(1) * 128 * 128);
  for (auto tensor : {decay, beta})
    TVM_FFI_ICHECK(tensor.ndim() == 2 && tensor.size(0) == q.size(0) &&
                   tensor.size(1) == v.size(1) && has_dtype(tensor, kDLFloat, 32) &&
                   tensor.stride(1) == 1 && tensor.stride(0) == v.size(1));
  TVM_FFI_ICHECK(output.ndim() == 3 && output.size(0) == v.size(0) && output.size(1) == v.size(1) &&
                 output.size(2) == 128 && has_dtype(output, kDLBfloat, 16) &&
                 output.stride(2) == 1 && output.stride(1) == 128 &&
                 output.stride(0) == v.size(1) * 128);
  for (auto destination : {output, final})
    for (auto source : {q, k, v, decay, beta, initial, starts})
      TVM_FFI_ICHECK(disjoint_storage(destination, source));
  TVM_FFI_ICHECK(disjoint_storage(output, final));
  auto stream = stream_for(q, "gdn_prefill");
  launch_kernel(
      gdn_prefill_kernel, dim3(starts.size(0) - 1, v.size(1), 8), 128, 0, stream,
      static_cast<const __nv_bfloat16 *>(q.data_ptr()),
      static_cast<const __nv_bfloat16 *>(k.data_ptr()),
      static_cast<const __nv_bfloat16 *>(v.data_ptr()),
      static_cast<const float *>(decay.data_ptr()), static_cast<const float *>(beta.data_ptr()),
      static_cast<const float *>(initial.data_ptr()), starts.data_ptr(), starts.dtype().bits == 64,
      static_cast<__nv_bfloat16 *>(output.data_ptr()), static_cast<float *>(final.data_ptr()),
      q.size(1), v.size(1), starts.size(0) - 1, q.size(0), q.stride(0), k.stride(0), v.stride(0));
  finish_cuda_launch(stream, "gdn_prefill");
}
