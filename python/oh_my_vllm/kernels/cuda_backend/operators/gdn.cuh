#pragma once

#include "common.cuh"

__global__ void gates_kernel(const __nv_bfloat16 *__restrict__ ba, const float *__restrict__ log,
                             const float *__restrict__ bias, float *__restrict__ decay,
                             float *__restrict__ beta, int n) {
  pdl_dependency_wait();
  pdl_launch_next();
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n * 48)
    return;
  int head = i % 48, offset = i / 48 * 96 + head;
  float b = __bfloat162float(ba[offset]);
  float a = __bfloat162float(ba[offset + 48]) + bias[head];
  float softplus = a > 20.f ? a : log1pf(expf(a));
  decay[i] = -expf(log[head]) * softplus;
  float e = __expf(-fabsf(b));
  beta[i] = __fdividef(b >= 0.f ? 1.f : e, 1.f + e);
}
void gates(TensorView ba, TensorView log, TensorView bias, TensorView decay, TensorView beta) {
  TVM_FFI_ICHECK(same_cuda_device(ba, ba) && same_cuda_device(log, ba) &&
                 same_cuda_device(bias, ba) && same_cuda_device(decay, ba) &&
                 same_cuda_device(beta, ba))
      << "CUDA gates tensors must share one CUDA device";
  TVM_FFI_ICHECK(ba.ndim() == 2 && ba.size(1) == 96 && log.ndim() == 1 && log.size(0) == 48 &&
                 bias.ndim() == 1 && bias.size(0) == 48 && decay.ndim() == 2 &&
                 decay.size(0) == ba.size(0) && decay.size(1) == 48 && beta.ndim() == 2 &&
                 beta.size(0) == ba.size(0) && beta.size(1) == 48 && has_dtype(ba, kDLBfloat, 16) &&
                 has_dtype(log, kDLFloat, 32) && has_dtype(bias, kDLFloat, 32) &&
                 has_dtype(decay, kDLFloat, 32) && has_dtype(beta, kDLFloat, 32) &&
                 ba.stride(1) == 1 && ba.stride(0) == 96 && log.stride(0) == 1 &&
                 bias.stride(0) == 1 && decay.stride(1) == 1 && decay.stride(0) == 48 &&
                 beta.stride(1) == 1 && beta.stride(0) == 48)
      << "CUDA gates requires BF16 projection and FP32 [rows,48] outputs";
  if (ba.size(0) == 0)
    return;
  auto stream = stream_for(ba, "gates");
  // Medium rows benefit from twice as many blocks for the transcendental work.
  int threads = ba.size(0) >= 128 && ba.size(0) < 4096 ? 128 : 256;
  launch_kernel(gates_kernel, (ba.size(0) * 48 + threads - 1) / threads, threads, 0, stream,
                static_cast<const __nv_bfloat16 *>(ba.data_ptr()),
                static_cast<const float *>(log.data_ptr()),
                static_cast<const float *>(bias.data_ptr()), static_cast<float *>(decay.data_ptr()),
                static_cast<float *>(beta.data_ptr()), ba.size(0));
  finish_cuda_launch(stream, "gates");
}

__device__ float half_warp_sum(float value) {
#pragma unroll
  for (int offset = 8; offset; offset >>= 1)
    value += __shfl_xor_sync(0xffffffff, value, offset, 16);
  return value;
}

template <int Threads, bool Joint>
__global__ void __launch_bounds__(Threads, 1)
    model_qk_kernel(const __nv_bfloat16 *__restrict__ q, const __nv_bfloat16 *__restrict__ k,
                    __nv_bfloat16 *__restrict__ oq, __nv_bfloat16 *__restrict__ ok) {
  pdl_dependency_wait();
  pdl_launch_next();
  static_assert(Threads == 128 || Threads == 256);
  // Fixed16-head rows fill every CTA, so no partial-head mask is needed.

  int64_t input_offset = (int64_t(blockIdx.x) * (Threads) / 256) * 10240 +
                         ((int64_t(blockIdx.x) * (Threads)) % 256) * 8 + int64_t(threadIdx.x) * 8;
  auto a = *reinterpret_cast<const AlignedVector<__nv_bfloat16, 8> *>(q + input_offset);
  auto b = *reinterpret_cast<const AlignedVector<__nv_bfloat16, 8> *>(k + input_offset);
  float2 av[4], bv[4], as = make_float2(0, 0), bs = make_float2(0, 0);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    av[j] = __bfloat1622float2(reinterpret_cast<__nv_bfloat162 *>(a.value)[j]);
    bv[j] = __bfloat1622float2(reinterpret_cast<__nv_bfloat162 *>(b.value)[j]);
    auto aq = __fmul2_rn(av[j], av[j]);
    auto bk = __fmul2_rn(bv[j], bv[j]);
    // The first squared values are nonnegative, including positive zero.
    // Avoid the initial add to zero on the latency-sensitive short-row path.
    if (Threads == 128 && j == 0) {
      as = aq;
      bs = bk;
    } else {
      as = __fadd2_rn(as, aq);
      bs = __fadd2_rn(bs, bk);
    }
  }
  float2 sums = make_float2(as.x + as.y, bs.x + bs.y);
  if constexpr (Joint) {
#pragma unroll
    for (int i = 8; i; i >>= 1)
      sums = __fadd2_rn(sums, make_float2(__shfl_xor_sync(0xffffffff, sums.x, i, 16),
                                          __shfl_xor_sync(0xffffffff, sums.y, i, 16)));
  } else {
    sums.x = half_warp_sum(sums.x);
    sums.y = half_warp_sum(sums.y);
  }
  float ai = rsqrtf(sums.x + 1e-6f), bi = rsqrtf(sums.y + 1e-6f);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    reinterpret_cast<__nv_bfloat162 *>(a.value)[j] =
        __float22bfloat162_rn(__fmul2_rn(av[j], make_float2(ai, ai)));
    reinterpret_cast<__nv_bfloat162 *>(b.value)[j] =
        __float22bfloat162_rn(__fmul2_rn(bv[j], make_float2(bi, bi)));
  }
  *reinterpret_cast<AlignedVector<__nv_bfloat16, 8> *>(oq + int64_t(blockIdx.x) * Threads * 8 +
                                                       threadIdx.x * 8) = a;
  *reinterpret_cast<AlignedVector<__nv_bfloat16, 8> *>(ok + int64_t(blockIdx.x) * Threads * 8 +
                                                       threadIdx.x * 8) = b;
}
__global__ void qk_kernel(const __nv_bfloat16 *__restrict__ q, const __nv_bfloat16 *__restrict__ k,
                          __nv_bfloat16 *__restrict__ oq, __nv_bfloat16 *__restrict__ ok, int rows,
                          int heads, int64_t qs, int64_t ks) {
  pdl_dependency_wait();
  pdl_launch_next();
  int row = blockIdx.x * 4 + threadIdx.x / 32, lane = threadIdx.x & 31;
  if (row >= rows)
    return;
  float qv[4], kv[4], qsum = 0, ksum = 0;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    qv[j] = __bfloat162float(q[row / heads * qs + row % heads * 128 + lane + j * 32]);
    kv[j] = __bfloat162float(k[row / heads * ks + row % heads * 128 + lane + j * 32]);
    qsum += qv[j] * qv[j];
    ksum += kv[j] * kv[j];
  }
  float qi = rsqrtf(warp_sum(qsum) + 1e-6f), ki = rsqrtf(warp_sum(ksum) + 1e-6f);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    oq[static_cast<int64_t>(row) * 128 + lane + j * 32] = __float2bfloat16_rn(qv[j] * qi);
    ok[static_cast<int64_t>(row) * 128 + lane + j * 32] = __float2bfloat16_rn(kv[j] * ki);
  }
}
void normalize_qk(TensorView q, TensorView k, TensorView oq, TensorView ok) {
  bool packed = q.size(1) == 16 && q.stride(0) == 10240 && k.stride(0) == 10240 &&
                reinterpret_cast<uintptr_t>(q.data_ptr()) % 16 == 0 &&
                reinterpret_cast<uintptr_t>(k.data_ptr()) % 16 == 0;
  auto stream = stream_for(q, "normalize_qk");
  if (packed) {
#define MODEL_QK(T, J)                                                                             \
  launch_kernel(model_qk_kernel<T, J>, q.size(0) * (256 / T), T, 0, stream,                        \
                static_cast<const __nv_bfloat16 *>(q.data_ptr()),                                  \
                static_cast<const __nv_bfloat16 *>(k.data_ptr()),                                  \
                static_cast<__nv_bfloat16 *>(oq.data_ptr()),                                       \
                static_cast<__nv_bfloat16 *>(ok.data_ptr()))
    if (q.size(0) < 1024) {
      MODEL_QK(128, true);
    } else if (q.size(0) < 4096) {
      MODEL_QK(256, false);
    } else {
      MODEL_QK(256, true);
    }
#undef MODEL_QK
    finish_cuda_launch(stream, "normalize_qk");
    record_variant(kQk, true);
    return;
  }
  int rows = q.size(0) * q.size(1);
  launch_kernel(
      qk_kernel, (rows + 3) / 4, 128, 0, stream, static_cast<const __nv_bfloat16 *>(q.data_ptr()),
      static_cast<const __nv_bfloat16 *>(k.data_ptr()), static_cast<__nv_bfloat16 *>(oq.data_ptr()),
      static_cast<__nv_bfloat16 *>(ok.data_ptr()), rows, q.size(1), q.stride(0), k.stride(0));
  finish_cuda_launch(stream, "normalize_qk");
  record_variant(kQk, false);
}
template <typename State, int Rows, int Warps>
__global__ void
recurrent_vector_kernel(const __nv_bfloat16 *__restrict__ q, const __nv_bfloat16 *__restrict__ k,
                        const __nv_bfloat16 *__restrict__ v, const float *__restrict__ decay,
                        const float *__restrict__ beta, State *__restrict__ pool,
                        const void *__restrict__ starts, const void *__restrict__ reads,
                        const void *__restrict__ writes, bool sw, bool rw, bool ww,
                        __nv_bfloat16 *__restrict__ out, int64_t qs, int64_t ks, int64_t vs) {
  pdl_dependency_wait();
  pdl_launch_next();
  static_assert(128 % (Rows * Warps * 2) == 0);
  constexpr int hq = 16, hv = 48;
  int seq = blockIdx.x, head = blockIdx.y, lane = threadIdx.x & 15;
  int first_v = blockIdx.z * (Rows * Warps * 2) + threadIdx.x / 16 * Rows;
  int qhead = head / (hv / hq);
  int64_t source = index_at(reads, rw, seq);
  float state[Rows][8];
#pragma unroll
  for (int i = 0; i < Rows; ++i) {
    auto v = *reinterpret_cast<const AlignedVector<State, 8> *>(
        pool + ((source * hv + head) * 128 + first_v + i) * 128 + lane * 8);
#pragma unroll
    for (int j = 0; j < 8; ++j)
      state[i][j] = static_cast<float>(v.value[j]);
  }
  // Persistent FP32 values survive all candidates; only destination snapshots round.
  for (int token = index_at(starts, sw, seq); token < index_at(starts, sw, seq + 1); ++token) {
    float qv[8], kv[8], qq = 0, kk = 0;
    auto qpack = *reinterpret_cast<const AlignedVector<__nv_bfloat16, 8> *>(q + token * qs +
                                                                            qhead * 128 + lane * 8);
    auto kpack = *reinterpret_cast<const AlignedVector<__nv_bfloat16, 8> *>(k + token * ks +
                                                                            qhead * 128 + lane * 8);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      qv[j] = __bfloat162float(qpack.value[j]);
      kv[j] = __bfloat162float(kpack.value[j]);
      qq += qv[j] * qv[j];
      kk += kv[j] * kv[j];
    }
    float qi = rsqrtf(half_warp_sum(qq) + 1e-6f) * 0.08838834764831844f;
    float ki = rsqrtf(half_warp_sum(kk) + 1e-6f);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      qv[j] *= qi;
      kv[j] *= ki;
    }
    float g = expf(decay[token * hv + head]), b = beta[token * hv + head];
    int64_t target = index_at(writes, ww, token);
#pragma unroll
    for (int i = 0; i < Rows; ++i) {
      float recall = 0;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        state[i][j] *= g;
        recall += state[i][j] * kv[j];
      }
      recall = half_warp_sum(recall);
      float residual = (__bfloat162float(v[token * vs + head * 128 + first_v + i]) - recall) * b;
      float result = 0;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        state[i][j] += residual * kv[j];
        result += state[i][j] * qv[j];
      }
      if (target >= 0) {
        AlignedVector<State, 8> snapshot;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          if constexpr (std::is_same_v<State, __nv_bfloat16>)
            reinterpret_cast<__nv_bfloat162 *>(snapshot.value)[j] =
                __floats2bfloat162_rn(state[i][j * 2], state[i][j * 2 + 1]);
          else {
            snapshot.value[j * 2] = state[i][j * 2];
            snapshot.value[j * 2 + 1] = state[i][j * 2 + 1];
          }
        }
        *reinterpret_cast<AlignedVector<State, 8> *>(
            pool + ((target * hv + head) * 128 + first_v + i) * 128 + lane * 8) = snapshot;
      }
      result = half_warp_sum(result);
      if (lane == 0)
        out[(static_cast<int64_t>(token) * hv + head) * 128 + first_v + i] =
            __float2bfloat16_rn(result);
    }
  }
}
template <typename State>
__global__ void recurrent_kernel(const __nv_bfloat16 *q, const __nv_bfloat16 *k,
                                 const __nv_bfloat16 *v, const float *decay, const float *beta,
                                 State *pool, const void *starts, const void *reads,
                                 const void *writes, bool sw, bool rw, bool ww, __nv_bfloat16 *out,
                                 int hq, int hv, int64_t qs, int64_t ks, int64_t vs) {
  pdl_dependency_wait();
  pdl_launch_next();
  int seq = blockIdx.x, head = blockIdx.y, lane = threadIdx.x & 31;
  int first_v = blockIdx.z * 16 + threadIdx.x / 32 * 4;
  int qhead = head / (hv / hq);
  int64_t source = index_at(reads, rw, seq);
  float state[4][4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j)
      state[i][j] = static_cast<float>(
          pool[((source * hv + head) * 128 + first_v + i) * 128 + lane + j * 32]);
  // Persistent FP32 values survive all candidates; only destination snapshots round.
  for (int token = index_at(starts, sw, seq); token < index_at(starts, sw, seq + 1); ++token) {
    float qv[4], kv[4], qq = 0, kk = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      qv[j] = __bfloat162float(q[token * qs + qhead * 128 + lane + j * 32]);
      kv[j] = __bfloat162float(k[token * ks + qhead * 128 + lane + j * 32]);
      qq += qv[j] * qv[j];
      kk += kv[j] * kv[j];
    }
    float qi = rsqrtf(warp_sum(qq) + 1e-6f) * 0.08838834764831844f;
    float ki = rsqrtf(warp_sum(kk) + 1e-6f);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      qv[j] *= qi;
      kv[j] *= ki;
    }
    float g = expf(decay[token * hv + head]), b = beta[token * hv + head];
    int64_t target = index_at(writes, ww, token);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      float recall = 0;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        state[i][j] *= g;
        recall += state[i][j] * kv[j];
      }
      recall = warp_sum(recall);
      float residual = (__bfloat162float(v[token * vs + head * 128 + first_v + i]) - recall) * b;
      float result = 0;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        state[i][j] += residual * kv[j];
        result += state[i][j] * qv[j];
        if (target >= 0)
          pool[((target * hv + head) * 128 + first_v + i) * 128 + lane + j * 32] =
              static_cast<State>(state[i][j]);
      }
      result = warp_sum(result);
      if (lane == 0)
        out[(static_cast<int64_t>(token) * hv + head) * 128 + first_v + i] =
            __float2bfloat16_rn(result);
    }
  }
}
void recurrent(TensorView q, TensorView k, TensorView v, TensorView decay, TensorView beta,
               TensorView pool, TensorView starts, TensorView reads, TensorView writes,
               TensorView out) {
  // Caller guarantees starts=[0,...,q.size(0)] is monotone, reads are in
  // [0,pool.size(0)), and writes are -1 (skip) or in that range. Their values
  // stay on GPU to avoid a hot-path sync.
  TVM_FFI_ICHECK(same_cuda_device(q, q) && same_cuda_device(k, q) && same_cuda_device(v, q) &&
                 same_cuda_device(decay, q) && same_cuda_device(beta, q) &&
                 same_cuda_device(pool, q) && same_cuda_device(starts, q) &&
                 same_cuda_device(reads, q) && same_cuda_device(writes, q) &&
                 same_cuda_device(out, q))
      << "CUDA recurrence tensors must share one CUDA device";
  TVM_FFI_ICHECK(
      q.ndim() == 3 && k.ndim() == 3 && v.ndim() == 3 && q.size(0) > 0 && q.size(1) > 0 &&
      q.size(2) == 128 && k.size(0) == q.size(0) && k.size(1) == q.size(1) && k.size(2) == 128 &&
      v.size(0) == q.size(0) && v.size(1) > 0 && v.size(2) == 128 && v.size(1) % q.size(1) == 0 &&
      has_dtype(q, kDLBfloat, 16) && has_dtype(k, kDLBfloat, 16) && has_dtype(v, kDLBfloat, 16) &&
      q.stride(2) == 1 && q.stride(1) == 128 && k.stride(2) == 1 && k.stride(1) == 128 &&
      v.stride(2) == 1 && v.stride(1) == 128 && q.stride(0) >= q.size(1) * 128 &&
      k.stride(0) >= k.size(1) * 128 && v.stride(0) >= v.size(1) * 128)
      << "CUDA recurrence requires nonempty BF16 q/k/v rows";
  TVM_FFI_ICHECK(decay.ndim() == 2 && beta.ndim() == 2 && decay.size(0) == q.size(0) &&
                 decay.size(1) == v.size(1) && beta.size(0) == q.size(0) &&
                 beta.size(1) == v.size(1) && has_dtype(decay, kDLFloat, 32) &&
                 has_dtype(beta, kDLFloat, 32) && decay.stride(1) == 1 &&
                 decay.stride(0) == v.size(1) && beta.stride(1) == 1 && beta.stride(0) == v.size(1))
      << "CUDA recurrence requires contiguous FP32 gates";
  TVM_FFI_ICHECK(starts.ndim() == 1 && reads.ndim() == 1 && writes.ndim() == 1 &&
                 reads.size(0) > 0 && starts.size(0) == reads.size(0) + 1 &&
                 writes.size(0) == q.size(0) && is_index_dtype(starts) && is_index_dtype(reads) &&
                 is_index_dtype(writes) && starts.stride(0) == 1 && reads.stride(0) == 1 &&
                 writes.stride(0) == 1)
      << "CUDA recurrence requires nonempty int32/int64 metadata";
  const auto state_dtype = pool.dtype();
  const bool bf16_state =
      state_dtype.code == kDLBfloat && state_dtype.bits == 16 && state_dtype.lanes == 1;
  const bool fp32_state =
      state_dtype.code == kDLFloat && state_dtype.bits == 32 && state_dtype.lanes == 1;
  TVM_FFI_ICHECK(bf16_state || fp32_state) << "CUDA recurrence state requires BF16 or FP32";
  TVM_FFI_ICHECK(pool.ndim() == 4 && pool.size(0) > 0 && pool.size(1) == v.size(1) &&
                 pool.size(2) == 128 && pool.size(3) == 128 && pool.stride(3) == 1 &&
                 pool.stride(2) == 128 && pool.stride(1) == 128 * 128 &&
                 pool.stride(0) == v.size(1) * 128 * 128)
      << "CUDA recurrence state shape or layout is invalid";
  TVM_FFI_ICHECK(out.ndim() == 3 && out.size(0) == v.size(0) && out.size(1) == v.size(1) &&
                 out.size(2) == 128 && has_dtype(out, kDLBfloat, 16) && out.stride(2) == 1 &&
                 out.stride(1) == 128 && out.stride(0) == v.size(1) * 128)
      << "CUDA recurrence output requires contiguous BF16 value shape";
  const auto state_alignment =
      bf16_state ? alignof(AlignedVector<__nv_bfloat16, 8>) : alignof(AlignedVector<float, 8>);
  bool vector = q.size(1) == 16 && v.size(1) == 48 && q.stride(0) % 8 == 0 &&
                k.stride(0) % 8 == 0 && reinterpret_cast<uintptr_t>(q.data_ptr()) % 16 == 0 &&
                reinterpret_cast<uintptr_t>(k.data_ptr()) % 16 == 0 &&
                reinterpret_cast<uintptr_t>(pool.data_ptr()) % state_alignment == 0;
  for (auto input : {q, k, v, decay, beta, starts, reads, writes})
    vector = vector && disjoint_storage(pool, input);
  auto stream = stream_for(q, "recurrent");
  if (vector) {
#define VECTOR_REC(S, R, W)                                                                        \
  launch_kernel(                                                                                   \
      recurrent_vector_kernel<S, R, W>, dim3(reads.size(0), 48, 128 / (R * W * 2)), W * 32, 0,     \
      stream, static_cast<const __nv_bfloat16 *>(q.data_ptr()),                                    \
      static_cast<const __nv_bfloat16 *>(k.data_ptr()),                                            \
      static_cast<const __nv_bfloat16 *>(v.data_ptr()),                                            \
      static_cast<const float *>(decay.data_ptr()), static_cast<const float *>(beta.data_ptr()),   \
      static_cast<S *>(pool.data_ptr()), starts.data_ptr(), reads.data_ptr(), writes.data_ptr(),   \
      starts.dtype().bits == 64, reads.dtype().bits == 64, writes.dtype().bits == 64,              \
      static_cast<__nv_bfloat16 *>(out.data_ptr()), q.stride(0), k.stride(0), v.stride(0))
    if (bf16_state) {
      if (reads.size(0) == 1) {
        VECTOR_REC(__nv_bfloat16, 2, 2);
      } else {
        VECTOR_REC(__nv_bfloat16, 4, 2);
      }
    } else {
      if (reads.size(0) == 1) {
        VECTOR_REC(float, 2, 2);
      } else if (reads.size(0) == 2) {
        VECTOR_REC(float, 8, 1);
      } else if (reads.size(0) == 3) {
        VECTOR_REC(float, 2, 4);
      } else if (reads.size(0) == 4) {
        VECTOR_REC(float, 2, 4);
      } else {
        VECTOR_REC(float, 4, 2);
      }
    }
#undef VECTOR_REC
    finish_cuda_launch(stream, "recurrent");
    record_variant(kRecurrent, true);
    return;
  }
  dim3 grid(reads.size(0), v.size(1), 8);
#define REC(S)                                                                                     \
  launch_kernel(                                                                                   \
      recurrent_kernel<S>, grid, 128, 0, stream, static_cast<const __nv_bfloat16 *>(q.data_ptr()), \
      static_cast<const __nv_bfloat16 *>(k.data_ptr()),                                            \
      static_cast<const __nv_bfloat16 *>(v.data_ptr()),                                            \
      static_cast<const float *>(decay.data_ptr()), static_cast<const float *>(beta.data_ptr()),   \
      static_cast<S *>(pool.data_ptr()), starts.data_ptr(), reads.data_ptr(), writes.data_ptr(),   \
      starts.dtype().bits == 64, reads.dtype().bits == 64, writes.dtype().bits == 64,              \
      static_cast<__nv_bfloat16 *>(out.data_ptr()), q.size(1), v.size(1), q.stride(0),             \
      k.stride(0), v.stride(0))
  if (bf16_state) {
    REC(__nv_bfloat16);
  } else {
    REC(float);
  }
#undef REC
  finish_cuda_launch(stream, "recurrent");
  record_variant(kRecurrent, false);
}
