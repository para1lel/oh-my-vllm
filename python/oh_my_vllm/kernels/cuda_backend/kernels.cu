#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>

using tvm::ffi::TensorView;

// One warp owns one complete 128-value scaling group. Independent warps share
// a CTA; reductions do not require shared memory or block-wide synchronization.
template <typename Input, bool Silu, bool Column>
__global__ void quantize_kernel(const Input *x, __nv_fp8_e4m3 *out, float *scales, int rows,
                                int width) {
  const int lane = threadIdx.x & 31;
  const int group = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  const int groups = width / 128;
  if (group >= rows * groups)
    return;
  const int row = group / groups;
  const int col = (group % groups) * 128;
  float value[4];
  float maximum = 0;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    int offset = row * width * (Silu ? 2 : 1) + col + lane + j * 32;
    float v = static_cast<float>(x[offset]);
    if constexpr (Silu) {
      float activated = __bfloat162float(__float2bfloat16_rn(v / (1.f + expf(-v))));
      v = __bfloat162float(__float2bfloat16_rn(activated * static_cast<float>(x[offset + width])));
    }
    value[j] = v;
    maximum = fmaxf(maximum, fabsf(v));
  }
#pragma unroll
  for (int delta = 16; delta; delta /= 2)
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, delta));
  float scale = __fdiv_rn(fmaxf(maximum, 1e-10f), 448.f);
  if (lane == 0)
    scales[Column ? (group % groups) * rows + row : group] = scale;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    float v = fminf(448.f, fmaxf(-448.f, __fdiv_rn(value[j], scale)));
    out[row * width + col + lane + j * 32] = __nv_fp8_e4m3(v);
  }
}

template <typename Input>
void launch_quantize(TensorView x, TensorView out, TensorView scales, bool column, bool silu,
                     cudaStream_t stream) {
  int rows = out.size(0), width = out.size(1);
  int blocks = (rows * (width / 128) + 3) / 4;
#define LAUNCH(S, C)                                                                               \
  quantize_kernel<Input, S, C><<<blocks, 128, 0, stream>>>(                                        \
      static_cast<const Input *>(x.data_ptr()), static_cast<__nv_fp8_e4m3 *>(out.data_ptr()),      \
      static_cast<float *>(scales.data_ptr()), rows, width)
  if (silu) {
    if (column) {
      LAUNCH(true, true);
    } else {
      LAUNCH(true, false);
    }
  } else {
    if (column) {
      LAUNCH(false, true);
    } else {
      LAUNCH(false, false);
    }
  }
#undef LAUNCH
}

void quantize(TensorView x, TensorView out, TensorView scales, bool column, bool silu) {
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, x.device().device_id));
  if (x.dtype().code == kDLBfloat)
    launch_quantize<__nv_bfloat16>(x, out, scales, column, silu, stream);
  else if (x.dtype().bits == 16)
    launch_quantize<__half>(x, out, scales, column, silu, stream);
  else
    launch_quantize<float>(x, out, scales, column, silu, stream);
  TVM_FFI_ICHECK(cudaGetLastError() == cudaSuccess) << "CUDA quantize launch failed";
}

__global__ void silu_kernel(const __nv_bfloat16 *x, __nv_bfloat16 *out, int rows, int width) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * width)
    return;
  int offset = (i / width) * width * 2 + i % width;
  float g = __bfloat162float(x[offset]);
  float a = __bfloat162float(__float2bfloat16_rn(g / (1.f + expf(-g))));
  out[i] = __float2bfloat16_rn(a * __bfloat162float(x[offset + width]));
}
void silu_mul(TensorView x, TensorView out) {
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, x.device().device_id));
  int n = out.size(0) * out.size(1);
  if (!n)
    return;
  silu_kernel<<<(n + 255) / 256, 256, 0, stream>>>(static_cast<const __nv_bfloat16 *>(x.data_ptr()),
                                                   static_cast<__nv_bfloat16 *>(out.data_ptr()),
                                                   out.size(0), out.size(1));
  TVM_FFI_ICHECK(cudaGetLastError() == cudaSuccess) << "CUDA SiLU launch failed";
}

__device__ float warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset; offset >>= 1)
    value += __shfl_down_sync(0xffffffff, value, offset);
  return __shfl_sync(0xffffffff, value, 0);
}
__device__ int64_t index_at(const void *data, bool wide, int index) {
  return wide ? static_cast<const int64_t *>(data)[index]
              : static_cast<const int32_t *>(data)[index];
}
cudaStream_t stream_for(TensorView x) {
  return static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, x.device().device_id));
}

__global__ void gates_kernel(const __nv_bfloat16 *ba, const float *log, const float *bias,
                             float *decay, float *beta, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n * 48)
    return;
  int head = i % 48, offset = i / 48 * 96 + head;
  float b = __bfloat162float(ba[offset]);
  float a = __bfloat162float(ba[offset + 48]) + bias[head];
  float softplus = a > 20.f ? a : log1pf(expf(a));
  decay[i] = -expf(log[head]) * softplus;
  beta[i] = 1.f / (1.f + expf(-b));
}
void gates(TensorView ba, TensorView log, TensorView bias, TensorView decay, TensorView beta) {
  gates_kernel<<<(ba.size(0) * 48 + 255) / 256, 256, 0, stream_for(ba)>>>(
      static_cast<const __nv_bfloat16 *>(ba.data_ptr()), static_cast<const float *>(log.data_ptr()),
      static_cast<const float *>(bias.data_ptr()), static_cast<float *>(decay.data_ptr()),
      static_cast<float *>(beta.data_ptr()), ba.size(0));
  TVM_FFI_ICHECK(cudaGetLastError() == cudaSuccess) << "CUDA gate launch failed";
}

struct Strides {
  int64_t token, head, dim;
};
template <bool Large, bool Gated, bool Residual>
__global__ void rms_kernel(const __nv_bfloat16 *x, const float *weight, const __nv_bfloat16 *gate,
                           __nv_bfloat16 *out, __nv_bfloat16 *summed, int rows, int heads,
                           int width, Strides xs, Strides gs, float epsilon) {
  int lane = threadIdx.x & 31;
  int row = Large ? blockIdx.x : blockIdx.x * 4 + threadIdx.x / 32;
  if (row >= rows)
    return;
  int step = Large ? 256 : 32;
  int first = Large ? threadIdx.x : lane;
  int64_t offset = (row / heads) * xs.token + (row % heads) * xs.head;
  int64_t goffset = (row / heads) * gs.token + (row % heads) * gs.head;
  float total = 0;
  for (int i = first; i < width; i += step) {
    float v = __bfloat162float(x[offset + i * xs.dim]);
    if constexpr (Residual) {
      v = __bfloat162float(__float2bfloat16_rn(v + __bfloat162float(gate[goffset + i * gs.dim])));
      summed[static_cast<int64_t>(row) * width + i] = __float2bfloat16_rn(v);
    }
    total += v * v;
  }
  total = warp_sum(total);
  if constexpr (Large) {
    __shared__ float partial[8];
    if (lane == 0)
      partial[threadIdx.x / 32] = total;
    __syncthreads();
    total = warp_sum(lane < 8 ? partial[lane] : 0.f);
  }
  float inv = rsqrtf(total / width + epsilon);
  for (int i = first; i < width; i += step) {
    float v = __bfloat162float(x[offset + i * xs.dim]);
    if constexpr (Residual)
      v = __bfloat162float(summed[static_cast<int64_t>(row) * width + i]);
    v = v * inv * weight[i];
    if constexpr (Gated) {
      float g = __bfloat162float(gate[goffset + i * gs.dim]);
      v = v * g / (1.f + expf(-g));
    }
    out[static_cast<int64_t>(row) * width + i] = __float2bfloat16_rn(v);
  }
}
void rms(TensorView x, TensorView weight, TensorView gate, TensorView out, double epsilon,
         bool gated) {
  int h = x.size(1), d = x.size(2), rows = x.size(0) * h;
  Strides xs{x.stride(0), x.stride(1), x.stride(2)};
  Strides gs{gate.stride(0), gate.stride(1), gate.stride(2)};
#define RMS(L, G)                                                                                  \
  rms_kernel<L, G, false><<<L ? rows : (rows + 3) / 4, L ? 256 : 128, 0, stream_for(x)>>>(         \
      static_cast<const __nv_bfloat16 *>(x.data_ptr()),                                            \
      static_cast<const float *>(weight.data_ptr()),                                               \
      static_cast<const __nv_bfloat16 *>(gate.data_ptr()),                                         \
      static_cast<__nv_bfloat16 *>(out.data_ptr()), nullptr, rows, h, d, xs, gs, epsilon)
  if (d > 256) {
    if (gated) {
      RMS(true, true);
    } else {
      RMS(true, false);
    }
  } else {
    if (gated) {
      RMS(false, true);
    } else {
      RMS(false, false);
    }
  }
#undef RMS
  TVM_FFI_ICHECK(cudaGetLastError() == cudaSuccess) << "CUDA RMS launch failed";
}
void add_rms(TensorView x, TensorView residual, TensorView weight, TensorView summed,
             TensorView out) {
  int d = x.size(1), rows = x.size(0);
  rms_kernel<true, false, true><<<rows, 256, 0, stream_for(x)>>>(
      static_cast<const __nv_bfloat16 *>(x.data_ptr()),
      static_cast<const float *>(weight.data_ptr()),
      static_cast<const __nv_bfloat16 *>(residual.data_ptr()),
      static_cast<__nv_bfloat16 *>(out.data_ptr()), static_cast<__nv_bfloat16 *>(summed.data_ptr()),
      rows, 1, d, {d, d, 1}, {d, d, 1}, 1e-6f);
  TVM_FFI_ICHECK(cudaGetLastError() == cudaSuccess) << "CUDA residual RMS launch failed";
}

__device__ float phase(int64_t position, int channel, int rotary, double theta) {
  double a = static_cast<double>(position) * exp(-log(theta) * channel * 2. / rotary);
  constexpr double tau = 6.283185307179586476925286766559;
  return static_cast<float>(a - nearbyint(a / tau) * tau);
}
__global__ void rms_rope_kernel(const __nv_bfloat16 *x, const float *weight, const void *positions,
                                __nv_bfloat16 *out, int rows, int heads, Strides xs, bool wide) {
  int row = blockIdx.x * 4 + threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  if (row >= rows)
    return;
  int64_t offset = row / heads * xs.token + row % heads * xs.head;
  float values[8], total = 0;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    values[j] = __bfloat162float(x[offset + (lane + j * 32) * xs.dim]);
    total += values[j] * values[j];
  }
  float inv = rsqrtf(warp_sum(total) / 256.f + 1e-6f);
#pragma unroll
  for (int j = 0; j < 8; ++j)
    values[j] = __bfloat162float(__float2bfloat16_rn(values[j] * inv * weight[lane + j * 32]));
  float a = phase(index_at(positions, wide, row / heads), lane, 64, 10000000.);
  float c = cosf(a), s = sinf(a);
  float left = values[0], right = values[1];
  values[0] = left * c - right * s;
  values[1] = right * c + left * s;
#pragma unroll
  for (int j = 0; j < 8; ++j)
    out[static_cast<int64_t>(row) * 256 + lane + j * 32] = __float2bfloat16_rn(values[j]);
}
void rms_rope(TensorView x, TensorView weight, TensorView positions, TensorView out) {
  int rows = x.size(0) * x.size(1);
  rms_rope_kernel<<<(rows + 3) / 4, 128, 0, stream_for(x)>>>(
      static_cast<const __nv_bfloat16 *>(x.data_ptr()),
      static_cast<const float *>(weight.data_ptr()), positions.data_ptr(),
      static_cast<__nv_bfloat16 *>(out.data_ptr()), rows, x.size(1),
      {x.stride(0), x.stride(1), x.stride(2)}, positions.dtype().bits == 64);
  TVM_FFI_ICHECK(cudaGetLastError() == cudaSuccess) << "CUDA RMS/RoPE launch failed";
}
__global__ void rope_kernel(const __nv_bfloat16 *x, const void *positions, __nv_bfloat16 *out,
                            int n, int h, int d, int rotary, double theta, bool wide) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n * h * d)
    return;
  int col = i % d;
  float v = __bfloat162float(x[i]);
  if (col < rotary) {
    int half = rotary / 2;
    float other = __bfloat162float(x[i + (col < half ? half : -half)]);
    float a = phase(index_at(positions, wide, i / (h * d)), col % half, rotary, theta);
    v = v * cosf(a) + (col < half ? -other : other) * sinf(a);
  }
  out[i] = __float2bfloat16_rn(v);
}
void rope(TensorView x, TensorView positions, TensorView out, int64_t rotary, double theta) {
  int n = x.size(0), h = x.size(1), d = x.size(2);
  rope_kernel<<<(n * h * d + 255) / 256, 256, 0, stream_for(x)>>>(
      static_cast<const __nv_bfloat16 *>(x.data_ptr()), positions.data_ptr(),
      static_cast<__nv_bfloat16 *>(out.data_ptr()), n, h, d, rotary, theta,
      positions.dtype().bits == 64);
  TVM_FFI_ICHECK(cudaGetLastError() == cudaSuccess) << "CUDA RoPE launch failed";
}

__global__ void qk_kernel(const __nv_bfloat16 *q, const __nv_bfloat16 *k, __nv_bfloat16 *oq,
                          __nv_bfloat16 *ok, int rows, int heads, int64_t qs, int64_t ks) {
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
  int rows = q.size(0) * q.size(1);
  qk_kernel<<<(rows + 3) / 4, 128, 0, stream_for(q)>>>(
      static_cast<const __nv_bfloat16 *>(q.data_ptr()),
      static_cast<const __nv_bfloat16 *>(k.data_ptr()), static_cast<__nv_bfloat16 *>(oq.data_ptr()),
      static_cast<__nv_bfloat16 *>(ok.data_ptr()), rows, q.size(1), q.stride(0), k.stride(0));
  TVM_FFI_ICHECK(cudaGetLastError() == cudaSuccess) << "CUDA Q/K normalization launch failed";
}

template <typename State>
__global__ void recurrent_kernel(const __nv_bfloat16 *q, const __nv_bfloat16 *k,
                                 const __nv_bfloat16 *v, const float *decay, const float *beta,
                                 State *pool, const void *starts, const void *reads,
                                 const void *writes, bool sw, bool rw, bool ww, __nv_bfloat16 *out,
                                 int hq, int hv, int64_t qs, int64_t ks, int64_t vs) {
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
  dim3 grid(reads.size(0), v.size(1), 8);
#define REC(S)                                                                                     \
  recurrent_kernel<S><<<grid, 128, 0, stream_for(q)>>>(                                            \
      static_cast<const __nv_bfloat16 *>(q.data_ptr()),                                            \
      static_cast<const __nv_bfloat16 *>(k.data_ptr()),                                            \
      static_cast<const __nv_bfloat16 *>(v.data_ptr()),                                            \
      static_cast<const float *>(decay.data_ptr()), static_cast<const float *>(beta.data_ptr()),   \
      static_cast<S *>(pool.data_ptr()), starts.data_ptr(), reads.data_ptr(), writes.data_ptr(),   \
      starts.dtype().bits == 64, reads.dtype().bits == 64, writes.dtype().bits == 64,              \
      static_cast<__nv_bfloat16 *>(out.data_ptr()), q.size(1), v.size(1), q.stride(0),             \
      k.stride(0), v.stride(0))
  if (pool.dtype().code == kDLBfloat) {
    REC(__nv_bfloat16);
  } else {
    REC(float);
  }
#undef REC
  TVM_FFI_ICHECK(cudaGetLastError() == cudaSuccess) << "CUDA recurrence launch failed";
}

template <bool Vector>
__global__ void append_kernel(const __nv_bfloat16 *k, const __nv_bfloat16 *v, __nv_bfloat16 *cache,
                              const void *slots, int width, bool wide) {
  int token = blockIdx.x;
  int64_t slot = index_at(slots, wide, token);
  if (slot < 0)
    return;
  int64_t dst = ((slot / 784) * 1568 + slot % 784) * width;
  if constexpr (Vector) {
    for (int j = threadIdx.x; j < width / 8; j += blockDim.x) {
      reinterpret_cast<uint4 *>(cache + dst)[j] =
          reinterpret_cast<const uint4 *>(k + static_cast<int64_t>(token) * width)[j];
      reinterpret_cast<uint4 *>(cache + dst + 784 * width)[j] =
          reinterpret_cast<const uint4 *>(v + static_cast<int64_t>(token) * width)[j];
    }
  } else {
    for (int j = threadIdx.x; j < width; j += blockDim.x) {
      cache[dst + j] = k[static_cast<int64_t>(token) * width + j];
      cache[dst + 784 * width + j] = v[static_cast<int64_t>(token) * width + j];
    }
  }
}
void append(TensorView k, TensorView v, TensorView cache, TensorView slots) {
  int width = k.size(1) * k.size(2);
  bool vector = width % 8 == 0 && reinterpret_cast<uintptr_t>(k.data_ptr()) % 16 == 0 &&
                reinterpret_cast<uintptr_t>(v.data_ptr()) % 16 == 0 &&
                reinterpret_cast<uintptr_t>(cache.data_ptr()) % 16 == 0;
#define APPEND(V)                                                                                  \
  append_kernel<V>                                                                                 \
      <<<k.size(0), 128, 0, stream_for(k)>>>(static_cast<const __nv_bfloat16 *>(k.data_ptr()),     \
                                             static_cast<const __nv_bfloat16 *>(v.data_ptr()),     \
                                             static_cast<__nv_bfloat16 *>(cache.data_ptr()),       \
                                             slots.data_ptr(), width, slots.dtype().bits == 64)
  if (vector) {
    APPEND(true);
  } else {
    APPEND(false);
  }
#undef APPEND
  TVM_FFI_ICHECK(cudaGetLastError() == cudaSuccess) << "CUDA KV append launch failed";
}

template <int Rows>
__global__ void convolution_kernel(const __nv_bfloat16 *x, const __nv_bfloat16 *weight,
                                   __nv_bfloat16 *pool, const void *ids, const void *starts,
                                   const __nv_bfloat16 *sources, const void *writes,
                                   __nv_bfloat16 *out, int n, int c, int64_t stride, bool iw,
                                   bool sw, bool ww) {
  int channel = blockIdx.y * 128 + threadIdx.x;
  if (channel >= c)
    return;
  float w[4];
#pragma unroll
  for (int j = 0; j < 4; ++j)
    w[j] = __bfloat162float(weight[channel * 4 + j]);
#pragma unroll
  for (int i = 0; i < Rows; ++i) {
    int token = blockIdx.x * Rows + i;
    if (token >= n)
      continue;
    int64_t seq = index_at(ids, iw, token), first = index_at(starts, sw, seq);
    int64_t target = index_at(writes, ww, token);
    float total = 0;
#pragma unroll
    for (int tap = 0; tap < 4; ++tap) {
      int64_t pos = token + tap - 3;
      __nv_bfloat16 value = pos >= first ? x[pos * stride + channel]
                                         : sources[(seq * c + channel) * 3 + pos - first + 3];
      total += __bfloat162float(value) * w[tap];
      if (tap > 0 && target >= 0)
        pool[(target * c + channel) * 3 + tap - 1] = value;
    }
    out[static_cast<int64_t>(token) * c + channel] =
        __float2bfloat16_rn(total / (1.f + expf(-total)));
  }
}
void convolution(TensorView x, TensorView weight, TensorView pool, TensorView ids,
                 TensorView starts, TensorView sources, TensorView writes, TensorView out) {
  int n = x.size(0), c = x.size(1);
#define CONV(R)                                                                                    \
  convolution_kernel<R><<<dim3((n + R - 1) / R, (c + 127) / 128), 128, 0, stream_for(x)>>>(        \
      static_cast<const __nv_bfloat16 *>(x.data_ptr()),                                            \
      static_cast<const __nv_bfloat16 *>(weight.data_ptr()),                                       \
      static_cast<__nv_bfloat16 *>(pool.data_ptr()), ids.data_ptr(), starts.data_ptr(),            \
      static_cast<const __nv_bfloat16 *>(sources.data_ptr()), writes.data_ptr(),                   \
      static_cast<__nv_bfloat16 *>(out.data_ptr()), n, c, x.stride(0), ids.dtype().bits == 64,     \
      starts.dtype().bits == 64, writes.dtype().bits == 64)
  if (n >= 128) {
    CONV(8);
  } else {
    CONV(1);
  }
#undef CONV
  TVM_FFI_ICHECK(cudaGetLastError() == cudaSuccess) << "CUDA convolution launch failed";
}

#include <mma.h>
namespace wmma = nvcuda::wmma;

__device__ float warp_max(float value) {
#pragma unroll
  for (int offset = 16; offset; offset >>= 1)
    value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
  return __shfl_sync(0xffffffff, value, 0);
}

// 16 logical query/head rows share a paged KV tile. The first implementation
// uses ordinary WMMA and explicit asynchronous copies, with independent splits.
__global__ void attention_partial_kernel(const __nv_bfloat16 *query, const __nv_bfloat16 *cache,
                                         const void *tables, const void *lengths,
                                         const void *starts, float *partial, float *lse, int h,
                                         int hk, int table_width, int splits, int first,
                                         bool grouped, bool tw, bool lw, bool sw, int query_tiles) {
  extern __shared__ __align__(32) unsigned char storage[];
  auto *q = reinterpret_cast<__nv_bfloat16 *>(storage);
  auto *k = q + 16 * 256;
  auto *v = k + 32 * 256;
  auto *p = v + 32 * 256;
  auto *score = reinterpret_cast<float *>(p + 16 * 32);
  auto *acc = score + 16 * 32;
  auto *maximum = acc + 16 * 256;
  auto *denom = maximum + 16;
  auto *alpha = denom + 16;
  const int warp = threadIdx.x / 32, lane = threadIdx.x & 31;
  const int seq = blockIdx.x / query_tiles, qbase = (blockIdx.x % query_tiles) * 16;
  const int kh = blockIdx.y, split = blockIdx.z, ratio = h / hk;
  const int start = grouped ? index_at(starts, sw, seq) : seq;
  const int end = grouped ? index_at(starts, sw, seq + 1) : seq + 1;
  const int64_t last = index_at(lengths, lw, end - 1);
  const int64_t chunk = ((max(last - first, int64_t(0)) + splits * 32 - 1) / (splits * 32)) * 32;
  const int64_t begin = first + split * chunk;
  for (int i = threadIdx.x; i < 16 * 256; i += 128) {
    int row = start + (qbase + i / 256) / ratio;
    q[i] =
        row < end
            ? query[(static_cast<int64_t>(row) * h + kh * ratio + (qbase + i / 256) % ratio) * 256 +
                    i % 256]
            : __float2bfloat16(0);
    acc[i] = 0;
  }
  if (threadIdx.x < 16) {
    maximum[threadIdx.x] = -INFINITY;
    denom[threadIdx.x] = 0;
  }
  __syncthreads();
  for (int64_t base = begin; base < min(begin + chunk, last); base += 32) {
    for (int i = threadIdx.x; i < 32 * 32; i += 128) {
      int64_t pos = base + i / 32;
      int64_t page = index_at(tables, tw,
                              static_cast<int64_t>(start) * table_width + min(pos, last - 1) / 784);
      int64_t offset = ((page * 1568 + pos % 784) * hk + kh) * 256 + (i % 32) * 8;
      unsigned kd = __cvta_generic_to_shared(k + i * 8);
      unsigned vd = __cvta_generic_to_shared(v + i * 8);
      int bytes = pos < last ? 16 : 0;
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(kd), "l"(cache + offset),
                   "r"(bytes)
                   : "memory");
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(vd),
                   "l"(cache + offset + 784 * hk * 256), "r"(bytes)
                   : "memory");
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    if (warp < 2) {
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
      wmma::fill_fragment(c, 0.f);
      for (int inner = 0; inner < 256; inner += 16) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b;
        wmma::load_matrix_sync(a, q + inner, 256);
        wmma::load_matrix_sync(b, k + warp * 16 * 256 + inner, 256);
        wmma::mma_sync(c, a, b, c);
      }
      wmma::store_matrix_sync(score + warp * 16, c, 32, wmma::mem_row_major);
    }
    __syncthreads();
    for (int r = warp; r < 16; r += 4) {
      int row = start + (qbase + r) / ratio;
      float value = row < end && base + lane < index_at(lengths, lw, row)
                        ? score[r * 32 + lane] * 0.09016844005556021f
                        : -INFINITY;
      float m = fmaxf(maximum[r], warp_max(value));
      float safe = m == -INFINITY ? 0.f : m;
      float a = exp2f(maximum[r] - safe), probability = exp2f(value - safe);
      float sum = warp_sum(probability);
      p[r * 32 + lane] = __float2bfloat16_rn(probability);
      if (lane == 0) {
        alpha[r] = a;
        maximum[r] = m;
        denom[r] = denom[r] * a + sum;
      }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < 16 * 256; i += 128)
      acc[i] *= alpha[i / 256];
    __syncthreads();
    for (int tile = warp; tile < 16; tile += 4) {
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
      wmma::load_matrix_sync(c, acc + tile * 16, 256, wmma::mem_row_major);
      for (int inner = 0; inner < 32; inner += 16) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b;
        wmma::load_matrix_sync(a, p + inner, 32);
        wmma::load_matrix_sync(b, v + inner * 256 + tile * 16, 256);
        wmma::mma_sync(c, a, b, c);
      }
      wmma::store_matrix_sync(acc + tile * 16, c, 256, wmma::mem_row_major);
    }
    __syncthreads();
  }
  for (int i = threadIdx.x; i < 16 * 256; i += 128) {
    int qr = qbase + i / 256, row = start + qr / ratio;
    if (row < end) {
      int64_t dst =
          ((static_cast<int64_t>(row) * h + kh * ratio + qr % ratio) * splits + split) * 256 +
          i % 256;
      partial[dst] = acc[i] / (denom[i / 256] > 0 ? denom[i / 256] : 1.f);
    }
  }
  if (threadIdx.x < 16) {
    int qr = qbase + threadIdx.x, row = start + qr / ratio;
    if (row < end)
      lse[(static_cast<int64_t>(row) * h + kh * ratio + qr % ratio) * splits + split] =
          maximum[threadIdx.x] + log2f(denom[threadIdx.x] > 0 ? denom[threadIdx.x] : 1.f);
  }
}
void attention_partial(TensorView q, TensorView cache, TensorView tables, TensorView lengths,
                       TensorView starts, TensorView partial, TensorView lse, int64_t first,
                       bool grouped) {
  int h = q.size(1), hk = cache.size(3), splits = lse.size(2);
  int tiles = ((grouped ? 5 : 1) * (h / hk) + 15) / 16;
  constexpr int shared_bytes =
      (16 * 256 + 2 * 32 * 256 + 16 * 32) * 2 + (16 * 32 + 16 * 256 + 3 * 16) * 4;
  TVM_FFI_ICHECK(cudaFuncSetAttribute(attention_partial_kernel,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      shared_bytes) == cudaSuccess);
  dim3 grid((grouped ? starts.size(0) - 1 : q.size(0)) * tiles, hk, splits);
  attention_partial_kernel<<<grid, 128, shared_bytes, stream_for(q)>>>(
      static_cast<const __nv_bfloat16 *>(q.data_ptr()),
      static_cast<const __nv_bfloat16 *>(cache.data_ptr()), tables.data_ptr(), lengths.data_ptr(),
      starts.data_ptr(), static_cast<float *>(partial.data_ptr()),
      static_cast<float *>(lse.data_ptr()), h, hk, tables.size(1), splits, first, grouped,
      tables.dtype().bits == 64, lengths.dtype().bits == 64, starts.dtype().bits == 64, tiles);
  TVM_FFI_ICHECK(cudaGetLastError() == cudaSuccess) << "CUDA attention partial launch failed";
}
__global__ void attention_merge_kernel(const float *partial, const float *lse, __nv_bfloat16 *out,
                                       int splits) {
  int row = blockIdx.x, lane = threadIdx.x & 31;
  extern __shared__ float weights[];
  // Compute each split weight once, shared by all256 output columns.
  if (threadIdx.x < 32) {
    float maximum = -INFINITY;
    for (int i = lane; i < splits; i += 32)
      maximum = fmaxf(maximum, lse[row * splits + i]);
    maximum = warp_max(maximum);
    float total = 0;
    for (int i = lane; i < splits; i += 32) {
      float weight = exp2f(lse[row * splits + i] - maximum);
      weights[i] = weight;
      total += weight;
    }
    total = warp_sum(total);
    if (lane == 0)
      weights[splits] = total;
  }
  __syncthreads();
  for (int col = threadIdx.x; col < 256; col += 128) {
    float value = 0;
    for (int i = 0; i < splits; ++i)
      value += partial[(static_cast<int64_t>(row) * splits + i) * 256 + col] * weights[i];
    out[static_cast<int64_t>(row) * 256 + col] = __float2bfloat16_rn(value / weights[splits]);
  }
}
void attention_merge(TensorView partial, TensorView lse, TensorView out) {
  attention_merge_kernel<<<lse.size(0) * lse.size(1), 128, (lse.size(2) + 1) * sizeof(float),
                           stream_for(out)>>>(
      static_cast<const float *>(partial.data_ptr()), static_cast<const float *>(lse.data_ptr()),
      static_cast<__nv_bfloat16 *>(out.data_ptr()), lse.size(2));
  TVM_FFI_ICHECK(cudaGetLastError() == cudaSuccess) << "CUDA attention merge launch failed";
}
