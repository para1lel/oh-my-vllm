#pragma once

#include "common.cuh"

template <int Rows>
__global__ void
model_convolution_kernel(const __nv_bfloat16 *__restrict__ x,
                         const __nv_bfloat16 *__restrict__ weight, __nv_bfloat16 *__restrict__ pool,
                         const void *__restrict__ ids, const void *__restrict__ starts,
                         const __nv_bfloat16 *__restrict__ sources, const void *__restrict__ writes,
                         __nv_bfloat16 *__restrict__ out, int n, bool iw, bool sw, bool ww) {
  pdl_dependency_wait();
  pdl_launch_next();
  constexpr int c = 10240, stride = 16384;
  int channel = blockIdx.y * 128 + threadIdx.x;
  if (channel >= c)
    return;
  auto packed = *reinterpret_cast<const Four<__nv_bfloat16> *>(weight + channel * 4);
  float w[4];
#pragma unroll
  for (int j = 0; j < 4; ++j)
    w[j] = __bfloat162float(packed.values[j]);
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
    float e = __expf(-fabsf(total));
    float activated = total * __fdividef(total >= 0.f ? 1.f : e, 1.f + e);
    out[static_cast<int64_t>(token) * c + channel] = __float2bfloat16_rn(activated);
  }
}
// Four adjacent channels share token metadata and aligned activation loads.
// Keep the scalar tap order, activation formula, and snapshot writes.
template <int Rows>
__global__ void vector_model_convolution_kernel(
    const __nv_bfloat16 *__restrict__ x, const __nv_bfloat16 *__restrict__ weight,
    __nv_bfloat16 *__restrict__ pool, const void *__restrict__ ids, const void *__restrict__ starts,
    const __nv_bfloat16 *__restrict__ sources, const void *__restrict__ writes,
    __nv_bfloat16 *__restrict__ out, int n, bool iw, bool sw, bool ww) {
  pdl_dependency_wait();
  pdl_launch_next();
  constexpr int c = 10240, stride = 16384, channels = 4;
  int channel = (blockIdx.y * 128 + threadIdx.x) * channels;
  float w[channels][4];
#pragma unroll
  for (int j = 0; j < channels; ++j) {
    auto packed = *reinterpret_cast<const Four<__nv_bfloat16> *>(weight + (channel + j) * 4);
#pragma unroll
    for (int tap = 0; tap < 4; ++tap)
      w[j][tap] = __bfloat162float(packed.values[tap]);
  }
#pragma unroll
  for (int i = 0; i < Rows; ++i) {
    int token = blockIdx.x * Rows + i;
    if (token >= n)
      continue;
    int64_t seq = index_at(ids, iw, token), first = index_at(starts, sw, seq);
    int64_t target = index_at(writes, ww, token);
    float total[channels] = {};
#pragma unroll
    for (int tap = 0; tap < 4; ++tap) {
      int64_t pos = token + tap - 3;
      Four<__nv_bfloat16> packed;
      if (pos >= first) {
        packed = *reinterpret_cast<const Four<__nv_bfloat16> *>(x + pos * stride + channel);
      } else {
#pragma unroll
        for (int j = 0; j < channels; ++j)
          packed.values[j] = sources[(seq * c + channel + j) * 3 + pos - first + 3];
      }
#pragma unroll
      for (int j = 0; j < channels; ++j) {
        auto value = packed.values[j];
        total[j] += __bfloat162float(value) * w[j][tap];
        if (tap > 0 && target >= 0)
          pool[(target * c + channel + j) * 3 + tap - 1] = value;
      }
    }
    Four<__nv_bfloat16> result;
#pragma unroll
    for (int j = 0; j < channels; ++j) {
      float e = __expf(-fabsf(total[j]));
      float value = total[j] * __fdividef(total[j] >= 0.f ? 1.f : e, 1.f + e);
      result.values[j] = __float2bfloat16_rn(value);
    }
    *reinterpret_cast<Four<__nv_bfloat16> *>(out + static_cast<int64_t>(token) * c + channel) =
        result;
  }
}
template <int Rows>
__global__ void convolution_kernel(const __nv_bfloat16 *x, const __nv_bfloat16 *weight,
                                   __nv_bfloat16 *pool, const void *ids, const void *starts,
                                   const __nv_bfloat16 *sources, const void *writes,
                                   __nv_bfloat16 *out, int n, int c, int64_t stride, bool iw,
                                   bool sw, bool ww) {
  pdl_dependency_wait();
  pdl_launch_next();
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
  bool model =
      c == 10240 && x.stride(0) == 16384 && reinterpret_cast<uintptr_t>(weight.data_ptr()) % 8 == 0;
  for (auto input : {x, weight, ids, starts, sources, writes})
    model = model && disjoint_storage(pool, input);
  auto stream = stream_for(x, "convolution");
  if (model && n >= 128 && reinterpret_cast<uintptr_t>(x.data_ptr()) % 8 == 0 &&
      reinterpret_cast<uintptr_t>(out.data_ptr()) % 8 == 0) {
#define VECTOR_CONV(R)                                                                             \
  launch_kernel(vector_model_convolution_kernel<R>, dim3((n + R - 1) / R, 20), 128, 0, stream,     \
                static_cast<const __nv_bfloat16 *>(x.data_ptr()),                                  \
                static_cast<const __nv_bfloat16 *>(weight.data_ptr()),                             \
                static_cast<__nv_bfloat16 *>(pool.data_ptr()), ids.data_ptr(), starts.data_ptr(),  \
                static_cast<const __nv_bfloat16 *>(sources.data_ptr()), writes.data_ptr(),         \
                static_cast<__nv_bfloat16 *>(out.data_ptr()), n, ids.dtype().bits == 64,           \
                starts.dtype().bits == 64, writes.dtype().bits == 64)
    if (n >= 4096) {
      VECTOR_CONV(8);
    } else {
      VECTOR_CONV(4);
    }
#undef VECTOR_CONV
    finish_cuda_launch(stream, "convolution");
    record_variant(kConvolution, true);
    return;
  }
  if (model) {
#define MODEL_CONV(R)                                                                              \
  launch_kernel(model_convolution_kernel<R>, dim3((n + R - 1) / R, 80), 128, 0, stream,            \
                static_cast<const __nv_bfloat16 *>(x.data_ptr()),                                  \
                static_cast<const __nv_bfloat16 *>(weight.data_ptr()),                             \
                static_cast<__nv_bfloat16 *>(pool.data_ptr()), ids.data_ptr(), starts.data_ptr(),  \
                static_cast<const __nv_bfloat16 *>(sources.data_ptr()), writes.data_ptr(),         \
                static_cast<__nv_bfloat16 *>(out.data_ptr()), n, ids.dtype().bits == 64,           \
                starts.dtype().bits == 64, writes.dtype().bits == 64)
    if (n >= 4096) {
      MODEL_CONV(8);
    } else if (n >= 128) {
      MODEL_CONV(4);
    } else {
      MODEL_CONV(1);
    }
#undef MODEL_CONV
    finish_cuda_launch(stream, "convolution");
    record_variant(kConvolution, true);
    return;
  }
#define CONV(R)                                                                                    \
  launch_kernel(convolution_kernel<R>, dim3((n + R - 1) / R, (c + 127) / 128), 128, 0, stream,     \
                static_cast<const __nv_bfloat16 *>(x.data_ptr()),                                  \
                static_cast<const __nv_bfloat16 *>(weight.data_ptr()),                             \
                static_cast<__nv_bfloat16 *>(pool.data_ptr()), ids.data_ptr(), starts.data_ptr(),  \
                static_cast<const __nv_bfloat16 *>(sources.data_ptr()), writes.data_ptr(),         \
                static_cast<__nv_bfloat16 *>(out.data_ptr()), n, c, x.stride(0),                   \
                ids.dtype().bits == 64, starts.dtype().bits == 64, writes.dtype().bits == 64)
  if (n >= 128) {
    CONV(8);
  } else {
    CONV(1);
  }
#undef CONV
  finish_cuda_launch(stream, "convolution");
  record_variant(kConvolution, false);
}
