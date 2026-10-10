#pragma once

#include "common.cuh"

template <bool Residual, int Threads, int Vector, bool Streaming>
__global__ void norm_quant5120_kernel(const __nv_bfloat16 *__restrict__ x,
                                      const __nv_bfloat16 *__restrict__ residual,
                                      const float *__restrict__ w, __nv_fp8_e4m3 *__restrict__ out,
                                      float *__restrict__ scales, bool column, int scale_rows,
                                      __nv_bfloat16 *__restrict__ summed, float eps) {
  pdl_dependency_wait();
  pdl_launch_next();
  static_assert(Threads % 32 == 0 && 5120 % (Threads * Vector) == 0);
  constexpr int Chunks = 5120 / (Threads * Vector);
  int row = blockIdx.x, lane = threadIdx.x & 31;
  float values[Chunks][Vector], total = 0;
#pragma unroll
  for (int chunk = 0; chunk < Chunks; ++chunk) {
    int col = chunk * Threads * Vector + threadIdx.x * Vector;
    auto q = load_aligned_vector<__nv_bfloat16, Vector>(x + static_cast<int64_t>(row) * 5120 + col);
    AlignedVector<__nv_bfloat16, Vector> r;
    if constexpr (Residual)
      r = load_aligned_vector<__nv_bfloat16, Vector>(residual + static_cast<int64_t>(row) * 5120 +
                                                     col);
    if constexpr (Residual) {
#pragma unroll
      for (int j = 0; j < Vector; j += 2) {
        auto pair = __hadd2(__halves2bfloat162(q.value[j], q.value[j + 1]),
                            __halves2bfloat162(r.value[j], r.value[j + 1]));
        q.value[j] = __low2bfloat16(pair);
        q.value[j + 1] = __high2bfloat16(pair);
      }
    }
#pragma unroll
    for (int j = 0; j < Vector; ++j) {
      float v = __bfloat162float(q.value[j]);
      values[chunk][j] = v;
      total += v * v;
    }
    if constexpr (Residual)
      store_rms_vector<Vector, Streaming>(summed + static_cast<int64_t>(row) * 5120 + col, q);
  }
  total = warp_sum(total);
  __shared__ float partial[Threads / 32];
  if (lane == 0)
    partial[threadIdx.x / 32] = total;
  __syncthreads();
  total = warp_sum(lane < Threads / 32 ? partial[lane] : 0.f);
  float inv = rsqrtf(total * (1.f / 5120.f) + eps);
#pragma unroll
  for (int chunk = 0; chunk < Chunks; ++chunk) {
    int col = chunk * Threads * Vector + threadIdx.x * Vector;
    auto weight = load_aligned_vector<float, Vector>(w + col);
    float value[Vector], maximum = 0;
#pragma unroll
    for (int j = 0; j < Vector; ++j) {
      value[j] = __bfloat162float(__float2bfloat16_rn(values[chunk][j] * inv * weight.value[j]));
      maximum = fmaxf(maximum, fabsf(value[j]));
    }
    unsigned mask = Vector == 8 ? (lane < 16 ? 0xffffu : 0xffff0000u) : 0xffffffffu;
    maximum = __uint_as_float(__reduce_max_sync(mask, __float_as_uint(maximum)));
    float scale = __fdiv_rn(fmaxf(maximum, 1e-10f), 448.f);
    int group = col / 128;
    if (lane % (128 / Vector) == 0)
      scales[column ? group * scale_rows + row : row * 40 + group] = scale;
    float inverse = 0.f;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(inverse) : "f"(scale));
#pragma unroll
    for (int j = 0; j < Vector; ++j) {
      float divided;
      if (isfinite(maximum)) {
        float magnitude = fabsf(value[j]);
        float initial = magnitude * inverse;
        float corrected = __fmaf_rn(__fmaf_rn(-initial, scale, magnitude), inverse, initial);
        divided = copysignf(corrected, value[j]);
      } else {
        divided = __fdiv_rn(value[j], scale);
      }
      value[j] = fminf(448.f, fmaxf(-448.f, divided));
    }
#pragma unroll
    for (int j = 0; j < Vector; j += 4)
      reinterpret_cast<uint32_t *>(out)[(static_cast<int64_t>(row) * 5120 + col + j) / 4] =
          __nv_fp8x4_e4m3(make_float4(value[j], value[j + 1], value[j + 2], value[j + 3])).__x;
  }
}

void rms_quantize(TensorView x, TensorView residual, TensorView weight, TensorView data,
                  TensorView scales, TensorView summed, bool column) {
  for (auto tensor : {residual, weight, data, scales, summed})
    TVM_FFI_ICHECK(same_cuda_device(tensor, x)) << "RMS quantization device mismatch";
  TVM_FFI_ICHECK(x.ndim() == 2 && x.size(1) == 5120 && x.size(0) > 0 && residual.ndim() == 2 &&
                 residual.size(0) == x.size(0) && residual.size(1) == 5120 && summed.ndim() == 2 &&
                 summed.size(0) == x.size(0) && summed.size(1) == 5120 && data.ndim() == 2 &&
                 data.size(0) == x.size(0) && data.size(1) == 5120 && scales.ndim() == 2 &&
                 scales.size(0) == x.size(0) && scales.size(1) == 40 && weight.ndim() == 1 &&
                 weight.size(0) == 5120)
      << "RMS quantization requires model-width matrices and scales";
  TVM_FFI_ICHECK(has_dtype(x, kDLBfloat, 16) && has_dtype(residual, kDLBfloat, 16) &&
                 has_dtype(summed, kDLBfloat, 16) && has_dtype(data, kDLFloat8_e4m3fn, 8) &&
                 has_dtype(weight, kDLFloat, 32) && has_dtype(scales, kDLFloat, 32))
      << "RMS quantization requires BF16 inputs, FP8 data, and FP32 weights/scales";
  for (auto tensor : {x, residual, summed, data})
    TVM_FFI_ICHECK(tensor.stride(1) == 1 && tensor.stride(0) == 5120)
        << "RMS quantization requires contiguous matrices";
  TVM_FFI_ICHECK(weight.stride(0) == 1 && scales.stride(0) == (column ? 1 : 40) &&
                 (column ? scales.stride(1) >= x.size(0) && scales.stride(1) - x.size(0) <= 3
                         : scales.stride(1) == 1))
      << "RMS quantization scale layout mismatch";
  TVM_FFI_ICHECK(fits_int32_flat_offsets(x, false) && aligned(x, 16) && aligned(residual, 16) &&
                 aligned(weight, 32) && aligned(summed, 16) && aligned(data, 8))
      << "RMS quantization offset or alignment is invalid";
  TVM_FFI_ICHECK(disjoint_storage(data, x) && disjoint_storage(data, residual) &&
                 disjoint_storage(data, weight) && disjoint_storage(data, summed) &&
                 disjoint_storage(data, scales) && disjoint_storage(summed, x) &&
                 disjoint_storage(summed, residual) && disjoint_storage(summed, weight) &&
                 disjoint_storage(summed, scales) && disjoint_storage(scales, x) &&
                 disjoint_storage(scales, residual) && disjoint_storage(scales, weight))
      << "RMS quantization outputs overlap inputs or each other";
  auto stream = stream_for(x, "rms_quantize");
#define RMS_QUANT(T, V, S)                                                                         \
  launch_kernel(norm_quant5120_kernel<true, T, V, S>, x.size(0), T, 0, stream,                     \
                static_cast<const __nv_bfloat16 *>(x.data_ptr()),                                  \
                static_cast<const __nv_bfloat16 *>(residual.data_ptr()),                           \
                static_cast<const float *>(weight.data_ptr()),                                     \
                static_cast<__nv_fp8_e4m3 *>(data.data_ptr()),                                     \
                static_cast<float *>(scales.data_ptr()), column,                                   \
                static_cast<int>(column ? scales.stride(1) : x.size(0)),                           \
                static_cast<__nv_bfloat16 *>(summed.data_ptr()), 1e-6f)
  if (x.size(0) >= 4096) {
    RMS_QUANT(320, 8, false);
  } else if (x.size(0) >= 2048) {
    RMS_QUANT(128, 4, true);
  } else {
    RMS_QUANT(256, 4, false);
  }
#undef RMS_QUANT
  finish_cuda_launch(stream, "rms_quantize");
}

// Keep the gated RMS reduction and its BF16 result rounding in registers.
// Each head is one complete 128-value FP8 group. The quantization keeps
// arbitrary FP32 scales and the previous corrected scale division.
template <int RowsPerWarp, bool Column>
__global__ void gated_quant128_kernel(const __nv_bfloat16 *__restrict__ x,
                                      const float *__restrict__ gamma,
                                      const __nv_bfloat16 *__restrict__ gate,
                                      __nv_fp8_e4m3 *__restrict__ data, float *__restrict__ scales,
                                      int tokens, int64_t x_stride, int64_t gate_stride) {
  pdl_dependency_wait();
  pdl_launch_next();
  int lane = threadIdx.x & 31;
  float weight[4];
#pragma unroll
  for (int j = 0; j < 4; ++j)
    weight[j] = gamma[lane + j * 32];
#pragma unroll
  for (int r = 0; r < RowsPerWarp; ++r) {
    int row = blockIdx.x * 4 * RowsPerWarp + (threadIdx.x / 32) * RowsPerWarp + r;
    if (row >= tokens * 48)
      continue;
    int token = row / 48, head = row % 48;
    int64_t offset = static_cast<int64_t>(token) * x_stride + head * 128;
    int64_t gate_offset = static_cast<int64_t>(token) * gate_stride + head * 128;
    float values[4], total = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float value = __bfloat162float(x[offset + lane + j * 32]);
      values[j] = value;
      total += value * value;
    }
    float inv = rsqrtf(warp_sum(total) * (1.f / 128.f) + 1e-6f), maximum = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float g = __bfloat162float(gate[gate_offset + lane + j * 32]);
      float e = __expf(-fabsf(g));
      float sigmoid = __fdividef(g >= 0.f ? 1.f : e, 1.f + e);
      float value = values[j] * inv * weight[j];
      values[j] = __bfloat162float(__float2bfloat16_rn(value * (g * sigmoid)));
      maximum = fmaxf(maximum, fabsf(values[j]));
    }
    maximum = __uint_as_float(__reduce_max_sync(0xffffffff, __float_as_uint(maximum)));
    float scale = __fdiv_rn(fmaxf(maximum, 1e-10f), 448.f);
    if (lane == 0)
      scales[Column ? head * tokens + token : row] = scale;
    float inverse;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(inverse) : "f"(scale));
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float divided;
      if (isfinite(maximum)) {
        float magnitude = fabsf(values[j]), initial = magnitude * inverse;
        float corrected = __fmaf_rn(__fmaf_rn(-initial, scale, magnitude), inverse, initial);
        divided = copysignf(corrected, values[j]);
      } else {
        divided = __fdiv_rn(values[j], scale);
      }
      divided = fminf(448.f, fmaxf(-448.f, divided));
      data[static_cast<int64_t>(row) * 128 + lane + j * 32] = __nv_fp8_e4m3(divided);
    }
  }
}

void gated_quantize(TensorView x, TensorView gamma, TensorView gate, TensorView data,
                    TensorView scales, bool column) {
  for (auto tensor : {gamma, gate, data, scales})
    TVM_FFI_ICHECK(same_cuda_device(tensor, x)) << "gated quantization device mismatch";
  TVM_FFI_ICHECK(x.ndim() == 3 && x.size(0) > 0 && x.size(1) == 48 && x.size(2) == 128 &&
                 gate.ndim() == 3 && gate.size(0) == x.size(0) && gate.size(1) == 48 &&
                 gate.size(2) == 128 && gamma.ndim() == 1 && gamma.size(0) == 128 &&
                 data.ndim() == 2 && data.size(0) == x.size(0) && data.size(1) == 6144 &&
                 scales.ndim() == 2 && scales.size(0) == x.size(0) && scales.size(1) == 48)
      << "gated quantization requires model heads, output data, and scales";
  TVM_FFI_ICHECK(has_dtype(x, kDLBfloat, 16) && has_dtype(gate, kDLBfloat, 16) &&
                 has_dtype(gamma, kDLFloat, 32) && has_dtype(data, kDLFloat8_e4m3fn, 8) &&
                 has_dtype(scales, kDLFloat, 32))
      << "gated quantization requires BF16 input/gate, FP8 data, and FP32 gamma/scales";
  TVM_FFI_ICHECK(x.stride(0) >= 6144 && x.stride(1) == 128 && x.stride(2) == 1 &&
                 gate.stride(0) >= 6144 && gate.stride(1) == 128 && gate.stride(2) == 1 &&
                 gamma.stride(0) == 1 && data.stride(0) == 6144 && data.stride(1) == 1 &&
                 scales.stride(0) == (column ? 1 : 48) &&
                 scales.stride(1) == (column ? x.size(0) : 1))
      << "gated quantization layout mismatch";
  constexpr int64_t max_offset = (int64_t(1) << 62) - 1;
  TVM_FFI_ICHECK(fits_int32_flat_offsets(data, false) && x.stride(0) <= max_offset / x.size(0) &&
                 gate.stride(0) <= max_offset / x.size(0) && aligned(x, 2) && aligned(gate, 2) &&
                 aligned(gamma, 4) && aligned(scales, 4))
      << "gated quantization offset or alignment is invalid";
  for (auto input : {x, gamma, gate})
    TVM_FFI_ICHECK(disjoint_storage(data, input) && disjoint_storage(scales, input))
        << "gated quantization outputs overlap inputs";
  TVM_FFI_ICHECK(disjoint_storage(data, scales)) << "gated quantization outputs overlap each other";
  auto stream = stream_for(x, "gated_quantize");
#define GATED_QUANT(R, C)                                                                          \
  launch_kernel(gated_quant128_kernel<R, C>, (x.size(0) * 48 + 4 * R - 1) / (4 * R), 128, 0,       \
                stream, static_cast<const __nv_bfloat16 *>(x.data_ptr()),                          \
                static_cast<const float *>(gamma.data_ptr()),                                      \
                static_cast<const __nv_bfloat16 *>(gate.data_ptr()),                               \
                static_cast<__nv_fp8_e4m3 *>(data.data_ptr()),                                     \
                static_cast<float *>(scales.data_ptr()), static_cast<int>(x.size(0)), x.stride(0), \
                gate.stride(0))
  if (column) {
    GATED_QUANT(1, true);
  } else if (x.size(0) >= 32144) {
    GATED_QUANT(4, false);
  } else {
    GATED_QUANT(2, false);
  }
#undef GATED_QUANT
  finish_cuda_launch(stream, "gated_quantize");
  record_variant(kGatedNormFp8Linear, true);
}
