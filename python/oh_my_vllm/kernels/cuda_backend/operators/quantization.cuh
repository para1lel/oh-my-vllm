#pragma once

#include "common.cuh"

// BF16 SiLU keeps both rounding boundaries. The fast exponential produces
// identical rounded SiLU for every finite BF16 input on the required SM100.
// One warp owns one complete 128-value scaling group. Independent warps share
// a CTA; reductions do not require shared memory or block-wide synchronization.
template <typename Input, bool Silu, bool Column, int RowsPerWarp, bool Flat = false, int Width = 0,
          bool Aligned = false>
__global__ void quantize_kernel(const Input *__restrict__ x, __nv_fp8_e4m3 *__restrict__ out,
                                float *__restrict__ scales, int rows, int runtime_width) {
  pdl_dependency_wait();
  pdl_launch_next();
  const int width = Width ? Width : runtime_width;
  const int lane = threadIdx.x & 31;
  const int groups = width / 128;
  const int linear_group = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  const int group = Flat ? linear_group % groups : blockIdx.y;
  const int col = group * 128;
#pragma unroll
  for (int r = 0; r < RowsPerWarp; ++r) {
    const int row =
        Flat ? linear_group / groups
             : blockIdx.x * (blockDim.x / 32) * RowsPerWarp + (threadIdx.x / 32) * RowsPerWarp + r;
    if (row >= rows)
      continue;
    float value[4], maximum = 0;
    int offset = row * width * (Silu ? 2 : 1) + col + lane * 4;
    Four<Input> packed;
    packed = load_four<Input, Aligned>(x + offset);
    Four<Input> up;
    if constexpr (Silu)
      up = load_four<Input, Aligned>(x + offset + width);
    if constexpr (Silu) {
#pragma unroll
      for (int j = 0; j < 4; j += 2) {
        float x0 = static_cast<float>(packed.values[j]);
        float x1 = static_cast<float>(packed.values[j + 1]);
        auto activated = __floats2bfloat162_rn(x0 / (1.f + __expf(-x0)), x1 / (1.f + __expf(-x1)));
        auto product =
            __hmul2(activated,
                    __halves2bfloat162(__float2bfloat16_rn(static_cast<float>(up.values[j])),
                                       __float2bfloat16_rn(static_cast<float>(up.values[j + 1]))));
        auto pair = __bfloat1622float2(product);
        value[j] = pair.x;
        value[j + 1] = pair.y;
        maximum = fmaxf(maximum, fmaxf(fabsf(pair.x), fabsf(pair.y)));
      }
    } else {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        float v = static_cast<float>(packed.values[j]);
        value[j] = v;
        maximum = fmaxf(maximum, fabsf(v));
      }
    }
    // Absolute nonnegative FP32 bit patterns preserve unsigned ordering.
    maximum = __uint_as_float(__reduce_max_sync(0xffffffff, __float_as_uint(maximum)));
    float scale = __fdiv_rn(fmaxf(maximum, 1e-10f), 448.f);
    if (lane == 0)
      scales[Column ? group * rows + row : row * groups + group] = scale;
    // BF16 scale and reciprocal stay normal throughout the finite domain.
    // One residual-refinement step uses two FMA operations and preserves
    // FP8 rounding; FP16/FP32 and nonfinite maxima retain exact division.
    // This is not FP32 equivalence.
    float inverse = 0.f;
    if constexpr (std::is_same_v<Input, __nv_bfloat16>)
      asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(inverse) : "f"(scale));
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float divided;
      if constexpr (std::is_same_v<Input, __nv_bfloat16>) {
        if (isfinite(maximum)) {
          float magnitude = fabsf(value[j]);
          float initial = magnitude * inverse;
          float corrected = __fmaf_rn(__fmaf_rn(-initial, scale, magnitude), inverse, initial);
          divided = copysignf(corrected, value[j]);
        } else {
          divided = __fdiv_rn(value[j], scale);
        }
      } else {
        divided = __fdiv_rn(value[j], scale);
      }
      value[j] = fminf(448.f, fmaxf(-448.f, divided));
    }
    reinterpret_cast<__nv_fp8x4_e4m3 *>(out)[(row * width + col) / 4 + lane] =
        __nv_fp8x4_e4m3(make_float4(value[0], value[1], value[2], value[3]));
  }
}

template <typename Input, int Width, bool Aligned>
void launch_quantize(TensorView x, TensorView out, TensorView scales, bool column, bool silu,
                     cudaStream_t stream) {
  int rows = out.size(0), width = out.size(1);
#define CALL(S, C, R, T)                                                                           \
  launch_kernel(quantize_kernel<Input, S, C, R, false, Width, Aligned>,                            \
                dim3((rows + (T / 32) * R - 1) / ((T / 32) * R), width / 128), T, 0, stream,       \
                static_cast<const Input *>(x.data_ptr()),                                          \
                static_cast<__nv_fp8_e4m3 *>(out.data_ptr()),                                      \
                static_cast<float *>(scales.data_ptr()), rows, width)
#define LAUNCH(S, C)                                                                               \
  if (width / 128 > 65535 || (rows >= 4 && rows < 128) ||                                          \
      (rows == 1 && width == 6144 && C && !S)) {                                                   \
    launch_kernel(quantize_kernel<Input, S, C, 1, true, Width, Aligned>,                           \
                  (rows * (width / 128) + 3) / 4, 128, 0, stream,                                  \
                  static_cast<const Input *>(x.data_ptr()),                                        \
                  static_cast<__nv_fp8_e4m3 *>(out.data_ptr()),                                    \
                  static_cast<float *>(scales.data_ptr()), rows, width);                           \
  } else if (S && rows >= 4096 && Width == 17408) {                                                \
    CALL(S, C, 4, 128);                                                                            \
  } else if (S && rows >= 128 && Width == 17408) {                                                 \
    CALL(S, C, 2, 128);                                                                            \
  } else if (S && rows >= 128) {                                                                   \
    CALL(S, C, 1, 128);                                                                            \
  } else if (rows >= 128) {                                                                        \
    CALL(S, C, 4, 128);                                                                            \
  } else {                                                                                         \
    CALL(S, C, 1, 32);                                                                             \
  }
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
#undef CALL
}

template <typename Input, bool Aligned>
void dispatch(TensorView x, TensorView out, TensorView scales, bool column, bool silu,
              cudaStream_t stream) {
  switch (out.size(1)) {
  case 5120:
    launch_quantize<Input, 5120, Aligned>(x, out, scales, column, silu, stream);
    break;
  case 6144:
    launch_quantize<Input, 6144, Aligned>(x, out, scales, column, silu, stream);
    break;
  case 17408:
    launch_quantize<Input, 17408, Aligned>(x, out, scales, column, silu, stream);
    break;
  default:
    launch_quantize<Input, 0, Aligned>(x, out, scales, column, silu, stream);
  }
}
template <typename Input>
void dispatch_alignment(TensorView x, TensorView out, TensorView scales, bool column, bool silu,
                        cudaStream_t stream) {
  // Contiguous rows and128-value groups keep every four-element load aligned
  // whenever the input base is aligned. Storage-offset views retain scalar loads.
  if ((reinterpret_cast<uintptr_t>(x.data_ptr()) & (sizeof(Input) * 4 - 1)) == 0)
    dispatch<Input, true>(x, out, scales, column, silu, stream);
  else
    dispatch<Input, false>(x, out, scales, column, silu, stream);
}
void quantize(TensorView x, TensorView out, TensorView scales, bool column, bool silu) {
  TVM_FFI_ICHECK(same_cuda_device(x, x) && same_cuda_device(out, x) && same_cuda_device(scales, x))
      << "CUDA quantize tensors must share one CUDA device";
  const auto dtype = x.dtype();
  TVM_FFI_ICHECK(dtype.lanes == 1 &&
                 ((dtype.code == kDLBfloat && dtype.bits == 16) ||
                  (dtype.code == kDLFloat && (dtype.bits == 16 || dtype.bits == 32))))
      << "CUDA quantize requires BF16, FP16, or FP32 input";
  TVM_FFI_ICHECK(!silu || has_dtype(x, kDLBfloat, 16))
      << "CUDA fused SiLU quantize requires BF16 input";
  TVM_FFI_ICHECK(has_dtype(out, kDLFloat8_e4m3fn, 8) && has_dtype(scales, kDLFloat, 32))
      << "CUDA quantize requires FP8 output and FP32 scales";
  TVM_FFI_ICHECK(x.ndim() == 2 && out.ndim() == 2 && scales.ndim() == 2 &&
                 out.size(0) == x.size(0) && out.size(0) > 0 && x.size(1) > 0 && out.size(1) > 0 &&
                 out.size(1) % 128 == 0 && x.size(1) % (silu ? 2 : 1) == 0 &&
                 x.size(1) / (silu ? 2 : 1) == out.size(1) && scales.size(0) == out.size(0) &&
                 scales.size(1) == out.size(1) / 128 && x.stride(1) == 1 &&
                 x.stride(0) == x.size(1) && out.stride(1) == 1 && out.stride(0) == out.size(1) &&
                 scales.stride(0) == (column ? 1 : scales.size(1)) &&
                 scales.stride(1) == (column ? out.size(0) : 1))
      << "CUDA quantize output shape or layout is invalid";
  TVM_FFI_ICHECK(fits_int32_flat_offsets(x, false) && fits_int32_flat_offsets(out, true))
      << "CUDA quantize input/output flat offsets exceed signed int32";
  auto stream = stream_for(x, "quantize");
  if (dtype.code == kDLBfloat)
    dispatch_alignment<__nv_bfloat16>(x, out, scales, column, silu, stream);
  else if (dtype.bits == 16)
    dispatch_alignment<__half>(x, out, scales, column, silu, stream);
  else
    dispatch_alignment<float>(x, out, scales, column, silu, stream);
  finish_cuda_launch(stream, "quantize");
}
__global__ void silu_kernel(const __nv_bfloat16 *__restrict__ x, __nv_bfloat16 *__restrict__ out,
                            int rows, int width) {
  pdl_dependency_wait();
  pdl_launch_next();
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * width)
    return;
  int offset = (i / width) * width * 2 + i % width;
  float g = __bfloat162float(x[offset]);
  float a = __bfloat162float(__float2bfloat16_rn(g / (1.f + __expf(-g))));
  out[i] = __float2bfloat16_rn(a * __bfloat162float(x[offset + width]));
}
void silu_mul(TensorView x, TensorView out) {
  TVM_FFI_ICHECK(same_cuda_device(x, x) && same_cuda_device(out, x))
      << "CUDA SiLU tensors must share one CUDA device";
  TVM_FFI_ICHECK(x.ndim() == 2 && out.ndim() == 2 && x.size(0) == out.size(0) && x.size(0) >= 0 &&
                 x.size(1) >= 0 && x.size(1) % 2 == 0 && out.size(1) == x.size(1) / 2 &&
                 has_dtype(x, kDLBfloat, 16) && has_dtype(out, kDLBfloat, 16))
      << "CUDA SiLU requires matching BF16 packed input and output";
  if (out.size(0) == 0 || out.size(1) == 0)
    return;
  TVM_FFI_ICHECK(x.stride(1) == 1 && x.stride(0) == x.size(1) && out.stride(1) == 1 &&
                 out.stride(0) == out.size(1))
      << "CUDA SiLU requires contiguous input and output";
  TVM_FFI_ICHECK(fits_int32_flat_offsets(x, false) && fits_int32_flat_offsets(out, true))
      << "CUDA SiLU input/output flat offsets exceed signed int32";
  auto stream = stream_for(x, "silu_mul");
  int n = out.size(0) * out.size(1);
  if (!n)
    return;
  launch_kernel(silu_kernel, (n + 255) / 256, 256, 0, stream,
                static_cast<const __nv_bfloat16 *>(x.data_ptr()),
                static_cast<__nv_bfloat16 *>(out.data_ptr()), out.size(0), out.size(1));
  finish_cuda_launch(stream, "silu_mul");
}
