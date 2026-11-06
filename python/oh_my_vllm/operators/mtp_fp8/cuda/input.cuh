#pragma once

#include "../../primitives/cuda/linear.cuh"
#include "projection.cuh"

namespace mtp_fp8 {
template <int Threads>
__global__ void
norm_kernel(const __nv_bfloat16 *hidden, const int64_t *tokens, const __nv_bfloat16 *embedding,
            const float *norm_hidden, const float *norm_embedding, const int32_t *hidden_rows,
            const __nv_bfloat16 *boundary_hidden, const int32_t *boundary_rows,
            const __nv_bfloat16 *media, const int32_t *media_rows, bool use_media, int media_count,
            __nv_bfloat16 *packed, __nv_bfloat16 *zero, int hidden_count, int boundaries) {
  int row = blockIdx.x, kind = blockIdx.y, tid = threadIdx.x;
  int source = hidden_rows[row], boundary = boundary_rows[row];
  int64_t token = tokens[row];
  if (source < -1 || source >= hidden_count || boundary < 0 || boundary >= boundaries ||
      token < 0 || token >= 248320) {
    asm volatile("trap;");
    return;
  }
  int media_row = use_media ? media_rows[row] : -1;
  if (media_row < -1 || media_row >= media_count) {
    asm volatile("trap;");
    return;
  }
  const __nv_bfloat16 *input =
      kind == 0 ? (media_row >= 0 ? media + int64_t(media_row) * 5120 : embedding + token * 5120)
      : source >= 0 ? hidden + int64_t(source) * 5120
                    : boundary_hidden + int64_t(boundary) * 5120;
  const float *weight = kind == 0 ? norm_embedding : norm_hidden;
  constexpr int Values = 5120 / Threads, Warps = Threads / 32;
  float values[Values];
  float total = 0;
#pragma unroll
  for (int j = 0; j < Values; ++j) {
    values[j] = __bfloat162float(input[tid + j * Threads]);
    total += values[j] * values[j];
  }
  __shared__ float scratch[Warps];
  total = warp_sum(total);
  if ((tid & 31) == 0)
    scratch[tid / 32] = total;
  __syncthreads();
  total = warp_sum((tid & 31) < Warps ? scratch[tid & 31] : 0.f);
  float inverse = rsqrtf(total / 5120.f + 1e-6f);
#pragma unroll
  for (int j = 0; j < Values; ++j) {
    int col = tid + j * Threads;
    packed[int64_t(row) * 10240 + kind * 5120 + col] =
        __float2bfloat16_rn(values[j] * inverse * (weight[col] + 1.f));
    if (kind == 0)
      zero[int64_t(row) * 5120 + col] = __float2bfloat16_rn(0.f);
  }
}

__global__ void merge_kernel(const float *partial, __nv_bfloat16 *output, int count) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    float value = 0;
#pragma unroll
    for (int split = 0; split < 4; ++split)
      value += partial[split * count + index];
    output[index] = __float2bfloat16_rn(value);
  }
}
} // namespace mtp_fp8

void mtp_fp8_input_contract(TensorView hidden, TensorView tokens, TensorView embedding,
                            TensorView norm_hidden, TensorView norm_embedding, TensorView weight,
                            TensorView hidden_rows, TensorView boundary_hidden,
                            TensorView boundary_rows, TensorView media, TensorView media_rows,
                            bool use_media, TensorView packed, TensorView output, TensorView zero,
                            TensorView partials) {
  TVM_FFI_ICHECK(hidden.ndim() == 2 && hidden.size(0) >= 0 && hidden.size(0) <= INT32_MAX &&
                 hidden.size(1) == 5120);
  TVM_FFI_ICHECK(tokens.ndim() == 1 && tokens.size(0) > 0 && tokens.size(0) <= 32768);
  TVM_FFI_ICHECK(boundary_hidden.ndim() == 2 && boundary_hidden.size(0) > 0 &&
                 boundary_hidden.size(0) <= 65536);
  int rows = tokens.size(0), hidden_count = hidden.size(0), boundaries = boundary_hidden.size(0);
  tensor_contract(hidden, hidden, kDLBfloat, 16, {hidden_count, 5120});
  tensor_contract(tokens, hidden, kDLInt, 64, {rows});
  tensor_contract(embedding, hidden, kDLBfloat, 16, {248320, 5120});
  for (auto tensor : {norm_hidden, norm_embedding})
    tensor_contract(tensor, hidden, kDLFloat, 32, {5120});
  tensor_contract(weight, hidden, kDLBfloat, 16, {5120, 10240});
  for (auto tensor : {hidden_rows, boundary_rows})
    tensor_contract(tensor, hidden, kDLInt, 32, {rows});
  tensor_contract(boundary_hidden, hidden, kDLBfloat, 16, {boundaries, 5120});
  TVM_FFI_ICHECK(media.ndim() == 2 && media.size(0) >= 0 && media.size(0) <= 65536);
  tensor_contract(media, hidden, kDLBfloat, 16, {media.size(0), 5120});
  tensor_contract(media_rows, hidden, kDLInt, 32, {rows});
  tensor_contract(packed, hidden, kDLBfloat, 16, {rows, 10240});
  for (auto tensor : {output, zero})
    tensor_contract(tensor, hidden, kDLBfloat, 16, {rows, 5120});
  auto inputs = {hidden,      tokens,          embedding,     norm_hidden, norm_embedding, weight,
                 hidden_rows, boundary_hidden, boundary_rows, media,       media_rows};
  tensor_contract(partials, hidden, kDLFloat, 32, {rows <= 32 ? 4 : 0, rows, 5120});
  auto outputs = {packed, output, zero, partials};
  for (auto a : outputs)
    for (auto b : inputs)
      disjoint(a, b);
  disjoint(packed, output);
  disjoint(packed, zero);
  disjoint(output, zero);
  for (auto tensor : {packed, output, zero})
    disjoint(partials, tensor);
}

void mtp_fp8_input(TensorView hidden, TensorView tokens, TensorView embedding,
                   TensorView norm_hidden, TensorView norm_embedding, TensorView weight,
                   TensorView hidden_rows, TensorView boundary_hidden, TensorView boundary_rows,
                   TensorView media, TensorView media_rows, bool use_media, TensorView packed,
                   TensorView output, TensorView zero, TensorView partials) {
  mtp_fp8_input_contract(hidden, tokens, embedding, norm_hidden, norm_embedding, weight,
                         hidden_rows, boundary_hidden, boundary_rows, media, media_rows, use_media,
                         packed, output, zero, partials);
  int rows = tokens.size(0), hidden_count = hidden.size(0), boundaries = boundary_hidden.size(0);
  tvm::ffi::CUDADeviceGuard guard(hidden.device().device_id);
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, hidden.device().device_id));
  auto normalize = [&](auto threads) {
    constexpr int Threads = decltype(threads)::value;
    mtp_fp8::norm_kernel<Threads><<<dim3(rows, 2), Threads, 0, stream>>>(
        static_cast<const __nv_bfloat16 *>(hidden.data_ptr()),
        static_cast<const int64_t *>(tokens.data_ptr()),
        static_cast<const __nv_bfloat16 *>(embedding.data_ptr()),
        static_cast<const float *>(norm_hidden.data_ptr()),
        static_cast<const float *>(norm_embedding.data_ptr()),
        static_cast<const int32_t *>(hidden_rows.data_ptr()),
        static_cast<const __nv_bfloat16 *>(boundary_hidden.data_ptr()),
        static_cast<const int32_t *>(boundary_rows.data_ptr()),
        static_cast<const __nv_bfloat16 *>(media.data_ptr()),
        static_cast<const int32_t *>(media_rows.data_ptr()), use_media, media.size(0),
        static_cast<__nv_bfloat16 *>(packed.data_ptr()),
        static_cast<__nv_bfloat16 *>(zero.data_ptr()), hidden_count, boundaries);
  };
  if (rows <= 32)
    normalize(std::integral_constant<int, 1024>{});
  else
    normalize(std::integral_constant<int, 256>{});
  check_cuda(cudaGetLastError());
  if (rows <= 32) {
    mtp_fp8::project(packed, weight, partials, rows, stream);
    mtp_fp8::merge_kernel<<<(rows * 5120 + 255) / 256, 256, 0, stream>>>(
        static_cast<const float *>(partials.data_ptr()),
        static_cast<__nv_bfloat16 *>(output.data_ptr()), rows * 5120);
    check_cuda(cudaGetLastError());
  } else if (rows >= 4096) {
    bf16_gemm<128, 256, false, cutlass::bfloat16_t>(
        static_cast<const __nv_bfloat16 *>(packed.data_ptr()),
        static_cast<const __nv_bfloat16 *>(weight.data_ptr()), output.data_ptr(), rows, 5120, 10240,
        stream);
  } else {
    bf16_linear(static_cast<const __nv_bfloat16 *>(packed.data_ptr()),
                static_cast<const __nv_bfloat16 *>(weight.data_ptr()), output.data_ptr(), rows,
                5120, 10240, stream);
  }
}
