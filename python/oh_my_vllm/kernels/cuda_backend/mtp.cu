#include "common.cuh"
#include "linear.cuh"

template <int Threads>
__global__ void
mtp_norm_kernel(const __nv_bfloat16 *hidden, const int64_t *tokens, const __nv_bfloat16 *embedding,
                const float *norm_hidden, const float *norm_embedding, const int32_t *hidden_rows,
                const __nv_bfloat16 *boundary_hidden, const int32_t *boundary_rows,
                __nv_bfloat16 *packed, __nv_bfloat16 *zero, int hidden_count, int boundaries) {
  int row = blockIdx.x, kind = blockIdx.y, tid = threadIdx.x;
  int source = hidden_rows[row], boundary = boundary_rows[row];
  int64_t token = tokens[row];
  if (source < -1 || source >= hidden_count || boundary < 0 || boundary >= boundaries ||
      token < 0 || token >= 248320) {
    asm volatile("trap;");
    return;
  }
  const __nv_bfloat16 *input = kind == 0     ? embedding + token * 5120
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

void mtp_input(TensorView hidden, TensorView tokens, TensorView embedding, TensorView norm_hidden,
               TensorView norm_embedding, TensorView weight, TensorView hidden_rows,
               TensorView boundary_hidden, TensorView boundary_rows, TensorView packed,
               TensorView output, TensorView zero) {
  TVM_FFI_ICHECK(hidden.ndim() == 2 && hidden.size(0) > 0 && hidden.size(0) <= INT32_MAX &&
                 hidden.size(1) == 5120);
  TVM_FFI_ICHECK(tokens.ndim() == 1 && tokens.size(0) > 0 && tokens.size(0) <= 32768);
  TVM_FFI_ICHECK(boundary_hidden.ndim() == 2 && boundary_hidden.size(0) > 0 &&
                 boundary_hidden.size(0) <= 32768);
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
  tensor_contract(packed, hidden, kDLBfloat, 16, {rows, 10240});
  for (auto tensor : {output, zero})
    tensor_contract(tensor, hidden, kDLBfloat, 16, {rows, 5120});
  auto inputs = {hidden, tokens,      embedding,       norm_hidden,  norm_embedding,
                 weight, hidden_rows, boundary_hidden, boundary_rows};
  auto outputs = {packed, output, zero};
  for (auto a : outputs)
    for (auto b : inputs)
      disjoint(a, b);
  disjoint(packed, output);
  disjoint(packed, zero);
  disjoint(output, zero);
  tvm::ffi::CUDADeviceGuard guard(hidden.device().device_id);
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, hidden.device().device_id));
  auto normalize = [&](auto threads) {
    constexpr int Threads = decltype(threads)::value;
    mtp_norm_kernel<Threads><<<dim3(rows, 2), Threads, 0, stream>>>(
        static_cast<const __nv_bfloat16 *>(hidden.data_ptr()),
        static_cast<const int64_t *>(tokens.data_ptr()),
        static_cast<const __nv_bfloat16 *>(embedding.data_ptr()),
        static_cast<const float *>(norm_hidden.data_ptr()),
        static_cast<const float *>(norm_embedding.data_ptr()),
        static_cast<const int32_t *>(hidden_rows.data_ptr()),
        static_cast<const __nv_bfloat16 *>(boundary_hidden.data_ptr()),
        static_cast<const int32_t *>(boundary_rows.data_ptr()),
        static_cast<__nv_bfloat16 *>(packed.data_ptr()),
        static_cast<__nv_bfloat16 *>(zero.data_ptr()), hidden_count, boundaries);
  };
  if (rows <= 32)
    normalize(std::integral_constant<int, 640>{});
  else
    normalize(std::integral_constant<int, 256>{});
  check_cuda(cudaGetLastError());
  bf16_linear(static_cast<const __nv_bfloat16 *>(packed.data_ptr()),
              static_cast<const __nv_bfloat16 *>(weight.data_ptr()), output.data_ptr(), rows, 5120,
              10240, stream);
}

__global__ void mtp_endpoints_kernel(const __nv_bfloat16 *value, const __nv_bfloat16 *zero,
                                     const int32_t *sequence_rows, __nv_bfloat16 *endpoints,
                                     __nv_bfloat16 *endpoint_zero, int rows, int batch) {
  int request = blockIdx.x, first = sequence_rows[request], last = sequence_rows[request + 1];
  if (sequence_rows[0] != 0 || sequence_rows[batch] != rows || first < 0 || last <= first ||
      last > rows) {
    asm volatile("trap;");
    return;
  }
  for (int column = threadIdx.x; column < 5120; column += blockDim.x) {
    endpoints[request * 5120 + column] = value[int64_t(last - 1) * 5120 + column];
    endpoint_zero[request * 5120 + column] = zero[int64_t(last - 1) * 5120 + column];
  }
}

void mtp_endpoints(TensorView value, TensorView zero, TensorView sequence_rows,
                   TensorView endpoints, TensorView endpoint_zero) {
  TVM_FFI_ICHECK(value.ndim() == 2 && value.size(0) > 0 && value.size(0) <= 32768 &&
                 value.size(1) == 5120);
  TVM_FFI_ICHECK(sequence_rows.ndim() == 1 && sequence_rows.size(0) > 1 &&
                 sequence_rows.size(0) <= value.size(0) + 1);
  int rows = value.size(0), batch = sequence_rows.size(0) - 1;
  for (auto tensor : {value, zero})
    tensor_contract(tensor, value, kDLBfloat, 16, {rows, 5120});
  tensor_contract(sequence_rows, value, kDLInt, 32, {batch + 1});
  for (auto tensor : {endpoints, endpoint_zero}) {
    tensor_contract(tensor, value, kDLBfloat, 16, {batch, 5120});
    for (auto input : {value, zero, sequence_rows})
      disjoint(tensor, input);
  }
  disjoint(endpoints, endpoint_zero);
  tvm::ffi::CUDADeviceGuard guard(value.device().device_id);
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, value.device().device_id));
  mtp_endpoints_kernel<<<batch, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16 *>(value.data_ptr()),
      static_cast<const __nv_bfloat16 *>(zero.data_ptr()),
      static_cast<const int32_t *>(sequence_rows.data_ptr()),
      static_cast<__nv_bfloat16 *>(endpoints.data_ptr()),
      static_cast<__nv_bfloat16 *>(endpoint_zero.data_ptr()), rows, batch);
  check_cuda(cudaGetLastError());
}
