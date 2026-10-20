#pragma once
#include "mtp_context_kernels.cuh"
#include <map>
#include <tuple>

namespace mtp_context_native {

struct LaunchState {
  alignas(128) CUtensorMap fc;
  alignas(128) CUtensorMap kv_four;
  alignas(128) CUtensorMap kv_one;
  int multiprocessors = 0;
  bool configured[2] = {};
};

template <int Rows>
bool run(const Arguments &a, TensorView fc_weight, TensorView kv_weight, cudaStream_t stream) {
  constexpr int Capacity = Rows == 4 ? 8 : 16;
  constexpr int Shared = 65536 + Capacity * 512 + 128;
  constexpr int Which = Rows == 4 ? 0 : 1;
  using Key = std::tuple<int, uintptr_t, uintptr_t>;
  static thread_local std::map<Key, LaunchState> states;
  int device = fc_weight.device().device_id;
  Key key{device, reinterpret_cast<uintptr_t>(fc_weight.data_ptr()),
          reinterpret_cast<uintptr_t>(kv_weight.data_ptr())};
  auto found = states.find(key);
  cudaStreamCaptureStatus capture;
  check_cuda(cudaStreamIsCapturing(stream, &capture));
  if (capture != cudaStreamCaptureStatusNone &&
      (found == states.end() || !found->second.configured[Which]))
    return false;
  if (found == states.end()) {
    LaunchState state{};
    check_cuda(
        cudaDeviceGetAttribute(&state.multiprocessors, cudaDevAttrMultiProcessorCount, device));
    TVM_FFI_ICHECK(state.multiprocessors > 0);
    uint32_t elements[3] = {1, 1, 1};
    uint64_t fc_dimensions[3] = {64, 5120, 160}, fc_strides[2] = {20480, 128};
    uint32_t fc_box[3] = {64, 64, 4};
    auto status = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
        &state.fc, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, fc_weight.data_ptr(), fc_dimensions,
        fc_strides, fc_box, elements, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TVM_FFI_ICHECK(status == CUDA_SUCCESS) << status;
    uint64_t kv_dimensions[3] = {128, 2048, 40}, kv_strides[2] = {5120, 128};
    auto encode_kv = [&](CUtensorMap *map, uint32_t groups) {
      uint32_t box[3] = {128, 64, groups};
      auto encoded = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
          map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, kv_weight.data_ptr(), kv_dimensions, kv_strides,
          box, elements, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
          CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      TVM_FFI_ICHECK(encoded == CUDA_SUCCESS) << encoded;
    };
    encode_kv(&state.kv_four, 4);
    encode_kv(&state.kv_one, 1);
    found = states.emplace(key, state).first;
  }
  auto &state = found->second;
  if (!state.configured[Which]) {
    auto check = [&](auto kernel, int threads, int shared, int tmem_columns, size_t local_limit) {
      if (shared) {
        check_cuda(
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared));
        check_cuda(
            cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
      }
      cudaFuncAttributes attributes{};
      check_cuda(cudaFuncGetAttributes(&attributes, kernel));
      TVM_FFI_ICHECK(attributes.maxThreadsPerBlock >= threads &&
                     attributes.localSizeBytes <= local_limit);
      int active = 0;
      check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, kernel, threads, shared));
      TVM_FFI_ICHECK(active >= 1) << "MTP context kernel cannot reside";
      if (tmem_columns)
        TVM_FFI_ICHECK(active * tmem_columns <= 512) << "ordinary TMEM allocation cannot reside";
    };
    check(mtp_norm_kernel<1024>, 1024, 0, 0, 0);
    check(fc_kernel<Capacity>, 128, Shared, 32, 0);
    check(middle_kernel<Rows>, 640, 0, 0, 0);
    check(kv_kernel<Rows>, 128, Shared, 4 * Capacity, 0);
    check(heads_kernel, 32, 0, 0, 0);
    state.configured[Which] = true;
  }
  mtp_norm_kernel<1024><<<dim3(Rows, 2), 1024, 0, stream>>>(
      a.hidden, a.tokens, a.embedding, a.norm_hidden, a.norm_embedding, a.hidden_rows,
      a.boundary_hidden, a.boundary_rows, a.media, a.media_rows, a.use_media, a.media_count,
      a.mtp_packed, a.zero, a.hidden_count, a.boundaries);
  check_cuda(cudaGetLastError());
  fc_kernel<Capacity>
      <<<dim3(80, 4), 128, Shared, stream>>>(a.mtp_packed, a.fc_partials, Rows, state.fc);
  check_cuda(cudaGetLastError());
  middle_kernel<Rows><<<Rows, 640, 0, stream>>>(a);
  check_cuda(cudaGetLastError());
  kv_kernel<Rows><<<dim3(32, 8), 128, Shared, stream>>>(a.quantized, a.scales, a.weight_scale,
                                                        a.kv_partials, state.kv_four, state.kv_one);
  check_cuda(cudaGetLastError());
  heads_kernel<<<dim3(Rows, 4), 32, 0, stream>>>(a.kv_partials, a.kv_packed, a.k_norm, a.positions,
                                                 a.slots, a.cache, Rows,
                                                 int64_t(a.cache_pages) * 784);
  check_cuda(cudaGetLastError());
  return true;
}

} // namespace mtp_context_native

bool mtp_context(TensorView hidden, TensorView tokens, TensorView embedding, TensorView norm_hidden,
                 TensorView norm_embedding, TensorView fc_weight, TensorView hidden_rows,
                 TensorView boundary_hidden, TensorView boundary_rows, TensorView media,
                 TensorView media_rows, bool use_media, TensorView fa_norm, TensorView kv_weight,
                 TensorView kv_weight_scale, TensorView k_norm, TensorView positions,
                 TensorView cache, TensorView slots, TensorView tables, TensorView sequence_rows,
                 TensorView lengths, TensorView mtp_packed, TensorView fc_output, TensorView zero,
                 TensorView fc_partials, TensorView summed, TensorView quantized, TensorView scales,
                 TensorView kv_packed, TensorView kv_partials, TensorView endpoints,
                 TensorView endpoint_zero) {
  TVM_FFI_ICHECK(tokens.ndim() == 1 && (tokens.size(0) == 4 || tokens.size(0) == 16));
  TVM_FFI_ICHECK(hidden.ndim() == 2 && hidden.size(0) > 0 && hidden.size(0) <= INT32_MAX);
  TVM_FFI_ICHECK(boundary_hidden.ndim() == 2 && boundary_hidden.size(0) > 0 &&
                 boundary_hidden.size(0) <= 65536);
  TVM_FFI_ICHECK(media.ndim() == 2 && media.size(0) >= 0 && media.size(0) <= 65536);
  TVM_FFI_ICHECK(sequence_rows.ndim() == 1 && sequence_rows.size(0) > 1 &&
                 sequence_rows.size(0) <= tokens.size(0) + 1);
  TVM_FFI_ICHECK(cache.ndim() == 5 && cache.size(0) > 0 && cache.size(0) <= INT32_MAX / 784);
  int rows = tokens.size(0), batch = sequence_rows.size(0) - 1;
  TVM_FFI_ICHECK((rows == 4 && batch == 1) || (rows == 16 && batch == 4));
  TVM_FFI_ICHECK(tables.ndim() == 2 && tables.size(0) == batch && tables.size(1) > 0 &&
                 tables.size(1) <= 335);
  int hidden_count = hidden.size(0), boundaries = boundary_hidden.size(0);
  tensor_contract(hidden, hidden, kDLBfloat, 16, {hidden_count, 5120});
  tensor_contract(tokens, hidden, kDLInt, 64, {rows});
  tensor_contract(embedding, hidden, kDLBfloat, 16, {248320, 5120});
  tensor_contract(boundary_hidden, hidden, kDLBfloat, 16, {boundaries, 5120});
  tensor_contract(media, hidden, kDLBfloat, 16, {media.size(0), 5120});
  for (auto tensor : {norm_hidden, norm_embedding, fa_norm})
    tensor_contract(tensor, hidden, kDLFloat, 32, {5120});
  tensor_contract(k_norm, hidden, kDLFloat, 32, {256});
  tensor_contract(fc_weight, hidden, kDLBfloat, 16, {5120, 10240});
  tensor_contract(kv_weight, hidden, kDLFloat8_e4m3fn, 8, {2048, 5120});
  tensor_contract(kv_weight_scale, hidden, kDLFloat, 32, {16, 40});
  for (auto tensor : {hidden_rows, boundary_rows, media_rows, slots})
    tensor_contract(tensor, hidden, kDLInt, 32, {rows});
  tensor_contract(positions, hidden, kDLInt, 64, {3, rows});
  tensor_contract(sequence_rows, hidden, kDLInt, 32, {batch + 1});
  tensor_contract(tables, hidden, kDLInt, 32, {batch, tables.size(1)});
  tensor_contract(lengths, hidden, kDLInt, 32, {rows});
  tensor_contract(cache, hidden, kDLBfloat, 16, {cache.size(0), 2, 784, 4, 256});
  tensor_contract(mtp_packed, hidden, kDLBfloat, 16, {rows, 10240});
  for (auto tensor : {fc_output, zero, summed})
    tensor_contract(tensor, hidden, kDLBfloat, 16, {rows, 5120});
  tensor_contract(fc_partials, hidden, kDLFloat, 32, {4, rows, 5120});
  tensor_contract(kv_partials, hidden, kDLFloat, 32, {8, rows, 5120});
  tensor_contract(quantized, hidden, kDLFloat8_e4m3fn, 8, {rows, 5120});
  tensor_contract(scales, hidden, kDLFloat, 32, {rows, 40});
  tensor_contract(kv_packed, hidden, kDLBfloat, 16, {rows, 2048});
  for (auto tensor : {endpoints, endpoint_zero})
    tensor_contract(tensor, hidden, kDLBfloat, 16, {batch, 5120});
  auto inputs = {hidden,     tokens,      embedding,       norm_hidden,     norm_embedding,
                 fc_weight,  hidden_rows, boundary_hidden, boundary_rows,   media,
                 media_rows, fa_norm,     kv_weight,       kv_weight_scale, k_norm,
                 positions,  slots,       tables,          sequence_rows,   lengths};
  auto outputs = {cache,     mtp_packed, fc_output, zero,        fc_partials, summed,
                  quantized, scales,     kv_packed, kv_partials, endpoints,   endpoint_zero};
  auto separate = [](TensorView a, TensorView b) {
    auto x = reinterpret_cast<uintptr_t>(a.data_ptr());
    auto y = reinterpret_cast<uintptr_t>(b.data_ptr());
    return x + tvm::ffi::GetDataSize(a) <= y || y + tvm::ffi::GetDataSize(b) <= x;
  };
  int left = 0;
  for (auto output : outputs) {
    for (auto input : inputs)
      if (!separate(output, input))
        return false;
    int right = 0;
    for (auto other : outputs)
      if (right++ > left && !separate(output, other))
        return false;
    ++left;
  }
  using namespace mtp_context_native;
  Arguments a{};
  a.hidden = static_cast<const __nv_bfloat16 *>(hidden.data_ptr());
  a.embedding = static_cast<const __nv_bfloat16 *>(embedding.data_ptr());
  a.boundary_hidden = static_cast<const __nv_bfloat16 *>(boundary_hidden.data_ptr());
  a.media = static_cast<const __nv_bfloat16 *>(media.data_ptr());
  a.norm_hidden = static_cast<const float *>(norm_hidden.data_ptr());
  a.norm_embedding = static_cast<const float *>(norm_embedding.data_ptr());
  a.fa_norm = static_cast<const float *>(fa_norm.data_ptr());
  a.weight_scale = static_cast<const float *>(kv_weight_scale.data_ptr());
  a.k_norm = static_cast<const float *>(k_norm.data_ptr());
  a.tokens = static_cast<const int64_t *>(tokens.data_ptr());
  a.positions = static_cast<const int64_t *>(positions.data_ptr());
  a.hidden_rows = static_cast<const int32_t *>(hidden_rows.data_ptr());
  a.boundary_rows = static_cast<const int32_t *>(boundary_rows.data_ptr());
  a.media_rows = static_cast<const int32_t *>(media_rows.data_ptr());
  a.slots = static_cast<const int32_t *>(slots.data_ptr());
  a.sequence_rows = static_cast<const int32_t *>(sequence_rows.data_ptr());
  a.mtp_packed = static_cast<__nv_bfloat16 *>(mtp_packed.data_ptr());
  a.fc_output = static_cast<__nv_bfloat16 *>(fc_output.data_ptr());
  a.zero = static_cast<__nv_bfloat16 *>(zero.data_ptr());
  a.summed = static_cast<__nv_bfloat16 *>(summed.data_ptr());
  a.kv_packed = static_cast<__nv_bfloat16 *>(kv_packed.data_ptr());
  a.cache = static_cast<__nv_bfloat16 *>(cache.data_ptr());
  a.endpoints = static_cast<__nv_bfloat16 *>(endpoints.data_ptr());
  a.endpoint_zero = static_cast<__nv_bfloat16 *>(endpoint_zero.data_ptr());
  a.quantized = static_cast<__nv_fp8_e4m3 *>(quantized.data_ptr());
  a.fc_partials = static_cast<float *>(fc_partials.data_ptr());
  a.kv_partials = static_cast<float *>(kv_partials.data_ptr());
  a.scales = static_cast<float *>(scales.data_ptr());
  a.rows = rows;
  a.batch = batch;
  a.hidden_count = hidden_count;
  a.boundaries = boundaries;
  a.media_count = media.size(0);
  a.cache_pages = cache.size(0);
  a.use_media = use_media;
  tvm::ffi::CUDADeviceGuard guard(hidden.device().device_id);
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, hidden.device().device_id));
  if (rows == 4)
    return run<4>(a, fc_weight, kv_weight, stream);
  else
    return run<16>(a, fc_weight, kv_weight, stream);
}
