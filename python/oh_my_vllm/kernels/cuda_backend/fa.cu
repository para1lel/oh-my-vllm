#include "common.cuh"
#include "projection_decode.cuh"
#include "groupwise.cuh"
#include "normalization.cuh"
#include "fa_attention.cuh"
#include "fa_input.cuh"
#include "fa_prefill.cuh"
#include "fa_small_input.cuh"

__global__ void fa_output_merge_kernel(const float *partials, __nv_bfloat16 *output, int count) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    float value = 0;
    for (int split = 0; split < 8; ++split)
      value += partials[split * count + index];
    output[index] = __float2bfloat16_rn(value);
  }
}

void fa(TensorView x, TensorView residual, TensorView norm, TensorView input_weight,
        TensorView input_weight_scale, TensorView q_norm, TensorView k_norm, TensorView positions,
        TensorView cache, TensorView slots, TensorView tables, TensorView sequence_rows,
        TensorView lengths, TensorView output_weight, TensorView output_weight_scale,
        TensorView output, TensorView summed, TensorView quantized, TensorView scales,
        TensorView packed, TensorView query, TensorView gate, TensorView partial, TensorView lse,
        TensorView gated, TensorView gated_scale, TensorView projection_partials, int64_t variant,
        int64_t max_queries, int64_t first_token) {
  TVM_FFI_ICHECK(x.ndim() == 2 && x.size(1) == 5120 && x.size(0) > 0 && x.size(0) <= 32768);
  int rows = x.size(0), columns = variant == 1 ? 2048 : variant == 2 ? 12288 : 14336;
  TVM_FFI_ICHECK(first_token == 0 || first_token == 1);
  TVM_FFI_ICHECK(variant >= 0 && variant <= 2 && max_queries > 0 && max_queries <= rows);
  TVM_FFI_ICHECK(tables.ndim() == 2 && tables.size(0) > 0 && tables.size(1) > 0);
  TVM_FFI_ICHECK(tables.size(0) <= rows && tables.size(1) <= 335);
  TVM_FFI_ICHECK(cache.ndim() == 5 && cache.size(0) > 0 && cache.size(0) <= INT32_MAX / 784);
  TVM_FFI_ICHECK(lse.ndim() == 3 && (lse.size(2) == 1 || lse.size(2) == 16 || lse.size(2) == 64 ||
                                     lse.size(2) == 128));
  int splits = lse.size(2), batch = tables.size(0), blocks = cache.size(0);
  for (auto tensor : {x, residual, output, summed})
    tensor_contract(tensor, x, kDLBfloat, 16, {rows, 5120});
  tensor_contract(norm, x, kDLFloat, 32, {5120});
  tensor_contract(input_weight, x, kDLFloat8_e4m3fn, 8, {columns, 5120});
  tensor_contract(input_weight_scale, x, kDLFloat, 32, {columns / 128, 40});
  for (auto tensor : {q_norm, k_norm})
    tensor_contract(tensor, x, kDLFloat, 32, {256});
  tensor_contract(positions, x, kDLInt, 64, {3, rows});
  tensor_contract(cache, x, kDLBfloat, 16, {blocks, 2, 784, 4, 256});
  tensor_contract(slots, x, kDLInt, 32, {rows});
  tensor_contract(tables, x, kDLInt, 32, {batch, tables.size(1)});
  tensor_contract(sequence_rows, x, kDLInt, 32, {batch + 1});
  tensor_contract(lengths, x, kDLInt, 32, {rows});
  tensor_contract(output_weight, x, kDLFloat8_e4m3fn, 8, {5120, 6144});
  tensor_contract(output_weight_scale, x, kDLFloat, 32, {40, 48});
  tensor_contract(quantized, x, kDLFloat8_e4m3fn, 8, {rows, 5120});
  tensor_contract(scales, x, kDLFloat, 32, {rows, 40});
  tensor_contract(packed, x, kDLBfloat, 16, {rows, columns});
  for (auto tensor : {query, gate})
    tensor_contract(tensor, x, kDLBfloat, 16, {rows, 24, 256});
  tensor_contract(partial, x, kDLFloat, 32, {rows, 24, splits, 256});
  tensor_contract(lse, x, kDLFloat, 32, {rows, 24, splits});
  tensor_contract(gated, x, kDLFloat8_e4m3fn, 8, {rows, 6144});
  tensor_contract(gated_scale, x, kDLFloat, 32, {rows, 48});
  if (rows <= 8)
    tensor_contract(projection_partials, x, kDLFloat, 32, {8, rows, 5120});
  else
    tensor_contract(projection_partials, x, kDLFloat, 32, {1});
  auto inputs = {x,
                 residual,
                 norm,
                 input_weight,
                 input_weight_scale,
                 q_norm,
                 k_norm,
                 positions,
                 slots,
                 tables,
                 sequence_rows,
                 lengths,
                 output_weight,
                 output_weight_scale};
  auto outputs = {
      output,      summed, quantized,          scales, packed, query, gate, partial, lse, gated,
      gated_scale, cache,  projection_partials};
  int left = 0;
  for (auto a : outputs) {
    for (auto b : inputs)
      disjoint(a, b);
    int right = 0;
    for (auto b : outputs)
      if (right++ > left)
        disjoint(a, b);
    ++left;
  }
  tvm::ffi::CUDADeviceGuard guard(x.device().device_id);
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, x.device().device_id));
  residual_rms_quant_launch(static_cast<const __nv_bfloat16 *>(x.data_ptr()),
                            static_cast<const __nv_bfloat16 *>(residual.data_ptr()),
                            static_cast<const float *>(norm.data_ptr()),
                            static_cast<__nv_bfloat16 *>(summed.data_ptr()),
                            static_cast<__nv_fp8_e4m3 *>(quantized.data_ptr()),
                            static_cast<float *>(scales.data_ptr()), rows, stream);
  if (variant != 1 && (rows == 1 || rows == 4 || rows == 5)) {
    if (rows == 1)
      fa_small_input_launch<1>(quantized, scales, input_weight, input_weight_scale, packed, columns,
                               stream);
    else if (rows == 4)
      fa_small_input_launch<4>(quantized, scales, input_weight, input_weight_scale, packed, columns,
                               stream);
    else
      fa_small_input_launch<5>(quantized, scales, input_weight, input_weight_scale, packed, columns,
                               stream);
  } else
    groupwise(static_cast<const __nv_fp8_e4m3 *>(quantized.data_ptr()),
              static_cast<const __nv_fp8_e4m3 *>(input_weight.data_ptr()),
              static_cast<const float *>(scales.data_ptr()),
              static_cast<const float *>(input_weight_scale.data_ptr()),
              static_cast<__nv_bfloat16 *>(packed.data_ptr()), rows, columns, 5120, stream);
  fa_heads_kernel<<<dim3(rows, variant == 1   ? 4
                               : variant == 2 ? 24
                                              : 28),
                    32, 0, stream>>>(
      static_cast<const __nv_bfloat16 *>(packed.data_ptr()),
      static_cast<const float *>(q_norm.data_ptr()), static_cast<const float *>(k_norm.data_ptr()),
      static_cast<const int64_t *>(positions.data_ptr()),
      static_cast<const int32_t *>(slots.data_ptr()),
      static_cast<__nv_bfloat16 *>(cache.data_ptr()),
      static_cast<__nv_bfloat16 *>(query.data_ptr()), static_cast<__nv_bfloat16 *>(gate.data_ptr()),
      rows, columns, variant, int64_t(blocks) * 784);
  check_cuda(cudaGetLastError());
  if (variant == 1)
    return;
  if (rows > 32 && splits == 1) {
    static std::once_flag sm100_configured;
    std::call_once(sm100_configured, [] {
      check_cuda(cudaFuncSetAttribute(
          fa_prefill_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, FAPrefillSharedBytes));
    });
    const auto &maps = fa_prefill_tensor_maps(cache);
    int tiles = (max_queries * 6 + 127) / 128;
    fa_prefill_kernel<<<dim3(batch * tiles, 4), 128, FAPrefillSharedBytes, stream>>>(
        static_cast<const __nv_bfloat16 *>(query.data_ptr()),
        static_cast<const __nv_bfloat16 *>(cache.data_ptr()),
        static_cast<const int32_t *>(tables.data_ptr()),
        static_cast<const int32_t *>(lengths.data_ptr()),
        static_cast<const int32_t *>(sequence_rows.data_ptr()),
        static_cast<float *>(partial.data_ptr()), static_cast<float *>(lse.data_ptr()),
        tables.size(1), tiles, max_queries, rows, blocks, first_token, maps.key, maps.value);
    check_cuda(cudaGetLastError());
  } else {
    auto attend = [&](auto query_tile) {
      constexpr int QueryTile = decltype(query_tile)::value;
      constexpr int shared_bytes =
          (QueryTile * 256 + 2 * 64 * 256 + QueryTile * 64) * 2 + 8 * QueryTile * 4;
      static std::once_flag configured;
      std::call_once(configured, [] {
        check_cuda(cudaFuncSetAttribute(fa_partial_kernel<true, true, int32_t, 1, QueryTile>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
      });
      int tiles = (max_queries * 6 + QueryTile - 1) / QueryTile;
      fa_partial_kernel<true, true, int32_t, 1, QueryTile>
          <<<dim3(batch * tiles, 4, splits), 128, shared_bytes, stream>>>(
              static_cast<const __nv_bfloat16 *>(query.data_ptr()),
              static_cast<const __nv_bfloat16 *>(cache.data_ptr()), tables.data_ptr(),
              lengths.data_ptr(), sequence_rows.data_ptr(),
              static_cast<float *>(partial.data_ptr()), static_cast<float *>(lse.data_ptr()), 24, 4,
              tables.size(1), splits, first_token, false, false, false, tiles, max_queries, rows,
              blocks);
      check_cuda(cudaGetLastError());
    };
    if (max_queries * 6 <= 16)
      attend(std::integral_constant<int, 16>{});
    else
      attend(std::integral_constant<int, 32>{});
  }
  auto merge = [&](auto warps) {
    constexpr int Warps = decltype(warps)::value;
    fa_merge_quant_kernel<Warps><<<rows * 48, Warps * 32, 0, stream>>>(
        static_cast<const float *>(partial.data_ptr()), static_cast<const float *>(lse.data_ptr()),
        static_cast<const __nv_bfloat16 *>(gate.data_ptr()),
        static_cast<__nv_fp8_e4m3 *>(gated.data_ptr()),
        static_cast<float *>(gated_scale.data_ptr()), splits);
  };
  if (splits == 1)
    merge(std::integral_constant<int, 1>{});
  else
    merge(std::integral_constant<int, 8>{});
  check_cuda(cudaGetLastError());
  if (rows <= 8) {
    decode_projection_launch(gated, output_weight, gated_scale, output_weight_scale,
                             projection_partials, rows, 8, stream);
    fa_output_merge_kernel<<<(rows * 5120 + 255) / 256, 256, 0, stream>>>(
        static_cast<const float *>(projection_partials.data_ptr()),
        static_cast<__nv_bfloat16 *>(output.data_ptr()), rows * 5120);
    check_cuda(cudaGetLastError());
  } else
    groupwise(static_cast<const __nv_fp8_e4m3 *>(gated.data_ptr()),
              static_cast<const __nv_fp8_e4m3 *>(output_weight.data_ptr()),
              static_cast<const float *>(gated_scale.data_ptr()),
              static_cast<const float *>(output_weight_scale.data_ptr()),
              static_cast<__nv_bfloat16 *>(output.data_ptr()), rows, 5120, 6144, stream);
}
