#include "common.cuh"
#include "fa_attention.cuh"
#include "fa_core.cuh"
#include "fa_decode_cache.cuh"
#include "fa_input.cuh"
#include "fa_kv_split.cuh"
#include "fa_mx.cuh"
#include "fa_small_input.cuh"
#include "groupwise.cuh"
#include "normalization.cuh"
#include "projection_output.cuh"
#include "projection_prefill.cuh"

void fa(tvm::ffi::Tensor x, tvm::ffi::Tensor residual, tvm::ffi::Tensor norm,
        tvm::ffi::Tensor input_weight, tvm::ffi::Tensor input_weight_scale, tvm::ffi::Tensor q_norm,
        tvm::ffi::Tensor k_norm, tvm::ffi::Tensor positions, tvm::ffi::Tensor cache,
        tvm::ffi::Tensor encoded_cache, tvm::ffi::Tensor cache_scales, tvm::ffi::Tensor slots,
        tvm::ffi::Tensor tables, tvm::ffi::Tensor sequence_rows, tvm::ffi::Tensor lengths,
        tvm::ffi::Tensor output_weight, tvm::ffi::Tensor output_weight_scale,
        tvm::ffi::Tensor output, tvm::ffi::Tensor summed, tvm::ffi::Tensor quantized,
        tvm::ffi::Tensor scales, tvm::ffi::Tensor packed, tvm::ffi::Tensor query,
        tvm::ffi::Tensor gate, tvm::ffi::Tensor partial, tvm::ffi::Tensor lse,
        tvm::ffi::Tensor gated, tvm::ffi::Tensor gated_scale, tvm::ffi::Tensor projection_partials,
        int64_t variant, int64_t max_queries, int64_t first_token, int64_t max_context,
        int64_t min_context, tvm::ffi::Optional<fa_mxio_native::MXContext> mx_context,
        tvm::ffi::Optional<fa_mxio_native::MXWeights> mx_weights) {
  TVM_FFI_ICHECK(x.ndim() == 2 && x.size(1) == 5120 && x.size(0) > 0 && x.size(0) <= 32768);
  int rows = x.size(0), columns = variant == 1 ? 2048 : variant == 2 ? 12288 : 14336;
  TVM_FFI_ICHECK(first_token == 0 || first_token == 1);
  TVM_FFI_ICHECK(variant >= 0 && variant <= 2 && max_queries > 0 && max_queries <= rows);
  int attention_rows = variant == 1 && rows > 32 ? 0 : rows;
  TVM_FFI_ICHECK(tables.ndim() == 2 && tables.size(0) > 0 && tables.size(1) > 0);
  TVM_FFI_ICHECK(tables.size(0) <= rows && tables.size(1) <= 335);
  TVM_FFI_ICHECK(max_context >= max_queries && max_context <= tables.size(1) * 784);
  TVM_FFI_ICHECK(min_context >= 0 && min_context <= max_context);
  TVM_FFI_ICHECK(cache.ndim() == 5 && cache.size(0) > 0 && cache.size(0) <= INT32_MAX / 784);
  TVM_FFI_ICHECK(lse.ndim() == 3 && (lse.size(2) == 1 || lse.size(2) == 4 || lse.size(2) == 16 ||
                                     lse.size(2) == 64 || lse.size(2) == 128));
  int splits = lse.size(2), batch = tables.size(0), blocks = cache.size(0);
  for (auto tensor : {x, residual, summed})
    tensor_contract(tensor, x, kDLBfloat, 16, {rows, 5120});
  tensor_contract(output, x, kDLBfloat, 16, {attention_rows, 5120});
  tensor_contract(norm, x, kDLFloat, 32, {5120});
  tensor_contract(input_weight, x, kDLFloat8_e4m3fn, 8, {columns, 5120});
  tensor_contract(input_weight_scale, x, kDLFloat, 32, {columns / 128, 40});
  for (auto tensor : {q_norm, k_norm})
    tensor_contract(tensor, x, kDLFloat, 32, {256});
  tensor_contract(positions, x, kDLInt, 64, {3, rows});
  tensor_contract(cache, x, kDLBfloat, 16, {blocks, 2, 784, 4, 256});
  tensor_contract(encoded_cache, x, kDLFloat8_e4m3fn, 8, {blocks, 2, 784, 4, 256});
  tensor_contract(cache_scales, x, kDLFloat, 32, {blocks, 2, 4, 784});
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
    tensor_contract(tensor, x, kDLBfloat, 16, {attention_rows, 24, 256});
  tensor_contract(partial, x, kDLFloat, 32, {attention_rows, 24, splits, 256});
  tensor_contract(lse, x, kDLFloat, 32, {attention_rows, 24, splits});
  tensor_contract(gated, x, kDLFloat8_e4m3fn, 8, {attention_rows, 6144});
  tensor_contract(gated_scale, x, kDLFloat, 32, {attention_rows, 48});
  if (rows <= 32)
    tensor_contract(projection_partials, x, kDLFloat, 32, {rows <= 8 ? 8 : 3, rows, 5120});
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
  auto outputs = {output,      summed, quantized,     scales,       packed,
                  query,       gate,   partial,       lse,          gated,
                  gated_scale, cache,  encoded_cache, cache_scales, projection_partials};
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
  TVM_FFI_ICHECK(mx_context.has_value() == mx_weights.has_value());
  TVM_FFI_ICHECK(!mx_context.has_value() || (variant == 0 && rows > 128))
      << "FA MX requires full attention with more than 128 rows";
  fa_mxio_native::Call mx(mx_context, mx_weights, norm, input_weight, input_weight_scale,
                          output_weight, output_weight_scale,
                          {x,
                           residual,
                           norm,
                           input_weight,
                           input_weight_scale,
                           q_norm,
                           k_norm,
                           positions,
                           cache,
                           slots,
                           tables,
                           sequence_rows,
                           lengths,
                           output_weight,
                           output_weight_scale,
                           output,
                           summed,
                           quantized,
                           scales,
                           packed,
                           query,
                           gate,
                           partial,
                           lse,
                           gated,
                           gated_scale,
                           projection_partials,
                           encoded_cache,
                           cache_scales},
                          rows, stream);
  bool fused_input = variant != 1 && rows == 1;
  if (mx.active())
    mx.normalize(static_cast<const __nv_bfloat16 *>(x.data_ptr()), residual, summed, stream);
  else if (!fused_input)
    residual_rms_quant_launch(static_cast<const __nv_bfloat16 *>(x.data_ptr()),
                              static_cast<const __nv_bfloat16 *>(residual.data_ptr()),
                              static_cast<const float *>(norm.data_ptr()),
                              static_cast<__nv_bfloat16 *>(summed.data_ptr()),
                              static_cast<__nv_fp8_e4m3 *>(quantized.data_ptr()),
                              static_cast<float *>(scales.data_ptr()), rows, stream);
  bool split_kv = variant == 1 && (rows == 4 || rows == 16) &&
                  int64_t(partial.numel()) >= int64_t(8) * rows * 5120;
  if (mx.active())
    mx.project(packed, stream);
  else if (split_kv) {
    if (rows == 4)
      fa_kv_split_projection_launch<4>(quantized, scales, input_weight, input_weight_scale, partial,
                                       stream);
    else
      fa_kv_split_projection_launch<16>(quantized, scales, input_weight, input_weight_scale,
                                        partial, stream);
  } else if (variant != 1 && (rows == 1 || rows == 4 || rows == 5)) {
    auto project = [&](auto count) {
      constexpr int Rows = decltype(count)::value;
      if constexpr (Rows == 1)
        fa_small_input_launch<Rows, true>(quantized, scales, input_weight, input_weight_scale,
                                          packed, columns, stream,
                                          static_cast<const __nv_bfloat16 *>(x.data_ptr()),
                                          static_cast<const __nv_bfloat16 *>(residual.data_ptr()),
                                          static_cast<const float *>(norm.data_ptr()),
                                          static_cast<__nv_bfloat16 *>(summed.data_ptr()));
      else
        fa_small_input_launch<Rows>(quantized, scales, input_weight, input_weight_scale, packed,
                                    columns, stream);
    };
    if (rows == 1)
      project(std::integral_constant<int, 1>{});
    else if (rows == 4)
      project(std::integral_constant<int, 4>{});
    else
      project(std::integral_constant<int, 5>{});
  } else if (rows > 32)
    prefill_projection::launch(static_cast<const __nv_fp8_e4m3 *>(quantized.data_ptr()),
                               static_cast<const __nv_fp8_e4m3 *>(input_weight.data_ptr()),
                               static_cast<const float *>(scales.data_ptr()),
                               static_cast<const float *>(input_weight_scale.data_ptr()),
                               static_cast<__nv_bfloat16 *>(packed.data_ptr()), rows, columns, 5120,
                               stream);
  else
    groupwise(static_cast<const __nv_fp8_e4m3 *>(quantized.data_ptr()),
              static_cast<const __nv_fp8_e4m3 *>(input_weight.data_ptr()),
              static_cast<const float *>(scales.data_ptr()),
              static_cast<const float *>(input_weight_scale.data_ptr()),
              static_cast<__nv_bfloat16 *>(packed.data_ptr()), rows, columns, 5120, stream);
  if (split_kv)
    fa_kv_split_heads_kernel<<<dim3(rows, 4), 32, 0, stream>>>(
        static_cast<const float *>(partial.data_ptr()),
        static_cast<__nv_bfloat16 *>(packed.data_ptr()),
        static_cast<const float *>(k_norm.data_ptr()),
        static_cast<const int64_t *>(positions.data_ptr()),
        static_cast<const int32_t *>(slots.data_ptr()),
        static_cast<__nv_bfloat16 *>(cache.data_ptr()), rows, int64_t(blocks) * 784);
  else
    fa_heads_kernel<<<dim3(rows, variant == 1   ? 1
                                 : variant == 2 ? 6
                                                : 7),
                      dim3(32, 4), 0, stream>>>(
        static_cast<const __nv_bfloat16 *>(packed.data_ptr()),
        static_cast<const float *>(q_norm.data_ptr()),
        static_cast<const float *>(k_norm.data_ptr()),
        static_cast<const int64_t *>(positions.data_ptr()),
        static_cast<const int32_t *>(slots.data_ptr()),
        static_cast<__nv_bfloat16 *>(cache.data_ptr()),
        static_cast<__nv_bfloat16 *>(query.data_ptr()),
        static_cast<__nv_bfloat16 *>(gate.data_ptr()), rows, columns, variant,
        int64_t(blocks) * 784,
        rows <= 32 && max_queries > 1 && max_queries <= 5
            ? static_cast<__nv_fp8_e4m3 *>(gated.data_ptr())
            : nullptr,
        rows <= 32 && max_queries > 1 && max_queries <= 5
            ? static_cast<float *>(gated_scale.data_ptr())
            : nullptr);
  check_cuda(cudaGetLastError());
  if (variant != 2)
    fa_cache_encode(cache, encoded_cache, cache_scales, slots, k_norm);
  if (variant == 1)
    return;
  auto *attended = rows > 32 ? static_cast<__nv_bfloat16 *>(packed.data_ptr()) : nullptr;
  splits = fa_attention_core_launch(query, cache, tables, lengths, sequence_rows, partial, lse,
                                    max_queries, first_token, stream, attended, max_context,
                                    min_context, encoded_cache, cache_scales, gated, gated_scale);
  auto merge = [&](auto warps) {
    constexpr int Warps = decltype(warps)::value;
    constexpr int Groups = Warps == 1 ? 8 : 1;
    fa_merge_quant_kernel<Warps, Groups><<<rows * 48 / Groups, Warps * Groups * 32, 0, stream>>>(
        static_cast<const float *>(partial.data_ptr()), static_cast<const float *>(lse.data_ptr()),
        static_cast<const __nv_bfloat16 *>(gate.data_ptr()),
        static_cast<__nv_fp8_e4m3 *>(gated.data_ptr()),
        static_cast<float *>(gated_scale.data_ptr()), splits);
  };
  bool mx_output = mx.active() && splits == 1;
  if (mx_output)
    mx.quantize_output(attended, gate, stream);
  else if (splits == 1 && rows > 32)
    fa_bf16_one_split_quant_kernel<<<rows * 6, 256, 0, stream>>>(
        attended, static_cast<const __nv_bfloat16 *>(gate.data_ptr()),
        static_cast<__nv_fp8_e4m3 *>(gated.data_ptr()),
        static_cast<float *>(gated_scale.data_ptr()));
  else if (splits == 1)
    fa_one_split_quant_kernel<<<rows * 6, 256, 0, stream>>>(
        static_cast<const float *>(partial.data_ptr()), static_cast<const float *>(lse.data_ptr()),
        static_cast<const __nv_bfloat16 *>(gate.data_ptr()),
        static_cast<__nv_fp8_e4m3 *>(gated.data_ptr()),
        static_cast<float *>(gated_scale.data_ptr()));
  else if (splits > 4)
    merge(std::integral_constant<int, 8>{});
  else
    merge(std::integral_constant<int, 1>{});
  check_cuda(cudaGetLastError());
  if (rows <= 32)
    output_projection(gated, output_weight, gated_scale, output_weight_scale, projection_partials,
                      output, rows, stream);
  else if (mx_output)
    mx.project_output(output, stream);
  else
    prefill_projection::launch(static_cast<const __nv_fp8_e4m3 *>(gated.data_ptr()),
                               static_cast<const __nv_fp8_e4m3 *>(output_weight.data_ptr()),
                               static_cast<const float *>(gated_scale.data_ptr()),
                               static_cast<const float *>(output_weight_scale.data_ptr()),
                               static_cast<__nv_bfloat16 *>(output.data_ptr()), rows, 5120, 6144,
                               stream);
  mx.finish();
}
