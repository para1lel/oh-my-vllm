#include "gdn_kernels.cuh"
#include "groupwise.cuh"
#include "linear.cuh"
#include "normalization.cuh"

inline void gdn_projection(const __nv_fp8_e4m3 *input, const __nv_fp8_e4m3 *weight,
                           const float *input_scale, const float *weight_scale,
                           __nv_bfloat16 *output, int rows, int columns, int width,
                           cudaStream_t stream) {
  if (rows <= 16 && columns == 5120)
    groupwise<128, 16, 256, 128, 1, true>(weight, input, weight_scale, input_scale, output, columns,
                                          rows, width, stream);
  else if (rows <= 32)
    groupwise<128, 32, 256, 128, 1, true>(weight, input, weight_scale, input_scale, output, columns,
                                          rows, width, stream);
  else
    groupwise(input, weight, input_scale, weight_scale, output, rows, columns, width, stream);
}

void gdn(TensorView x, TensorView residual, TensorView norm, TensorView qkvz_weight,
         TensorView qkvz_scale, TensorView ba_weight, TensorView convolution_weight,
         TensorView a_log, TensorView dt_bias, TensorView head_norm, TensorView output_weight,
         TensorView output_scale, TensorView state, TensorView convolution_state, TensorView starts,
         TensorView reads, TensorView writes, TensorView embedding, TensorView tokens,
         TensorView media, TensorView media_rows, bool first_layer, bool verify, TensorView output,
         TensorView summed, TensorView gathered, TensorView normalized, TensorView input_data,
         TensorView input_scales, TensorView qkvz, TensorView ba, TensorView qkv,
         TensorView output_data, TensorView output_scales) {
  TVM_FFI_ICHECK(x.ndim() == 2 && x.size(1) == 5120 && x.size(0) >= 0 && x.size(0) <= 32768);
  TVM_FFI_ICHECK(reads.ndim() == 1 && reads.size(0) > 0 && reads.size(0) <= 32768);
  TVM_FFI_ICHECK(state.ndim() == 4 && state.size(0) > 0 && state.size(0) <= 65536);
  TVM_FFI_ICHECK(embedding.ndim() == 2 && embedding.size(0) > 0);
  TVM_FFI_ICHECK(!first_layer || embedding.size(0) == 248320);
  TVM_FFI_ICHECK(media.ndim() == 2 && media.size(0) >= 0 && media.size(0) <= x.size(0));
  TVM_FFI_ICHECK(first_layer || media.size(0) == 0);
  int rows = x.size(0), batch = reads.size(0), slots = state.size(0);
  for (auto tensor : {x, residual, output, summed, gathered, normalized})
    tensor_contract(tensor, x, kDLBfloat, 16, {rows, 5120});
  tensor_contract(norm, x, kDLFloat, 32, {5120});
  tensor_contract(qkvz_weight, x, kDLFloat8_e4m3fn, 8, {16384, 5120});
  tensor_contract(qkvz_scale, x, kDLFloat, 32, {128, 40});
  tensor_contract(ba_weight, x, kDLBfloat, 16, {96, 5120});
  tensor_contract(convolution_weight, x, kDLBfloat, 16, {10240, 4});
  tensor_contract(a_log, x, kDLFloat, 32, {48});
  tensor_contract(dt_bias, x, kDLFloat, 32, {48});
  tensor_contract(head_norm, x, kDLFloat, 32, {128});
  tensor_contract(output_weight, x, kDLFloat8_e4m3fn, 8, {5120, 6144});
  tensor_contract(output_scale, x, kDLFloat, 32, {40, 48});
  tensor_contract(state, x, kDLBfloat, 16, {slots, 48, 128, 128});
  tensor_contract(convolution_state, x, kDLBfloat, 16, {slots, 10240, 3});
  tensor_contract(starts, x, kDLInt, 32, {batch + 1});
  tensor_contract(reads, x, kDLInt, 32, {batch});
  tensor_contract(writes, x, kDLInt, 32, {rows});
  tensor_contract(embedding, x, kDLBfloat, 16, {embedding.size(0), 5120});
  tensor_contract(tokens, x, kDLInt, 64, {rows});
  tensor_contract(media, x, kDLBfloat, 16, {media.size(0), 5120});
  tensor_contract(media_rows, x, kDLInt, 32, {media.size(0)});
  tensor_contract(input_data, x, kDLFloat8_e4m3fn, 8, {rows, 5120});
  tensor_contract(input_scales, x, kDLFloat, 32, {rows, 40});
  tensor_contract(qkvz, x, kDLBfloat, 16, {rows, 16384});
  tensor_contract(ba, x, kDLBfloat, 16, {rows, 96});
  tensor_contract(qkv, x, kDLBfloat, 16, {rows, 10240});
  tensor_contract(output_data, x, kDLFloat8_e4m3fn, 8, {rows, 6144});
  tensor_contract(output_scales, x, kDLFloat, 32, {rows, 48});
  auto tensors = {
      x,      residual, norm,        qkvz_weight,   qkvz_scale,   ba_weight,    convolution_weight,
      a_log,  dt_bias,  head_norm,   output_weight, output_scale, state,        convolution_state,
      starts, reads,    writes,      embedding,     tokens,       media,        media_rows,
      output, summed,   gathered,    normalized,    input_data,   input_scales, qkvz,
      ba,     qkv,      output_data, output_scales};
  int left = 0;
  for (auto a : tensors) {
    int right = 0;
    for (auto b : tensors)
      if (right++ > left)
        disjoint(a, b);
    ++left;
  }
  if (!rows)
    return;
  tvm::ffi::CUDADeviceGuard guard(x.device().device_id);
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, x.device().device_id));
  const auto *input = static_cast<const __nv_bfloat16 *>(x.data_ptr());
  if (first_layer) {
    gdn_gather_kernel<<<rows, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16 *>(embedding.data_ptr()),
        static_cast<const int64_t *>(tokens.data_ptr()),
        static_cast<const __nv_bfloat16 *>(media.data_ptr()),
        static_cast<const int *>(media_rows.data_ptr()),
        static_cast<__nv_bfloat16 *>(gathered.data_ptr()), rows, embedding.size(0), media.size(0));
    check_cuda(cudaGetLastError());
    input = static_cast<const __nv_bfloat16 *>(gathered.data_ptr());
  }
  residual_rms_quant_launch(input, static_cast<const __nv_bfloat16 *>(residual.data_ptr()),
                            static_cast<const float *>(norm.data_ptr()),
                            static_cast<__nv_bfloat16 *>(summed.data_ptr()),
                            static_cast<__nv_fp8_e4m3 *>(input_data.data_ptr()),
                            static_cast<float *>(input_scales.data_ptr()), rows, stream,
                            static_cast<__nv_bfloat16 *>(normalized.data_ptr()));
  gdn_projection(static_cast<const __nv_fp8_e4m3 *>(input_data.data_ptr()),
                 static_cast<const __nv_fp8_e4m3 *>(qkvz_weight.data_ptr()),
                 static_cast<const float *>(input_scales.data_ptr()),
                 static_cast<const float *>(qkvz_scale.data_ptr()),
                 static_cast<__nv_bfloat16 *>(qkvz.data_ptr()), rows, 16384, 5120, stream);
  if (rows > 32)
    bf16_linear(static_cast<const __nv_bfloat16 *>(normalized.data_ptr()),
                static_cast<const __nv_bfloat16 *>(ba_weight.data_ptr()), ba.data_ptr(), rows, 96,
                5120, stream);
  gdn_convolution_kernel<<<dim3(rows, 20), 128, 0, stream>>>(
      static_cast<const __nv_bfloat16 *>(qkvz.data_ptr()),
      static_cast<const __nv_bfloat16 *>(convolution_weight.data_ptr()),
      static_cast<__nv_bfloat16 *>(convolution_state.data_ptr()),
      static_cast<const int *>(starts.data_ptr()), static_cast<const int *>(reads.data_ptr()),
      static_cast<const int *>(writes.data_ptr()), static_cast<__nv_bfloat16 *>(qkv.data_ptr()),
      rows, batch, slots, verify);
  check_cuda(cudaGetLastError());
  auto scan = [&](auto fused) {
    gdn_scan_quant_kernel<decltype(fused)::value><<<dim3(batch, 48), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16 *>(qkv.data_ptr()),
        static_cast<const __nv_bfloat16 *>(qkvz.data_ptr()),
        static_cast<__nv_bfloat16 *>(ba.data_ptr()),
        static_cast<const __nv_bfloat16 *>(normalized.data_ptr()),
        static_cast<const __nv_bfloat16 *>(ba_weight.data_ptr()),
        static_cast<const float *>(a_log.data_ptr()),
        static_cast<const float *>(dt_bias.data_ptr()),
        static_cast<const float *>(head_norm.data_ptr()),
        static_cast<__nv_bfloat16 *>(state.data_ptr()), static_cast<const int *>(starts.data_ptr()),
        static_cast<const int *>(reads.data_ptr()), static_cast<const int *>(writes.data_ptr()),
        static_cast<__nv_fp8_e4m3 *>(output_data.data_ptr()),
        static_cast<float *>(output_scales.data_ptr()), rows, slots, verify);
  };
  if (rows <= 32)
    scan(std::true_type{});
  else
    scan(std::false_type{});
  check_cuda(cudaGetLastError());
  gdn_projection(static_cast<const __nv_fp8_e4m3 *>(output_data.data_ptr()),
                 static_cast<const __nv_fp8_e4m3 *>(output_weight.data_ptr()),
                 static_cast<const float *>(output_scales.data_ptr()),
                 static_cast<const float *>(output_scale.data_ptr()),
                 static_cast<__nv_bfloat16 *>(output.data_ptr()), rows, 5120, 6144, stream);
}
