#include "gdn_kernels.cuh"
#include "gdn_value_pair.cuh"
#include "gdn_input.cuh"
#include "projection_decode_16dp.cuh"
#include "groupwise.cuh"
#include "projection_prefill.cuh"
#include "linear.cuh"
#include "normalization.cuh"

__global__ void gdn_output_projection_merge_kernel(const float *partials, __nv_bfloat16 *output,
                                                   int count) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    float value = 0;
    for (int split = 0; split < 8; ++split)
      value += partials[split * count + index];
    output[index] = __float2bfloat16_rn(value);
  }
}

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
    prefill_projection::launch(input, weight, input_scale, weight_scale, output, rows, columns,
                               width, stream);
}

void gdn(TensorView x, TensorView residual, TensorView norm, TensorView qkvz_weight,
         TensorView qkvz_scale, TensorView ba_weight, TensorView convolution_weight,
         TensorView a_log, TensorView dt_bias, TensorView head_norm, TensorView output_weight,
         TensorView output_scale, TensorView state, TensorView convolution_state, TensorView starts,
         TensorView reads, TensorView writes, TensorView embedding, TensorView tokens,
         TensorView media, TensorView media_rows, bool first_layer, bool verify, TensorView output,
         TensorView summed, TensorView gathered, TensorView normalized, TensorView input_data,
         TensorView input_scales, TensorView qkvz, TensorView ba, TensorView qkv,
         TensorView output_data, TensorView output_scales, TensorView recurrent,
         TensorView projection_partials, TensorView gates, int max_queries) {
  TVM_FFI_ICHECK(x.ndim() == 2 && x.size(1) == 5120 && x.size(0) >= 0 && x.size(0) <= 32768);
  TVM_FFI_ICHECK(reads.ndim() == 1 && reads.size(0) > 0 && reads.size(0) <= 32768);
  TVM_FFI_ICHECK(state.ndim() == 4 && state.size(0) > 0 && state.size(0) <= 65536);
  TVM_FFI_ICHECK(embedding.ndim() == 2 && embedding.size(0) > 0);
  TVM_FFI_ICHECK(!first_layer || embedding.size(0) == 248320);
  TVM_FFI_ICHECK(media.ndim() == 2 && media.size(0) >= 0 && media.size(0) <= x.size(0));
  TVM_FFI_ICHECK(first_layer || media.size(0) == 0);
  int rows = x.size(0), batch = reads.size(0), slots = state.size(0);
  TVM_FFI_ICHECK(max_queries >= 0 && max_queries <= rows && (rows == 0 || max_queries > 0));
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
  bool split_values = (!verify && rows > 32) ||
                      (verify && ((batch == 1 && rows == 5) || (batch == 4 && rows == 20)));
  tensor_contract(recurrent, x, kDLBfloat, 16, {split_values ? rows : 0, 6144});
  bool precomputed_gates = !verify && rows > 32;
  tensor_contract(gates, x, kDLFloat, 32, {precomputed_gates ? rows : 0, 96});
  if (rows <= 8)
    tensor_contract(projection_partials, x, kDLFloat, 32, {8, rows, 5120});
  else
    tensor_contract(projection_partials, x, kDLFloat, 32, {1});
  auto tensors = {x,
                  residual,
                  norm,
                  qkvz_weight,
                  qkvz_scale,
                  ba_weight,
                  convolution_weight,
                  a_log,
                  dt_bias,
                  head_norm,
                  output_weight,
                  output_scale,
                  state,
                  convolution_state,
                  starts,
                  reads,
                  writes,
                  embedding,
                  tokens,
                  media,
                  media_rows,
                  output,
                  summed,
                  gathered,
                  normalized,
                  input_data,
                  input_scales,
                  qkvz,
                  ba,
                  qkv,
                  output_data,
                  output_scales,
                  recurrent,
                  projection_partials,
                  gates};
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
  GDNInputConvolution convolution{static_cast<const __nv_bfloat16 *>(convolution_weight.data_ptr()),
                                  static_cast<__nv_bfloat16 *>(convolution_state.data_ptr()),
                                  static_cast<const int *>(starts.data_ptr()),
                                  static_cast<const int *>(reads.data_ptr()),
                                  static_cast<const int *>(writes.data_ptr()),
                                  static_cast<__nv_bfloat16 *>(qkv.data_ptr()),
                                  batch,
                                  slots};
  auto project_input = [&](auto count) {
    constexpr int Rows = decltype(count)::value;
    if constexpr (Rows == 1) {
      if (verify) {
        gdn_input_launch<Rows, true, true, true>(input_data, input_scales, qkvz_weight, qkvz_scale,
                                                 qkvz, convolution, stream);
        return;
      }
    }
    gdn_input_launch<Rows, false, true, true>(input_data, input_scales, qkvz_weight, qkvz_scale,
                                              qkvz, convolution, stream);
  };
  if (rows == 1)
    project_input(std::integral_constant<int, 1>{});
  else if (rows == 4)
    project_input(std::integral_constant<int, 4>{});
  else if (rows == 5)
    project_input(std::integral_constant<int, 5>{});
  else
    gdn_projection(static_cast<const __nv_fp8_e4m3 *>(input_data.data_ptr()),
                   static_cast<const __nv_fp8_e4m3 *>(qkvz_weight.data_ptr()),
                   static_cast<const float *>(input_scales.data_ptr()),
                   static_cast<const float *>(qkvz_scale.data_ptr()),
                   static_cast<__nv_bfloat16 *>(qkvz.data_ptr()), rows, 16384, 5120, stream);
  if (rows > 32)
    bf16_linear(static_cast<const __nv_bfloat16 *>(normalized.data_ptr()),
                static_cast<const __nv_bfloat16 *>(ba_weight.data_ptr()), ba.data_ptr(), rows, 96,
                5120, stream);
  if (precomputed_gates) {
    gdn_gate_kernel<<<(rows * 48 + 255) / 256, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16 *>(ba.data_ptr()),
        static_cast<const float *>(a_log.data_ptr()),
        static_cast<const float *>(dt_bias.data_ptr()), static_cast<float *>(gates.data_ptr()),
        rows);
    check_cuda(cudaGetLastError());
  }
  auto prepare = [&](auto project_ba) {
    constexpr bool ProjectBA = decltype(project_ba)::value;
    launch_kernel(
        gdn_convolution_kernel<ProjectBA>, dim3(rows, ProjectBA ? 68 : 20),
        dim3(ProjectBA ? 256 : 128), 0, stream, static_cast<const __nv_bfloat16 *>(qkvz.data_ptr()),
        static_cast<const __nv_bfloat16 *>(convolution_weight.data_ptr()),
        static_cast<__nv_bfloat16 *>(convolution_state.data_ptr()),
        static_cast<const int *>(starts.data_ptr()), static_cast<const int *>(reads.data_ptr()),
        static_cast<const int *>(writes.data_ptr()), static_cast<__nv_bfloat16 *>(qkv.data_ptr()),
        rows, batch, slots, verify, static_cast<const __nv_bfloat16 *>(normalized.data_ptr()),
        static_cast<const __nv_bfloat16 *>(ba_weight.data_ptr()),
        static_cast<__nv_bfloat16 *>(ba.data_ptr()));
  };
  if (rows > 1 && rows <= 32)
    prepare(std::true_type{});
  else if (!(rows == 1 && verify))
    prepare(std::false_type{});
  check_cuda(cudaGetLastError());
  auto scan = [&](auto fused, auto value_tile, auto thread_count, auto prepared_gates,
                  auto staged) {
    constexpr int ValueTile = decltype(value_tile)::value;
    constexpr int Threads = decltype(thread_count)::value;
    constexpr int SharedBytes = decltype(staged)::value ? GDNPrefetchSlotBytes<ValueTile> * 2 : 0;
    auto kernel = gdn_scan_quant_kernel<decltype(fused)::value, ValueTile, Threads,
                                        decltype(prepared_gates)::value, decltype(staged)::value,
                                        decltype(staged)::value>;
    if constexpr (!decltype(staged)::value) {
      if (verify)
        kernel = gdn_scan_quant_kernel<decltype(fused)::value, ValueTile, Threads,
                                       decltype(prepared_gates)::value, false, false, true>;
    }
    launch_kernel(
        kernel, dim3(batch, 48, 128 / ValueTile), dim3(Threads), SharedBytes, stream,
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
        static_cast<float *>(output_scales.data_ptr()), rows, slots, verify,
        static_cast<__nv_bfloat16 *>(recurrent.data_ptr()),
        static_cast<const float *>(gates.data_ptr()));
  };
  if (precomputed_gates) {
    auto paired = [&](auto value_tile, auto kernel) {
      constexpr int ValueTile = decltype(value_tile)::value;
      launch_kernel(
          kernel, dim3(batch, 48, 128 / ValueTile), dim3(128), 2 * GDNPrefetchSlotBytes<ValueTile>,
          stream, static_cast<const __nv_bfloat16 *>(qkv.data_ptr()),
          static_cast<__nv_bfloat16 *>(state.data_ptr()),
          static_cast<const int *>(starts.data_ptr()), static_cast<const int *>(reads.data_ptr()),
          static_cast<const int *>(writes.data_ptr()),
          static_cast<__nv_bfloat16 *>(recurrent.data_ptr()),
          static_cast<const float *>(gates.data_ptr()), rows, slots);
    };
    if (gdn_prefill_two_pairs(rows, max_queries, x.device().device_id))
      paired(std::integral_constant<int, 32>{}, gdn_value_two_pair_kernel<32, 128>);
    else
      paired(std::integral_constant<int, 16>{}, gdn_value_pair_kernel<16, 128>);
  } else if (verify && batch == 4 && rows == 20)
    scan(std::false_type{}, std::integral_constant<int, 32>{}, std::integral_constant<int, 128>{},
         std::false_type{}, std::false_type{});
  else if (split_values)
    scan(std::false_type{}, std::integral_constant<int, 16>{}, std::integral_constant<int, 256>{},
         std::false_type{}, std::false_type{});
  else {
    auto run = [&](auto threads) {
      if (rows == 1)
        scan(std::true_type{}, std::integral_constant<int, 128>{}, threads, std::false_type{},
             std::false_type{});
      else
        scan(std::false_type{}, std::integral_constant<int, 128>{}, threads, std::false_type{},
             std::false_type{});
    };
    if (verify && rows > 1)
      run(std::integral_constant<int, 512>{});
    else
      run(std::integral_constant<int, 256>{});
  }
  check_cuda(cudaGetLastError());
  if (split_values) {
    gdn_output_quant_kernel<<<rows * 6, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16 *>(recurrent.data_ptr()),
        static_cast<const __nv_bfloat16 *>(qkvz.data_ptr()),
        static_cast<const float *>(head_norm.data_ptr()),
        static_cast<__nv_fp8_e4m3 *>(output_data.data_ptr()),
        static_cast<float *>(output_scales.data_ptr()));
    check_cuda(cudaGetLastError());
  }
  if (rows <= 8) {
    down16dp::launch(output_data, output_weight, output_scales, output_scale, projection_partials,
                     rows, 8, stream);
    gdn_output_projection_merge_kernel<<<(rows * 5120 + 255) / 256, 256, 0, stream>>>(
        static_cast<const float *>(projection_partials.data_ptr()),
        static_cast<__nv_bfloat16 *>(output.data_ptr()), rows * 5120);
    check_cuda(cudaGetLastError());
  } else
    gdn_projection(static_cast<const __nv_fp8_e4m3 *>(output_data.data_ptr()),
                   static_cast<const __nv_fp8_e4m3 *>(output_weight.data_ptr()),
                   static_cast<const float *>(output_scales.data_ptr()),
                   static_cast<const float *>(output_scale.data_ptr()),
                   static_cast<__nv_bfloat16 *>(output.data_ptr()), rows, 5120, 6144, stream);
}
