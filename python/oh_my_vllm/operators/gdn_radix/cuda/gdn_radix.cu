#define CUTLASS_ENABLE_GDC_FOR_SM100
#include "../../gdn/cuda/input/gdn_convolution_prefill.cuh"
#include "../../gdn/cuda/input/gdn_input.cuh"
#include "../../gdn/cuda/recurrent/gdn_kernels.cuh"
#include "../../gdn/cuda/recurrent/gdn_recurrent_core.cuh"
#include "../../gdn/cuda/recurrent/gdn_value_pair.cuh"
#include "launch.cuh"
#include "../../primitives/cuda/groupwise.cuh"
#include "../../primitives/cuda/linear.cuh"
#include "../../primitives/cuda/normalization.cuh"
#include "../../primitives/cuda/projection_output.cuh"
#include "../../primitives/cuda/projection_prefill.cuh"

inline void radix_projection(const __nv_fp8_e4m3 *input, const __nv_fp8_e4m3 *weight,
                             const float *input_scale, const float *weight_scale,
                             __nv_bfloat16 *output, int rows, int columns, int width,
                             cudaStream_t stream) {
  if (rows <= 32)
    groupwise<128, 32, 256, 128, 1, true>(weight, input, weight_scale, input_scale, output, columns,
                                          rows, width, stream, true);
  else
    prefill_projection::launch(input, weight, input_scale, weight_scale, output, rows, columns,
                               width, stream);
}

struct RadixInvocation {
  const tvm::ffi::Tensor &x;
  const tvm::ffi::Tensor &residual;
  const tvm::ffi::Tensor &norm;
  const tvm::ffi::Tensor &qkvz_weight;
  const tvm::ffi::Tensor &qkvz_scale;
  const tvm::ffi::Tensor &ba_weight;
  const tvm::ffi::Tensor &convolution_weight;
  const tvm::ffi::Tensor &a_log;
  const tvm::ffi::Tensor &dt_bias;
  const tvm::ffi::Tensor &head_norm;
  const tvm::ffi::Tensor &output_weight;
  const tvm::ffi::Tensor &output_scale;
  const tvm::ffi::Tensor &state;
  const tvm::ffi::Tensor &convolution_state;
  const tvm::ffi::Tensor &starts;
  const tvm::ffi::Tensor &reads;
  const tvm::ffi::Tensor &writes;
  const tvm::ffi::Tensor &embedding;
  const tvm::ffi::Tensor &tokens;
  const tvm::ffi::Tensor &media;
  const tvm::ffi::Tensor &media_rows;
  bool first_layer;
  bool verify;
  const tvm::ffi::Tensor &output;
  const tvm::ffi::Tensor &summed;
  const tvm::ffi::Tensor &gathered;
  const tvm::ffi::Tensor &normalized;
  const tvm::ffi::Tensor &input_data;
  const tvm::ffi::Tensor &input_scales;
  const tvm::ffi::Tensor &qkvz;
  const tvm::ffi::Tensor &ba;
  const tvm::ffi::Tensor &qkv;
  const tvm::ffi::Tensor &output_data;
  const tvm::ffi::Tensor &output_scales;
  const tvm::ffi::Tensor &recurrent;
  const tvm::ffi::Tensor &projection_partials;
  const tvm::ffi::Tensor &gates;
  const tvm::ffi::Tensor &chunk_metadata;
  const tvm::ffi::Tensor &chunk_offsets;
  const tvm::ffi::Tensor &chunk_transforms;
  const tvm::ffi::Tensor &chunk_prefixes;
  const tvm::ffi::Tensor &chunk_snapshots;
  const tvm::ffi::Tensor &chunk_coefficients;
  const tvm::ffi::Tensor &slot_owners;
  int max_queries;

  int rows = 0, batch = 0, slots = 0, chunks = 0;
  bool split_values = false, precomputed_gates = false, chunked = false;

  void validate() {
    TVM_FFI_ICHECK(x.ndim() == 2 && x.size(1) == 5120 && x.size(0) >= 0 && x.size(0) <= 32768);
    TVM_FFI_ICHECK(reads.ndim() == 1 && reads.size(0) > 0 && reads.size(0) <= 32768);
    TVM_FFI_ICHECK(state.ndim() == 4 && state.size(0) > 0 && state.size(0) <= 65536);
    TVM_FFI_ICHECK(embedding.ndim() == 2 && embedding.size(0) > 0);
    TVM_FFI_ICHECK(!first_layer || embedding.size(0) == 248320);
    TVM_FFI_ICHECK(media.ndim() == 2 && media.size(0) >= 0 && media.size(0) <= x.size(0));
    TVM_FFI_ICHECK(first_layer || media.size(0) == 0);
    rows = x.size(0);
    batch = reads.size(0);
    slots = state.size(0);
    TVM_FFI_ICHECK(max_queries >= 0 && max_queries <= rows && (rows == 0 || max_queries > 0));
    for (auto tensor : {x, output, summed, gathered, normalized})
      tensor_contract(tensor, x, kDLBfloat, 16, {rows, 5120});
    residual_contract(residual, x, rows);
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
    tensor_contract(state, x, kDLFloat, 32, {slots, 48, 128, 128});
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
    split_values = rows > 32 || verify;
    tensor_contract(recurrent, x, kDLBfloat, 16, {split_values ? rows : 0, 6144});
    precomputed_gates = !verify && rows > 32;
    tensor_contract(gates, x, kDLFloat, 32, {precomputed_gates ? rows : 0, 96});
    chunked = gdn_radix_wy::selected(rows, batch, max_queries, verify);
    TVM_FFI_ICHECK(chunk_metadata.ndim() == 2 && chunk_metadata.size(0) >= 0 &&
                   chunk_metadata.size(0) <= rows);
    tensor_contract(slot_owners, x, kDLInt, 32, {slots});
    chunks = chunk_metadata.size(0);
    TVM_FFI_ICHECK(!chunked || chunks > 0);
    gdn_radix_wy::workspace_contract(x, chunk_metadata, chunk_offsets, chunk_transforms,
                                     chunk_prefixes, chunk_snapshots, chunk_coefficients, chunks,
                                     batch);
    if (rows <= 32)
      tensor_contract(projection_partials, x, kDLFloat, 32, {rows <= 8 ? 8 : 3, rows, 5120});
    else
      tensor_contract(projection_partials, x, kDLFloat, 32, {1});
  }

  void validate_overlap() {
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
                    gates,
                    chunk_metadata,
                    chunk_offsets,
                    chunk_transforms,
                    chunk_prefixes,
                    chunk_snapshots,
                    chunk_coefficients,
                    slot_owners};
    int left = 0;
    for (auto a : tensors) {
      int right = 0;
      for (auto b : tensors)
        if (right++ > left)
          disjoint(a, b);
      ++left;
    }
  }

  void project_input(cudaStream_t stream) {
    const auto *input = static_cast<const __nv_bfloat16 *>(x.data_ptr());
    if (first_layer) {
      gdn_gather_kernel<<<rows, 256, 0, stream>>>(
          static_cast<const __nv_bfloat16 *>(embedding.data_ptr()),
          static_cast<const int64_t *>(tokens.data_ptr()),
          static_cast<const __nv_bfloat16 *>(media.data_ptr()),
          static_cast<const int *>(media_rows.data_ptr()),
          static_cast<__nv_bfloat16 *>(gathered.data_ptr()), rows, embedding.size(0),
          media.size(0));
      check_cuda(cudaGetLastError());
      input = static_cast<const __nv_bfloat16 *>(gathered.data_ptr());
    }
    dispatch_residual(residual, [&](auto skip) {
      residual_rms_quant_launch<true>(input, skip, static_cast<const float *>(norm.data_ptr()),
                                      static_cast<__nv_bfloat16 *>(summed.data_ptr()),
                                      static_cast<__nv_fp8_e4m3 *>(input_data.data_ptr()),
                                      static_cast<float *>(input_scales.data_ptr()), rows, stream,
                                      static_cast<__nv_bfloat16 *>(normalized.data_ptr()));
    });
    GDNInputConvolution convolution{
        static_cast<const __nv_bfloat16 *>(convolution_weight.data_ptr()),
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
          gdn_input_launch<Rows, true>(input_data, input_scales, qkvz_weight, qkvz_scale, qkvz,
                                       convolution, stream);
          return;
        }
      }
      gdn_input_launch<Rows, false>(input_data, input_scales, qkvz_weight, qkvz_scale, qkvz,
                                    convolution, stream);
    };
    if (rows == 1)
      project_input(std::integral_constant<int, 1>{});
    else if (rows == 4)
      project_input(std::integral_constant<int, 4>{});
    else if (rows == 5)
      project_input(std::integral_constant<int, 5>{});
    else
      radix_projection(static_cast<const __nv_fp8_e4m3 *>(input_data.data_ptr()),
                       static_cast<const __nv_fp8_e4m3 *>(qkvz_weight.data_ptr()),
                       static_cast<const float *>(input_scales.data_ptr()),
                       static_cast<const float *>(qkvz_scale.data_ptr()),
                       static_cast<__nv_bfloat16 *>(qkvz.data_ptr()), rows, 16384, 5120, stream);
  }

  void prepare_convolution(cudaStream_t stream) {
    bool single_query_verify = verify && max_queries == 1;
    if (rows > 32 || single_query_verify)
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
      if constexpr (!ProjectBA) {
        if (!verify && max_queries > 32) {
          auto tiled = [&](auto tile_rows) {
            constexpr int TileRows = decltype(tile_rows)::value;
            launch_kernel(gdn_convolution_tiled_kernel<TileRows>,
                          dim3((max_queries + TileRows - 1) / TileRows, 20, batch), dim3(128), 0,
                          stream, static_cast<const __nv_bfloat16 *>(qkvz.data_ptr()),
                          static_cast<const __nv_bfloat16 *>(convolution_weight.data_ptr()),
                          static_cast<__nv_bfloat16 *>(convolution_state.data_ptr()),
                          static_cast<const int *>(starts.data_ptr()),
                          static_cast<const int *>(reads.data_ptr()),
                          static_cast<const int *>(writes.data_ptr()),
                          static_cast<__nv_bfloat16 *>(qkv.data_ptr()), rows, batch, slots, verify);
          };
          if (max_queries < 4096)
            tiled(std::integral_constant<int, 4>{});
          else
            tiled(std::integral_constant<int, 8>{});
          return;
        }
      }
      launch_kernel(
          gdn_convolution_kernel<ProjectBA>, dim3(rows, ProjectBA ? 68 : 20),
          dim3(ProjectBA ? 256 : 128), 0, stream,
          static_cast<const __nv_bfloat16 *>(qkvz.data_ptr()),
          static_cast<const __nv_bfloat16 *>(convolution_weight.data_ptr()),
          static_cast<__nv_bfloat16 *>(convolution_state.data_ptr()),
          static_cast<const int *>(starts.data_ptr()), static_cast<const int *>(reads.data_ptr()),
          static_cast<const int *>(writes.data_ptr()), static_cast<__nv_bfloat16 *>(qkv.data_ptr()),
          rows, batch, slots, verify, static_cast<const __nv_bfloat16 *>(normalized.data_ptr()),
          static_cast<const __nv_bfloat16 *>(ba_weight.data_ptr()),
          static_cast<__nv_bfloat16 *>(ba.data_ptr()));
    };
    if (rows > 1 && rows <= 32 && !single_query_verify)
      prepare(std::true_type{});
    else if (!(rows == 1 && verify))
      prepare(std::false_type{});
    check_cuda(cudaGetLastError());
  }

  void update_state(cudaStream_t stream) {
    auto scan = [&](auto fused) {
      auto kernel = gdn_scan_quant_kernel<decltype(fused)::value>;
      launch_kernel(
          kernel, dim3(batch, 48), dim3(256), 0, stream,
          static_cast<const __nv_bfloat16 *>(qkv.data_ptr()),
          static_cast<const __nv_bfloat16 *>(qkvz.data_ptr()),
          static_cast<__nv_bfloat16 *>(ba.data_ptr()),
          static_cast<const __nv_bfloat16 *>(normalized.data_ptr()),
          static_cast<const __nv_bfloat16 *>(ba_weight.data_ptr()),
          static_cast<const float *>(a_log.data_ptr()),
          static_cast<const float *>(dt_bias.data_ptr()),
          static_cast<const float *>(head_norm.data_ptr()), static_cast<float *>(state.data_ptr()),
          static_cast<const int *>(starts.data_ptr()), static_cast<const int *>(reads.data_ptr()),
          static_cast<const int *>(writes.data_ptr()),
          static_cast<__nv_fp8_e4m3 *>(output_data.data_ptr()),
          static_cast<float *>(output_scales.data_ptr()), rows, slots);
    };
    if (chunked) {
      gdn_radix_wy::native_launch(
          static_cast<const __nv_bfloat16 *>(qkv.data_ptr()),
          static_cast<const float *>(gates.data_ptr()), static_cast<const int *>(starts.data_ptr()),
          static_cast<const int *>(reads.data_ptr()), static_cast<const int *>(writes.data_ptr()),
          static_cast<float *>(state.data_ptr()),
          static_cast<__nv_bfloat16 *>(recurrent.data_ptr()),
          static_cast<int *>(chunk_metadata.data_ptr()),
          static_cast<int *>(chunk_offsets.data_ptr()),
          static_cast<__nv_bfloat16 *>(chunk_transforms.data_ptr()),
          static_cast<float *>(chunk_prefixes.data_ptr()),
          static_cast<__nv_bfloat16 *>(chunk_snapshots.data_ptr()),
          static_cast<__nv_bfloat16 *>(chunk_coefficients.data_ptr()), rows, slots, batch, chunks,
          stream);
    } else if (precomputed_gates) {
      auto paired = [&](auto value_tile, auto kernel) {
        constexpr int ValueTile = decltype(value_tile)::value;
        launch_kernel(
            kernel, dim3(batch, 48, 128 / ValueTile), dim3(128),
            2 * GDNPrefetchSlotBytes<ValueTile>, stream,
            static_cast<const __nv_bfloat16 *>(qkv.data_ptr()),
            static_cast<float *>(state.data_ptr()), static_cast<const int *>(starts.data_ptr()),
            static_cast<const int *>(reads.data_ptr()), static_cast<const int *>(writes.data_ptr()),
            static_cast<__nv_bfloat16 *>(recurrent.data_ptr()),
            static_cast<const float *>(gates.data_ptr()), rows, slots);
      };
      if (gdn_prefill_two_pairs(rows, max_queries, x.device().device_id))
        paired(std::integral_constant<int, 32>{}, gdn_value_two_pair_kernel<32, 128>);
      else
        paired(std::integral_constant<int, 16>{}, gdn_value_pair_kernel<16, 128>);
    } else if (verify) {
      const auto *interleaved = static_cast<const __nv_bfloat16 *>(qkv.data_ptr());
      const auto *ba_values = static_cast<const __nv_bfloat16 *>(ba.data_ptr());
      gdn_recurrent::launch_interleaved(
          interleaved, interleaved + 2048, interleaved + 4096, ba_values + 48, ba_values,
          static_cast<const float *>(a_log.data_ptr()),
          static_cast<const float *>(dt_bias.data_ptr()),
          static_cast<const int *>(starts.data_ptr()), static_cast<const int *>(reads.data_ptr()),
          static_cast<const int *>(writes.data_ptr()), static_cast<float *>(state.data_ptr()),
          static_cast<__nv_bfloat16 *>(recurrent.data_ptr()), rows, batch, slots, max_queries,
          batch >= 4 ? 32 : 16, stream);
    } else if (rows == 1)
      scan(std::true_type{});
    else
      scan(std::false_type{});
    check_cuda(cudaGetLastError());
  }

  void project_output(cudaStream_t stream) {
    if (split_values) {
      auto quantize = [&](auto programmatic) {
        constexpr bool Programmatic = decltype(programmatic)::value;
        gdn_output_quant_kernel<Programmatic>
            <<<rows * 6, 256, 0, stream>>>(static_cast<const __nv_bfloat16 *>(recurrent.data_ptr()),
                                           static_cast<const __nv_bfloat16 *>(qkvz.data_ptr()),
                                           static_cast<const float *>(head_norm.data_ptr()),
                                           static_cast<__nv_fp8_e4m3 *>(output_data.data_ptr()),
                                           static_cast<float *>(output_scales.data_ptr()));
      };
      if (rows <= 32)
        quantize(std::true_type{});
      else
        quantize(std::false_type{});
      check_cuda(cudaGetLastError());
    }
    if (rows <= 32)
      output_projection<true>(output_data, output_weight, output_scales, output_scale,
                              projection_partials, output, rows, stream);
    else
      radix_projection(static_cast<const __nv_fp8_e4m3 *>(output_data.data_ptr()),
                       static_cast<const __nv_fp8_e4m3 *>(output_weight.data_ptr()),
                       static_cast<const float *>(output_scales.data_ptr()),
                       static_cast<const float *>(output_scale.data_ptr()),
                       static_cast<__nv_bfloat16 *>(output.data_ptr()), rows, 5120, 6144, stream);
  }
};

void gdn_radix(tvm::ffi::Tensor x, tvm::ffi::Tensor residual, tvm::ffi::Tensor norm,
               tvm::ffi::Tensor qkvz_weight, tvm::ffi::Tensor qkvz_scale,
               tvm::ffi::Tensor ba_weight, tvm::ffi::Tensor convolution_weight,
               tvm::ffi::Tensor a_log, tvm::ffi::Tensor dt_bias, tvm::ffi::Tensor head_norm,
               tvm::ffi::Tensor output_weight, tvm::ffi::Tensor output_scale,
               tvm::ffi::Tensor state, tvm::ffi::Tensor convolution_state, tvm::ffi::Tensor starts,
               tvm::ffi::Tensor reads, tvm::ffi::Tensor writes, tvm::ffi::Tensor embedding,
               tvm::ffi::Tensor tokens, tvm::ffi::Tensor media, tvm::ffi::Tensor media_rows,
               bool first_layer, bool verify, tvm::ffi::Tensor output, tvm::ffi::Tensor summed,
               tvm::ffi::Tensor gathered, tvm::ffi::Tensor normalized, tvm::ffi::Tensor input_data,
               tvm::ffi::Tensor input_scales, tvm::ffi::Tensor qkvz, tvm::ffi::Tensor ba,
               tvm::ffi::Tensor qkv, tvm::ffi::Tensor output_data, tvm::ffi::Tensor output_scales,
               tvm::ffi::Tensor recurrent, tvm::ffi::Tensor projection_partials,
               tvm::ffi::Tensor gates, tvm::ffi::Tensor chunk_metadata,
               tvm::ffi::Tensor chunk_offsets, tvm::ffi::Tensor chunk_transforms,
               tvm::ffi::Tensor chunk_prefixes, tvm::ffi::Tensor chunk_snapshots,
               tvm::ffi::Tensor chunk_coefficients, tvm::ffi::Tensor slot_owners, int max_queries) {
  RadixInvocation call{x,
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
                       first_layer,
                       verify,
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
                       gates,
                       chunk_metadata,
                       chunk_offsets,
                       chunk_transforms,
                       chunk_prefixes,
                       chunk_snapshots,
                       chunk_coefficients,
                       slot_owners,
                       max_queries};
  call.validate();
  call.validate_overlap();
  if (!call.rows)
    return;
  tvm::ffi::CUDADeviceGuard guard(x.device().device_id);
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, x.device().device_id));
  gdn_radix_metadata::validate(
      static_cast<const int *>(starts.data_ptr()), static_cast<const int *>(reads.data_ptr()),
      static_cast<const int *>(writes.data_ptr()), static_cast<int *>(slot_owners.data_ptr()),
      call.rows, call.batch, call.slots, max_queries, call.chunked ? call.chunks : -1, verify,
      stream);
  call.project_input(stream);
  call.prepare_convolution(stream);
  call.update_state(stream);
  call.project_output(stream);
}
