#include "invocation.cuh"

void fa_fp8_run(TensorView x, TensorView residual, TensorView norm, TensorView weight,
                TensorView weight_scale, TensorView qnorm, TensorView knorm, TensorView positions,
                TensorView codes, TensorView cache_scales, TensorView slots, TensorView tables,
                TensorView starts, TensorView lengths, TensorView mask, bool use_mask,
                TensorView out_weight, TensorView out_weight_scale, TensorView output,
                TensorView summed, TensorView quantized, TensorView scales, TensorView qp,
                TensorView kvp, TensorView query, TensorView gate, TensorView partial,
                TensorView lse, TensorView gated, TensorView gated_scale,
                TensorView projection_partials, tvm::ffi::Shape ranges, int64_t variant,
                int64_t max_queries, int64_t first, int64_t max_context, int64_t min_context) {
  fa_fp8::Invocation call{x,
                          residual,
                          norm,
                          weight,
                          weight_scale,
                          qnorm,
                          knorm,
                          positions,
                          codes,
                          cache_scales,
                          slots,
                          tables,
                          starts,
                          lengths,
                          mask,
                          use_mask,
                          out_weight,
                          out_weight_scale,
                          output,
                          summed,
                          quantized,
                          scales,
                          qp,
                          kvp,
                          query,
                          gate,
                          partial,
                          lse,
                          gated,
                          gated_scale,
                          projection_partials,
                          ranges,
                          variant,
                          max_queries,
                          first,
                          max_context,
                          min_context};
  call.validate_plan();
  call.validate_inputs();
  call.validate_outputs();
  call.validate_overlap();
  tvm::ffi::CUDADeviceGuard guard(x.device().device_id);
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, x.device().device_id));
  call.validate_metadata(stream);
  call.input(stream);
  if (variant != 1)
    call.output_project(stream);
}
