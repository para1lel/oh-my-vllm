#pragma once

#include "attention.cuh"
#include "prefill.cuh"
#include "copy.cuh"
#include "input.cuh"
#include "metadata.cuh"
#include "../../primitives/cuda/groupwise.cuh"
#include "../../primitives/cuda/normalization.cuh"
#include "../../primitives/cuda/projection_output.cuh"
#include "../../primitives/cuda/projection_prefill.cuh"
#include <tvm/ffi/container/shape.h>

namespace fa_fp8 {
struct Invocation {
  TensorView x, residual, norm, weight, weight_scale, qnorm, knorm, positions;
  TensorView codes, cache_scales, slots, tables, starts, lengths, mask;
  bool use_mask;
  TensorView out_weight, out_weight_scale;
  TensorView output, summed, quantized, scales, qp, kvp, query, gate, partial, lse;
  TensorView gated, gated_scale, projection_partials;
  tvm::ffi::Shape ranges;
  int64_t variant, max_queries, first, max_context, min_context;
  int rows = 0, active = 0, written = 0, columns = 0;

  void validate_plan() {
    TVM_FFI_ICHECK(x.ndim() == 2 && x.size(1) == 5120 && x.size(0) > 0 && x.size(0) <= 32768);
    rows = x.size(0);
    TVM_FFI_ICHECK(variant >= 0 && variant <= 2 && (first == 0 || first == 1));
    active = variant == 1 ? 0 : rows;
    columns = variant == 1 ? 2048 : variant == 2 ? 12288 : 14336;
    TVM_FFI_ICHECK(tables.ndim() == 2 && tables.size(0) > 0 && tables.size(0) <= rows &&
                   tables.size(1) > 0 && tables.size(1) <= 2048);
    TVM_FFI_ICHECK(max_queries > 0 && max_queries <= rows && rows <= tables.size(0) * max_queries &&
                   max_context >= max_queries && max_context <= 262144 &&
                   max_context <= tables.size(1) * PageRows && min_context >= 0 &&
                   min_context <= max_context);
    TVM_FFI_ICHECK(ranges.size() % 2 == 0 && (variant != 2 || ranges.empty()));
    int64_t previous = 0;
    for (size_t i = 0; i < ranges.size(); i += 2) {
      TVM_FFI_ICHECK(previous <= ranges[i] && ranges[i] < ranges[i + 1] && ranges[i + 1] <= rows);
      written += ranges[i + 1] - ranges[i];
      previous = ranges[i + 1];
    }
    cache_contract(codes, cache_scales, x);
  }

  void validate_inputs() {
    tensor_contract(x, x, kDLBfloat, 16, {rows, 5120});
    residual_contract(residual, x, rows);
    tensor_contract(norm, x, kDLFloat, 32, {5120});
    tensor_contract(weight, x, kDLFloat8_e4m3fn, 8, {columns, 5120});
    tensor_contract(weight_scale, x, kDLFloat, 32, {columns / 128, 40});
    for (auto tensor : {qnorm, knorm})
      tensor_contract(tensor, x, kDLFloat, 32, {256});
    tensor_contract(positions, x, kDLInt, 64, {3, rows});
    tensor_contract(slots, x, kDLInt, 32, {rows});
    tensor_contract(tables, x, kDLInt, 32, {tables.size(0), tables.size(1)});
    tensor_contract(starts, x, kDLInt, 32, {tables.size(0) + 1});
    tensor_contract(lengths, x, kDLInt, 32, {rows});
    if (use_mask) {
      bool boolean = mask.dtype().code == kDLBool;
      tensor_contract(mask, x, boolean ? kDLBool : kDLInt, 8, {rows});
    }
    tensor_contract(out_weight, x, kDLFloat8_e4m3fn, 8, {5120, 6144});
    tensor_contract(out_weight_scale, x, kDLFloat, 32, {40, 48});
  }

  void validate_outputs() {
    tensor_contract(output, x, kDLBfloat, 16, {active, 5120});
    tensor_contract(summed, x, kDLBfloat, 16, {rows, 5120});
    tensor_contract(quantized, x, kDLFloat8_e4m3fn, 8, {rows, 5120});
    tensor_contract(scales, x, kDLFloat, 32, {rows, 40});
    tensor_contract(qp, x, kDLBfloat, 16, {active, 12288});
    tensor_contract(kvp, x, kDLBfloat, 16, {written, 2048});
    for (auto tensor : {query, gate})
      tensor_contract(tensor, x, kDLBfloat, 16, {active, 24, 256});
    int splits = rows <= 32 ? (tables.size(0) >= 3   ? 16
                               : tables.size(0) == 2 ? 32
                                                     : 64)
                 : rows <= 4096 && max_queries < 512 && tables.size(1) >= 64 ? 4
                                                                             : 1;
    tensor_contract(partial, x, kDLFloat, 32, {active, 24, splits, 256});
    tensor_contract(lse, x, kDLFloat, 32, {active, 24, splits});
    tensor_contract(gated, x, kDLFloat8_e4m3fn, 8, {active, 6144});
    tensor_contract(gated_scale, x, kDLFloat, 32, {active, 48});
    if (rows <= 32)
      tensor_contract(projection_partials, x, kDLFloat, 32, {rows <= 8 ? 8 : 3, active, 5120});
    else
      tensor_contract(projection_partials, x, kDLFloat, 32, {1});
  }

  void validate_overlap() {
    auto inputs = {x,         residual, norm,   weight, weight_scale, qnorm,      knorm,
                   positions, slots,    tables, starts, lengths,      out_weight, out_weight_scale};
    auto outputs = {codes,   cache_scales, output, summed,      quantized,
                    scales,  qp,           kvp,    query,       gate,
                    partial, lse,          gated,  gated_scale, projection_partials};
    int left = 0;
    for (auto a : outputs) {
      for (auto b : inputs)
        disjoint(a, b);
      if (use_mask)
        disjoint(a, mask);
      int right = 0;
      for (auto b : outputs)
        if (right++ > left)
          disjoint(a, b);
      ++left;
    }
  }

  void validate_metadata(cudaStream_t stream) {
    int previous = 0;
    auto check = [&](int start, int end, bool write) {
      if (start == end)
        return;
      validate_rows<<<(end - start + 255) / 256, 256, 0, stream>>>(
          static_cast<const int32_t *>(slots.data_ptr()),
          static_cast<const int8_t *>(mask.data_ptr()), use_mask, start, end, rows,
          codes.size(0) * PageRows, write);
    };
    for (size_t i = 0; i < ranges.size(); i += 2) {
      check(previous, ranges[i], false);
      check(ranges[i], ranges[i + 1], true);
      previous = ranges[i + 1];
    }
    check(previous, rows, false);
    if (variant != 1)
      validate_sequences<<<tables.size(0), 128, 0, stream>>>(
          static_cast<const int32_t *>(starts.data_ptr()),
          static_cast<const int32_t *>(lengths.data_ptr()),
          static_cast<const int32_t *>(tables.data_ptr()), rows, tables.size(0), tables.size(1),
          max_queries, max_context, first, codes.size(0));
    check_cuda(cudaGetLastError());
  }

  void project(int start, int count, int weight_start, int width, __nv_bfloat16 *output,
               cudaStream_t stream) {
    auto *input = static_cast<const __nv_fp8_e4m3 *>(quantized.data_ptr()) + int64_t(start) * 5120;
    auto *weights =
        static_cast<const __nv_fp8_e4m3 *>(weight.data_ptr()) + int64_t(weight_start) * 5120;
    auto *input_scale = static_cast<const float *>(scales.data_ptr()) + int64_t(start) * 40;
    auto *wscale =
        static_cast<const float *>(weight_scale.data_ptr()) + int64_t(weight_start / 128) * 40;
    if (count > 32)
      prefill_projection::launch(input, weights, input_scale, wscale, output, count, width, 5120,
                                 stream);
    else
      groupwise(input, weights, input_scale, wscale, output, count, width, 5120, stream);
  }

  void input(cudaStream_t stream) {
    dispatch_residual(residual, [&](auto skip) {
      residual_rms_quant_launch(static_cast<const __nv_bfloat16 *>(x.data_ptr()), skip,
                                static_cast<const float *>(norm.data_ptr()),
                                static_cast<__nv_bfloat16 *>(summed.data_ptr()),
                                static_cast<__nv_fp8_e4m3 *>(quantized.data_ptr()),
                                static_cast<float *>(scales.data_ptr()), rows, stream);
    });
    if (variant != 1) {
      project(0, rows, 0, 12288, static_cast<__nv_bfloat16 *>(qp.data_ptr()), stream);
      fa_heads_kernel<<<dim3(rows, 6), dim3(32, 4), 0, stream>>>(
          static_cast<const __nv_bfloat16 *>(qp.data_ptr()),
          static_cast<const float *>(qnorm.data_ptr()),
          static_cast<const float *>(knorm.data_ptr()),
          static_cast<const int64_t *>(positions.data_ptr()),
          static_cast<const int32_t *>(slots.data_ptr()), nullptr,
          static_cast<__nv_bfloat16 *>(query.data_ptr()),
          static_cast<__nv_bfloat16 *>(gate.data_ptr()), rows, 12288, 2, codes.size(0) * PageRows);
    }
    int packed_start = 0;
    for (size_t i = 0; i < ranges.size(); i += 2) {
      int start = ranges[i], count = ranges[i + 1] - start;
      project(start, count, variant == 1 ? 0 : 12288, 2048,
              static_cast<__nv_bfloat16 *>(kvp.data_ptr()) + int64_t(packed_start) * 2048, stream);
      kv_heads<<<dim3(count, 4), 32, 0, stream>>>(
          static_cast<const __nv_bfloat16 *>(kvp.data_ptr()),
          static_cast<const float *>(knorm.data_ptr()),
          static_cast<const int64_t *>(positions.data_ptr()),
          static_cast<const int32_t *>(slots.data_ptr()), static_cast<uint8_t *>(codes.data_ptr()),
          static_cast<float *>(cache_scales.data_ptr()), rows, start, packed_start,
          codes.size(0) * PageRows);
      packed_start += count;
    }
    check_cuda(cudaGetLastError());
  }

  void output_project(cudaStream_t stream) {
    auto *attended = static_cast<__nv_bfloat16 *>(qp.data_ptr());
    if (rows <= 32)
      attend(query, codes, cache_scales, tables, lengths, starts, partial, lse, max_queries, first,
             stream);
    else
      prefill_attend(query, codes, cache_scales, tables, lengths, starts, partial, lse, max_queries,
                     first, attended, stream);
    int splits = lse.size(2);
    if (rows > 32 && splits == 1)
      fa_bf16_one_split_quant_kernel<<<rows * 6, 256, 0, stream>>>(
          attended, static_cast<const __nv_bfloat16 *>(gate.data_ptr()),
          static_cast<__nv_fp8_e4m3 *>(gated.data_ptr()),
          static_cast<float *>(gated_scale.data_ptr()));
    else if (splits == 1)
      fa_one_split_quant_kernel<<<rows * 6, 256, 0, stream>>>(
          static_cast<const float *>(partial.data_ptr()),
          static_cast<const float *>(lse.data_ptr()),
          static_cast<const __nv_bfloat16 *>(gate.data_ptr()),
          static_cast<__nv_fp8_e4m3 *>(gated.data_ptr()),
          static_cast<float *>(gated_scale.data_ptr()));
    else if (splits > 4)
      fa_merge_quant_kernel<8>
          <<<rows * 48, 256, 0, stream>>>(static_cast<const float *>(partial.data_ptr()),
                                          static_cast<const float *>(lse.data_ptr()),
                                          static_cast<const __nv_bfloat16 *>(gate.data_ptr()),
                                          static_cast<__nv_fp8_e4m3 *>(gated.data_ptr()),
                                          static_cast<float *>(gated_scale.data_ptr()), splits);
    else
      fa_merge_quant_kernel<1, 8>
          <<<rows * 6, 256, 0, stream>>>(static_cast<const float *>(partial.data_ptr()),
                                         static_cast<const float *>(lse.data_ptr()),
                                         static_cast<const __nv_bfloat16 *>(gate.data_ptr()),
                                         static_cast<__nv_fp8_e4m3 *>(gated.data_ptr()),
                                         static_cast<float *>(gated_scale.data_ptr()), splits);
    check_cuda(cudaGetLastError());
    if (rows <= 32)
      output_projection(gated, out_weight, gated_scale, out_weight_scale, projection_partials,
                        output, rows, stream);
    else
      prefill_projection::launch(static_cast<const __nv_fp8_e4m3 *>(gated.data_ptr()),
                                 static_cast<const __nv_fp8_e4m3 *>(out_weight.data_ptr()),
                                 static_cast<const float *>(gated_scale.data_ptr()),
                                 static_cast<const float *>(out_weight_scale.data_ptr()),
                                 static_cast<__nv_bfloat16 *>(output.data_ptr()), rows, 5120, 6144,
                                 stream);
  }
};
} // namespace fa_fp8
