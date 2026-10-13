#pragma once

#include "common.cuh"

void validate_gdn_rows(TensorView tensor, TensorView device, int64_t tokens, int64_t heads) {
  TVM_FFI_ICHECK(same_cuda_device(tensor, device) && tensor.ndim() == 3 &&
                 tensor.size(0) == tokens && tensor.size(1) == heads && tensor.size(2) == 128 &&
                 has_dtype(tensor, kDLBfloat, 16) && tensor.stride(2) == 1 &&
                 tensor.stride(1) == 128 && tensor.stride(0) >= heads * 128);
}

void validate_gdn_starts(TensorView starts, TensorView device) {
  TVM_FFI_ICHECK(same_cuda_device(starts, device) && starts.ndim() == 1 && starts.size(0) > 1 &&
                 starts.size(0) <= INT32_MAX && is_index_dtype(starts) && starts.stride(0) == 1);
}

void validate_gdn_states(TensorView state, TensorView device, int64_t sequences, int64_t heads) {
  TVM_FFI_ICHECK(same_cuda_device(state, device) && state.ndim() == 4 &&
                 state.size(0) == sequences && state.size(1) == heads && state.size(2) == 128 &&
                 state.size(3) == 128 && has_dtype(state, kDLFloat, 32) && state.stride(3) == 1 &&
                 state.stride(2) == 128 && state.stride(1) == 128 * 128 &&
                 state.stride(0) == heads * 128 * 128);
}

void validate_gdn_scratch(TensorView coefficients, TensorView cumulative, TensorView device,
                          int64_t chunks, int64_t heads) {
  TVM_FFI_ICHECK(chunks > 0 && chunks <= INT32_MAX && same_cuda_device(coefficients, device) &&
                 coefficients.ndim() == 4 && coefficients.size(0) == chunks &&
                 coefficients.size(1) == heads && coefficients.size(2) == 64 &&
                 coefficients.size(3) == 64 && has_dtype(coefficients, kDLBfloat, 16) &&
                 coefficients.stride(3) == 1 && coefficients.stride(2) == 64 &&
                 coefficients.stride(1) == 64 * 64 && coefficients.stride(0) == heads * 64 * 64 &&
                 reinterpret_cast<uintptr_t>(coefficients.data_ptr()) % 16 == 0);
  TVM_FFI_ICHECK(same_cuda_device(cumulative, device) && cumulative.ndim() == 3 &&
                 cumulative.size(0) == chunks && cumulative.size(1) == heads &&
                 cumulative.size(2) == 64 && has_dtype(cumulative, kDLFloat, 32) &&
                 cumulative.stride(2) == 1 && cumulative.stride(1) == 64 &&
                 cumulative.stride(0) == heads * 64 && disjoint_storage(coefficients, cumulative));
}
