#include <math_constants.h>
#include "../../runtime/cuda/contracts.cuh"

namespace {

__global__ void route_kernel(const float *logits, const float *bias, const float *bias_vl,
                             const uint8_t *image_mask, int32_t *expert_ids, float *coefficients,
                             int experts, int topk) {
  int row = blockIdx.x;
  const float *selected_bias = image_mask[row] == 1 ? bias_vl : bias;
  __shared__ float scores[384], corrected[384];
  for (int expert = threadIdx.x; expert < experts; expert += blockDim.x) {
    float value = logits[row * experts + expert];
    float score = sqrtf(fmaxf(value, 0.f) + log1pf(expf(-fabsf(value))));
    scores[expert] = score;
    corrected[expert] = score + selected_bias[expert];
  }
  __syncthreads();
  if (threadIdx.x != 0)
    return;

  bool valid = image_mask[row] <= 1;
  for (int expert = 0; expert < experts; ++expert)
    valid = valid && isfinite(logits[row * experts + expert]) && isfinite(selected_bias[expert]) &&
            isfinite(corrected[expert]);
  if (!valid) {
    for (int slot = 0; slot < topk; ++slot) {
      expert_ids[row * topk + slot] = -1;
      coefficients[row * topk + slot] = CUDART_NAN_F;
    }
    return;
  }

  float selected[6], sum = 0.f;
  for (int slot = 0; slot < topk; ++slot) {
    int best = -1;
    float peak = -CUDART_INF_F;
    // Ascending traversal keeps the lowest global ID on exact ties.
    for (int expert = 0; expert < experts; ++expert) {
      if (corrected[expert] > peak) {
        best = expert;
        peak = corrected[expert];
      }
    }
    expert_ids[row * topk + slot] = best;
    selected[slot] = scores[best];
    sum += selected[slot];
    corrected[best] = -CUDART_INF_F;
  }
  for (int slot = 0; slot < topk; ++slot)
    coefficients[row * topk + slot] = selected[slot] / (sum + 1e-20f) * 1.5f;
}

} // namespace

void route(TensorView logits, TensorView bias, TensorView bias_vl, TensorView image_mask,
           TensorView expert_ids, TensorView coefficients) {
  TVM_FFI_ICHECK(logits.ndim() == 2 && logits.size(0) >= 0 && logits.size(0) <= 384);
  int rows = logits.size(0);
  TVM_FFI_ICHECK(logits.size(1) == 128 || logits.size(1) == 384);
  int experts = logits.size(1), topk = experts == 384 ? 6 : 3;
  tensor_contract(logits, logits, kDLFloat, 32, {rows, experts});
  tensor_contract(bias, logits, kDLFloat, 32, {experts});
  tensor_contract(bias_vl, logits, kDLFloat, 32, {experts});
  tensor_contract(image_mask, logits, kDLUInt, 8, {rows});
  tensor_contract(expert_ids, logits, kDLInt, 32, {rows, topk});
  tensor_contract(coefficients, logits, kDLFloat, 32, {rows, topk});
  for (auto output : {expert_ids, coefficients})
    for (auto input : {logits, bias, bias_vl, image_mask})
      disjoint(output, input);
  disjoint(expert_ids, coefficients);
  if (!rows)
    return;
  tvm::ffi::CUDADeviceGuard guard(logits.device().device_id);
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, logits.device().device_id));
  route_kernel<<<rows, 128, 0, stream>>>(
      static_cast<const float *>(logits.data_ptr()), static_cast<const float *>(bias.data_ptr()),
      static_cast<const float *>(bias_vl.data_ptr()),
      static_cast<const uint8_t *>(image_mask.data_ptr()),
      static_cast<int32_t *>(expert_ids.data_ptr()), static_cast<float *>(coefficients.data_ptr()),
      experts, topk);
  check_cuda(cudaGetLastError());
}
