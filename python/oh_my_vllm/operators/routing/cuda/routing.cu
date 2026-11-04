#include <cuda_bf16.h>
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
    float score = sqrtf(value > 20.f ? value : log1pf(expf(value)));
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

  float selected[6];
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
    corrected[best] = -CUDART_INF_F;
  }
  float sum;
  // The released reduction uses four lanes for six values and two for three.
  if (topk == 6)
    sum = __fadd_rn(__fadd_rn(__fadd_rn(selected[0], selected[4]), selected[2]),
                    __fadd_rn(__fadd_rn(selected[1], selected[5]), selected[3]));
  else
    sum = __fadd_rn(__fadd_rn(selected[0], selected[2]), selected[1]);
  for (int slot = 0; slot < topk; ++slot)
    coefficients[row * topk + slot] = selected[slot] / (sum + 1e-20f) * 1.5f;
}

__global__ void expert_sum_kernel(const __nv_bfloat16 *values, const int32_t *route_rows,
                                  float *output, int width, int topk, int routed_rows,
                                  const int32_t *status = nullptr) {
  __shared__ int selected[6];
  if (threadIdx.x == 0) {
    for (int slot = 0; slot < topk; ++slot) {
      int row = route_rows[blockIdx.x * topk + slot];
      selected[slot] =
          (!status || *status == 0) && row >= 0 && row < routed_rows ? row : routed_rows;
    }
    // Projection rows follow increasing expert ID, independently of route slot order.
    for (int slot = 1; slot < topk; ++slot) {
      int row = selected[slot], previous = slot - 1;
      while (previous >= 0 && selected[previous] > row) {
        selected[previous + 1] = selected[previous];
        --previous;
      }
      selected[previous + 1] = row;
    }
  }
  __syncthreads();
  for (int column = blockIdx.y * blockDim.x + threadIdx.x; column < width;
       column += gridDim.y * blockDim.x) {
    float sum = 0.f;
    for (int slot = 0; slot < topk; ++slot) {
      int row = selected[slot];
      if (row < routed_rows)
        sum = __fadd_rn(sum, __bfloat162float(values[static_cast<int64_t>(row) * width + column]));
    }
    output[static_cast<int64_t>(blockIdx.x) * width + column] = sum;
  }
}

__global__ void padded_route_map(const int64_t *routes, const int32_t *counts, int32_t *route_rows,
                                 int32_t *status, int capacity, int groups, int slots) {
  __shared__ int live_rows;
  if (threadIdx.x < 32) {
    int64_t sum = 0;
    bool valid = true;
    for (int group = threadIdx.x; group < groups; group += 32) {
      int count = counts[group];
      valid = valid && count >= 0;
      sum += count;
    }
    unsigned invalid = __ballot_sync(0xffffffffu, !valid);
    for (int offset = 16; offset; offset /= 2)
      sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (threadIdx.x == 0) {
      valid = !invalid && sum <= capacity && sum <= slots;
      live_rows = valid ? static_cast<int>(sum) : 0;
      if (!valid)
        atomicOr(status, 1);
    }
  }
  __syncthreads();
  for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < live_rows;
       row += gridDim.x * blockDim.x) {
    int64_t slot = routes[row];
    if (slot < 0 || slot >= slots) {
      atomicOr(status, 1);
      continue;
    }
    if (atomicCAS(route_rows + slot, -1, row) != -1)
      atomicOr(status, 1);
  }
}

} // namespace

void route(TensorView logits, TensorView bias, TensorView bias_vl, TensorView image_mask,
           TensorView expert_ids, TensorView coefficients) {
  TVM_FFI_ICHECK(logits.ndim() == 2 && logits.size(0) >= 0 && logits.size(0) <= 32768);
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

void expert_sum(TensorView values, TensorView route_rows, TensorView output) {
  TVM_FFI_ICHECK(values.ndim() == 2 && route_rows.ndim() == 2);
  int64_t rows = route_rows.size(0), topk = route_rows.size(1), width = values.size(1);
  TVM_FFI_ICHECK(rows >= 0 && rows <= 32768 && topk >= 1 && topk <= 6);
  TVM_FFI_ICHECK(width > 0 && width <= 5120);
  TVM_FFI_ICHECK(values.size(0) >= 0 && values.size(0) <= rows * topk);
  tensor_contract(values, values, kDLBfloat, 16, {values.size(0), width});
  tensor_contract(route_rows, values, kDLInt, 32, {rows, topk});
  tensor_contract(output, values, kDLFloat, 32, {rows, width});
  disjoint(output, values);
  disjoint(output, route_rows);
  if (!rows)
    return;
  tvm::ffi::CUDADeviceGuard guard(values.device().device_id);
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, values.device().device_id));
  int column_blocks = rows >= 128 ? 1 : (width + 255) / 256;
  expert_sum_kernel<<<dim3(rows, column_blocks), 256, 0, stream>>>(
      static_cast<const __nv_bfloat16 *>(values.data_ptr()),
      static_cast<const int32_t *>(route_rows.data_ptr()), static_cast<float *>(output.data_ptr()),
      width, topk, values.size(0));
  check_cuda(cudaGetLastError());
}

void expert_sum_padded(TensorView values, TensorView sorted_routes, TensorView counts,
                       TensorView output, TensorView route_rows, TensorView status) {
  TVM_FFI_ICHECK(values.ndim() == 2 && route_rows.ndim() == 2 && counts.ndim() == 1);
  int64_t capacity = values.size(0), rows = route_rows.size(0), topk = route_rows.size(1);
  int64_t width = values.size(1), groups = counts.size(0);
  TVM_FFI_ICHECK(capacity >= 0 && capacity <= 1 << 20);
  TVM_FFI_ICHECK(rows >= 0 && rows <= 1 << 20 && topk >= 1 && topk <= 6);
  TVM_FFI_ICHECK(width > 0 && width <= 5120 && groups > 0 && groups <= 384);
  tensor_contract(values, values, kDLBfloat, 16, {capacity, width});
  tensor_contract(sorted_routes, values, kDLInt, 64, {capacity});
  tensor_contract(counts, values, kDLInt, 32, {groups});
  tensor_contract(output, values, kDLFloat, 32, {rows, width});
  tensor_contract(route_rows, values, kDLInt, 32, {rows, topk});
  tensor_contract(status, values, kDLInt, 32, {1});
  for (auto writable : {output, route_rows, status})
    for (auto input : {values, sorted_routes, counts})
      disjoint(writable, input);
  disjoint(output, route_rows);
  disjoint(output, status);
  disjoint(route_rows, status);
  tvm::ffi::CUDADeviceGuard guard(values.device().device_id);
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, values.device().device_id));
  if (rows)
    check_cuda(cudaMemsetAsync(route_rows.data_ptr(), 255, rows * topk * sizeof(int32_t), stream));
  int blocks = max(1, min(static_cast<int>((capacity + 255) / 256), 148 * 8));
  padded_route_map<<<blocks, 256, 0, stream>>>(
      static_cast<const int64_t *>(sorted_routes.data_ptr()),
      static_cast<const int32_t *>(counts.data_ptr()),
      static_cast<int32_t *>(route_rows.data_ptr()), static_cast<int32_t *>(status.data_ptr()),
      capacity, groups, rows * topk);
  check_cuda(cudaGetLastError());
  if (!rows)
    return;
  int column_blocks = rows >= 128 ? 1 : (width + 255) / 256;
  expert_sum_kernel<<<dim3(rows, column_blocks), 256, 0, stream>>>(
      static_cast<const __nv_bfloat16 *>(values.data_ptr()),
      static_cast<const int32_t *>(route_rows.data_ptr()), static_cast<float *>(output.data_ptr()),
      width, topk, capacity, static_cast<const int32_t *>(status.data_ptr()));
  check_cuda(cudaGetLastError());
}
