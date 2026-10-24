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

__device__ int warp_prefix(int value) {
  for (int delta = 1; delta < 32; delta *= 2) {
    int previous = __shfl_up_sync(0xffffffffU, value, delta);
    if ((threadIdx.x & 31) >= delta)
      value += previous;
  }
  return value;
}

__global__ void dispatch_metadata(const int32_t *ids, const float *routes, float *coefficients,
                                  int32_t *counts, int32_t *offsets, int32_t *map, uint8_t *status,
                                  int32_t *source_rows, int rows, int experts, int topk, int start,
                                  int groups) {
  __shared__ uint32_t members[384][12];
  __shared__ uint16_t earlier[384][12];
  __shared__ int starts[384], warp_sums[16];
  int thread = threadIdx.x, lane = thread & 31, warp = thread / 32;
  for (int i = thread; i < groups * 12; i += blockDim.x)
    members[i / 12][i % 12] = 0;
  for (int edge = thread; edge < rows * topk; edge += blockDim.x) {
    map[edge] = -1;
    source_rows[edge] = -1;
    coefficients[edge] = 0.f;
  }
  bool valid = thread < rows;
  if (valid) {
    for (int slot = 0; slot < topk; ++slot) {
      int edge = thread * topk + slot, expert = ids[edge];
      valid = valid && expert >= 0 && expert < experts && isfinite(routes[edge]);
      for (int previous = 0; previous < slot; ++previous)
        valid = valid && expert != ids[thread * topk + previous];
    }
    status[thread] = valid ? 0 : 1;
  }
  __syncthreads();
  if (valid)
    for (int slot = 0; slot < topk; ++slot) {
      int group = ids[thread * topk + slot] - start;
      if (group >= 0 && group < groups)
        atomicOr(&members[group][thread / 32], 1U << lane);
    }
  __syncthreads();
  int count = 0;
  if (thread < groups) {
    for (int word = 0; word < 12; ++word) {
      earlier[thread][word] = count;
      count += __popc(members[thread][word]);
    }
    counts[thread] = count;
  }
  int inclusive = warp_prefix(count);
  if (lane == 31)
    warp_sums[warp] = inclusive;
  __syncthreads();
  if (warp == 0) {
    int sum = warp_prefix(lane < 16 ? warp_sums[lane] : 0);
    if (lane < 16)
      warp_sums[lane] = sum;
  }
  __syncthreads();
  if (thread < groups) {
    int first = (warp ? warp_sums[warp - 1] : 0) + inclusive - count;
    starts[thread] = first;
    offsets[thread] = first;
    if (thread == groups - 1)
      offsets[groups] = first + count;
  }
  __syncthreads();
  // Shared integer bitmaps give token order without atomic slot allocation.
  if (valid)
    for (int slot = 0; slot < topk; ++slot) {
      int edge = thread * topk + slot, group = ids[edge] - start;
      if (group >= 0 && group < groups) {
        int destination = starts[group] + earlier[group][thread / 32] +
                          __popc(members[group][thread / 32] & ((1U << lane) - 1));
        map[edge] = destination;
        source_rows[destination] = thread;
        coefficients[destination] = routes[edge];
      }
    }
}

__global__ void dispatch_gather(const uint16_t *values, const int32_t *source_rows,
                                uint16_t *packed) {
  int destination = blockIdx.x, row = source_rows[destination];
  auto source = reinterpret_cast<const uint4 *>(values);
  auto target = reinterpret_cast<uint4 *>(packed);
  for (int column = threadIdx.x; column < 640; column += blockDim.x)
    target[destination * 640 + column] =
        row < 0 ? make_uint4(0, 0, 0, 0) : source[row * 640 + column];
}

__global__ void combine_kernel(const uint16_t *down, const int32_t *map, const uint8_t *status,
                               float *output, uint8_t *combined_status, int rows, int topk) {
  int row = blockIdx.x, capacity = rows * topk;
  __shared__ int slots[6], count, invalid;
  if (threadIdx.x == 0) {
    count = 0;
    invalid = status[row] != 0;
    for (int slot = 0; slot < topk; ++slot) {
      int value = map[row * topk + slot];
      if (value == -1)
        continue;
      if (value < 0 || value >= capacity) {
        invalid = 1;
        continue;
      }
      int position = count;
      for (int previous = 0; previous < count; ++previous)
        invalid |= slots[previous] == value;
      while (position && slots[position - 1] > value) {
        slots[position] = slots[position - 1];
        --position;
      }
      slots[position] = value;
      ++count;
    }
  }
  __syncthreads();
  bool bad = invalid;
  for (int column = threadIdx.x; column < 5120; column += blockDim.x) {
    float sum = 0.f;
    if (!invalid)
      for (int index = 0; index < count; ++index) {
        uint16_t bits = down[slots[index] * 5120 + column];
        if ((bits & 0x7f80) == 0x7f80) {
          bad = true;
          continue;
        }
        // Packed slots follow global expert order. Each add has one FP32 rounding.
        sum = __fadd_rn(sum, __uint_as_float(uint32_t(bits) << 16));
        bad |= !isfinite(sum);
      }
    output[row * 5120 + column] = sum;
  }
  int row_bad = __syncthreads_or(bad);
  if (row_bad)
    for (int column = threadIdx.x; column < 5120; column += blockDim.x)
      output[row * 5120 + column] = 0.f;
  if (threadIdx.x == 0)
    combined_status[row] = row_bad ? 1 : 0;
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

void dispatch(TensorView values, TensorView expert_ids, TensorView routes, TensorView packed,
              TensorView coefficients, TensorView counts, TensorView offsets, TensorView source_map,
              TensorView status, TensorView source_rows, int64_t experts, int64_t expert_start,
              int64_t expert_count) {
  TVM_FFI_ICHECK(values.ndim() == 2 && values.size(0) >= 0 && values.size(0) <= 384);
  TVM_FFI_ICHECK(experts == 128 || experts == 384);
  TVM_FFI_ICHECK(expert_count == experts || expert_count == experts / 4);
  TVM_FFI_ICHECK(expert_start >= 0 && expert_start <= experts - expert_count &&
                 expert_start % expert_count == 0);
  int rows = values.size(0), topk = experts == 384 ? 6 : 3, capacity = rows * topk;
  tensor_contract(values, values, kDLBfloat, 16, {rows, 5120});
  tensor_contract(expert_ids, values, kDLInt, 32, {rows, topk});
  tensor_contract(routes, values, kDLFloat, 32, {rows, topk});
  tensor_contract(packed, values, kDLBfloat, 16, {capacity, 5120});
  tensor_contract(coefficients, values, kDLFloat, 32, {capacity});
  tensor_contract(counts, values, kDLInt, 32, {expert_count});
  tensor_contract(offsets, values, kDLInt, 32, {expert_count + 1});
  tensor_contract(source_map, values, kDLInt, 32, {rows, topk});
  tensor_contract(status, values, kDLUInt, 8, {rows});
  tensor_contract(source_rows, values, kDLInt, 32, {capacity});
  TensorView outputs[] = {packed, coefficients, counts, offsets, source_map, status, source_rows};
  for (int i = 0; i < 7; ++i) {
    for (auto input : {values, expert_ids, routes})
      disjoint(outputs[i], input);
    for (int j = 0; j < i; ++j)
      disjoint(outputs[i], outputs[j]);
  }
  tvm::ffi::CUDADeviceGuard guard(values.device().device_id);
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, values.device().device_id));
  dispatch_metadata<<<1, 512, 0, stream>>>(
      static_cast<const int32_t *>(expert_ids.data_ptr()),
      static_cast<const float *>(routes.data_ptr()), static_cast<float *>(coefficients.data_ptr()),
      static_cast<int32_t *>(counts.data_ptr()), static_cast<int32_t *>(offsets.data_ptr()),
      static_cast<int32_t *>(source_map.data_ptr()), static_cast<uint8_t *>(status.data_ptr()),
      static_cast<int32_t *>(source_rows.data_ptr()), rows, experts, topk, expert_start,
      expert_count);
  if (rows)
    dispatch_gather<<<capacity, 256, 0, stream>>>(
        static_cast<const uint16_t *>(values.data_ptr()),
        static_cast<const int32_t *>(source_rows.data_ptr()),
        static_cast<uint16_t *>(packed.data_ptr()));
  check_cuda(cudaGetLastError());
}

void combine(TensorView down, TensorView source_map, TensorView status, TensorView output,
             TensorView combined_status) {
  TVM_FFI_ICHECK(source_map.ndim() == 2 && source_map.size(0) >= 0 && source_map.size(0) <= 384);
  TVM_FFI_ICHECK(source_map.size(1) == 3 || source_map.size(1) == 6);
  int rows = source_map.size(0), topk = source_map.size(1);
  tensor_contract(down, down, kDLBfloat, 16, {rows * topk, 5120});
  tensor_contract(source_map, down, kDLInt, 32, {rows, topk});
  tensor_contract(status, down, kDLUInt, 8, {rows});
  tensor_contract(output, down, kDLFloat, 32, {rows, 5120});
  tensor_contract(combined_status, down, kDLUInt, 8, {rows});
  for (auto target : {output, combined_status})
    for (auto input : {down, source_map, status})
      disjoint(target, input);
  disjoint(output, combined_status);
  if (!rows)
    return;
  tvm::ffi::CUDADeviceGuard guard(down.device().device_id);
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, down.device().device_id));
  combine_kernel<<<rows, 256, 0, stream>>>(
      static_cast<const uint16_t *>(down.data_ptr()),
      static_cast<const int32_t *>(source_map.data_ptr()),
      static_cast<const uint8_t *>(status.data_ptr()), static_cast<float *>(output.data_ptr()),
      static_cast<uint8_t *>(combined_status.data_ptr()), rows, topk);
  check_cuda(cudaGetLastError());
}
