#pragma once

constexpr int AcceptBlocks = (Vocabulary + 255) / 256;

__device__ inline double acceptance_uniform(curandStatePhilox4_32_10_t *state) {
  uint64_t bits = curand(state);
  bits ^= uint64_t(curand(state)) << 21;
  return double(bits) * 0x1p-53 + 0x1p-54;
}

__global__ void mtp_accept_metadata_kernel(const float *target, const float *proposal,
                                           const int64_t *drafts, const int64_t *target_tokens,
                                           uint64_t *rng, const int *private_slots, int64_t *kept,
                                           int *counts, int *selected_slots, double *uniform,
                                           int state_capacity, int batch) {
  int request = blockIdx.x;
  // The fifth draft field marks a one-hot proposal distribution.
  int64_t greedy_proposal = drafts[request * 5 + 4];
  if ((greedy_proposal != 0 && greedy_proposal != 1) || (target_tokens && !greedy_proposal)) {
    asm volatile("trap;");
    return;
  }
  for (int i = 0; i < 5; ++i) {
    int slot = private_slots[request * 5 + i];
    if (slot < 0 || slot >= state_capacity) {
      asm volatile("trap;");
      return;
    }
    for (int other = 0; other < batch * 5; ++other)
      if (other != request * 5 + i && slot == private_slots[other]) {
        asm volatile("trap;");
        return;
      }
    kept[request * 7 + i] = -1;
  }
  curandStatePhilox4_32_10_t state;
  curand_init(rng[request * 2], 0, rng[request * 2 + 1], &state);
  int attempts = 0, accepted = 0;
  bool inactive = false;
  for (int i = 0; i < 4; ++i) {
    int64_t token = drafts[request * 5 + i];
    if (token == -1)
      inactive = true;
    else if (inactive || token < 0 || token >= Vocabulary) {
      asm volatile("trap;");
      return;
    }
  }
  for (int i = 0; i < 4; ++i) {
    int64_t token = drafts[request * 5 + i];
    if (token == -1)
      break;
    float p = target_tokens ? float(target_tokens[request * 5 + i] == token)
                            : target[(request * 5 + i) * Vocabulary + token];
    float q = greedy_proposal ? 1.f : proposal[(request * 4 + i) * Vocabulary + token];
    if (!(isfinite(p) && p >= 0 && isfinite(q) && q >= 0)) {
      asm volatile("trap;");
      return;
    }
    double draw = acceptance_uniform(&state);
    ++attempts;
    if (q == 0 || double(p / q) < draw)
      break;
    kept[request * 7 + i] = token;
    ++accepted;
  }
  uniform[request] = acceptance_uniform(&state);
  rng[request * 2 + 1] += 2 * (attempts + 1);
  counts[request] = accepted + 1;
  kept[request * 7 + 5] = accepted + 1;
  reinterpret_cast<uint64_t *>(kept)[request * 7 + 6] = rng[request * 2 + 1];
  selected_slots[request] = private_slots[request * 5 + accepted];
  if (target_tokens) {
    int64_t token = target_tokens[request * 5 + accepted];
    if (token < 0 || token >= Vocabulary) {
      asm volatile("trap;");
      return;
    }
    kept[request * 7 + accepted] = token;
  }
}

__global__ void mtp_accept_mass_kernel(const float *target, const float *proposal,
                                       const int64_t *drafts, const int *counts, float *partials) {
  using Reduce = cub::BlockReduce<float, 256>;
  __shared__ typename Reduce::TempStorage temporary;
  int request = blockIdx.x, block = blockIdx.y;
  int token = block * 256 + threadIdx.x, accepted = counts[request] - 1;
  float mass = 0;
  bool rejected = accepted < 4 && drafts[request * 5 + accepted] >= 0;
  if (token < Vocabulary) {
    float p = target[(request * 5 + accepted) * Vocabulary + token];
    float q = rejected ? (drafts[request * 5 + 4]
                              ? float(token == drafts[request * 5 + accepted])
                              : proposal[(request * 4 + accepted) * Vocabulary + token])
                       : 0;
    if (!(isfinite(p) && p >= 0 && isfinite(q) && q >= 0)) {
      asm volatile("trap;");
      return;
    }
    mass = rejected ? fmaxf(p - q, 0.f) : p;
  }
  float total = Reduce(temporary).Sum(mass);
  if (!threadIdx.x)
    partials[request * AcceptBlocks + block] = total;
}

__global__ void mtp_accept_select_kernel(const float *target, const float *proposal,
                                         const int64_t *drafts, const int *counts,
                                         const double *uniform, const float *partials,
                                         int64_t *kept) {
  using Scan = cub::BlockScan<float, 256>;
  __shared__ typename Scan::TempStorage temporary;
  __shared__ int selected_block, last_block, selected_token, last_token;
  __shared__ float block_prefix;
  int request = blockIdx.x, lane = threadIdx.x, accepted = counts[request] - 1;
  bool rejected = accepted < 4 && drafts[request * 5 + accepted] >= 0;
  if (!lane) {
    selected_block = AcceptBlocks;
    last_block = -1;
    selected_token = Vocabulary;
    last_token = -1;
  }
  float masses[4], inclusive[4], total;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int block = lane * 4 + i;
    masses[i] = block < AcceptBlocks ? partials[request * AcceptBlocks + block] : 0;
  }
  Scan(temporary).InclusiveSum(masses, inclusive, total);
  if (!(total > 0 && isfinite(total))) {
    asm volatile("trap;");
    return;
  }
  double threshold = uniform[request] * total;
  __syncthreads();
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    if (masses[i] > 0) {
      int block = lane * 4 + i;
      atomicMax(&last_block, block);
      if (inclusive[i] >= threshold)
        atomicMin(&selected_block, block);
    }
  }
  __syncthreads();
  if (!lane && selected_block == AcceptBlocks)
    selected_block = last_block;
  __syncthreads();
  if (selected_block < 0 || selected_block >= AcceptBlocks) {
    asm volatile("trap;");
    return;
  }
  if (lane == selected_block / 4)
    block_prefix = inclusive[selected_block % 4] - masses[selected_block % 4];
  __syncthreads();
  int token = selected_block * 256 + lane;
  float mass = 0;
  if (token < Vocabulary) {
    float p = target[(request * 5 + accepted) * Vocabulary + token];
    float q = rejected ? (drafts[request * 5 + 4]
                              ? float(token == drafts[request * 5 + accepted])
                              : proposal[(request * 4 + accepted) * Vocabulary + token])
                       : 0;
    mass = rejected ? fmaxf(p - q, 0.f) : p;
  }
  float cumulative;
  Scan(temporary).InclusiveSum(mass, cumulative);
  if (mass > 0) {
    atomicMax(&last_token, token);
    if (block_prefix + cumulative >= threshold)
      atomicMin(&selected_token, token);
  }
  __syncthreads();
  if (!lane) {
    if (selected_token == Vocabulary)
      selected_token = last_token;
    if (selected_token < 0) {
      asm volatile("trap;");
      return;
    }
    kept[request * 7 + accepted] = selected_token;
  }
}

void mtp_accept(TensorView target, TensorView proposal, TensorView drafts, TensorView rng,
                TensorView private_slots, TensorView kept, TensorView counts,
                TensorView selected_slots, TensorView uniform, TensorView partials,
                int64_t state_capacity) {
  TVM_FFI_ICHECK((target.ndim() == 2 || target.ndim() == 3) && target.size(0) > 0 &&
                 state_capacity > 0 && state_capacity <= INT32_MAX);
  bool greedy = target.dtype().code == kDLInt && target.dtype().bits == 64;
  int batch = greedy || target.ndim() == 3 ? target.size(0) : target.size(0) / 5;
  TVM_FFI_ICHECK(batch > 0 && batch <= 4);
  if (greedy)
    tensor_contract(target, target, kDLInt, 64, {batch, 5});
  else if (target.ndim() == 3)
    tensor_contract(target, target, kDLFloat, 32, {batch, 5, Vocabulary});
  else
    tensor_contract(target, target, kDLFloat, 32, {batch * 5, Vocabulary});
  tensor_contract(proposal, target, kDLFloat, 32, {batch, 4, Vocabulary});
  tensor_contract(drafts, target, kDLInt, 64, {batch, 5});
  tensor_contract(rng, target, kDLUInt, 64, {batch, 2});
  tensor_contract(private_slots, target, kDLInt, 32, {batch, 5});
  tensor_contract(kept, target, kDLInt, 64, {batch, 7});
  for (auto tensor : {counts, selected_slots})
    tensor_contract(tensor, target, kDLInt, 32, {batch});
  tensor_contract(uniform, target, kDLFloat, 64, {batch});
  tensor_contract(partials, target, kDLFloat, 32, {batch, AcceptBlocks});
  auto tensors = {target, proposal, drafts,         rng,     private_slots,
                  kept,   counts,   selected_slots, uniform, partials};
  int left = 0;
  for (auto a : tensors) {
    int right = 0;
    for (auto b : tensors)
      if (right++ > left)
        disjoint(a, b);
    ++left;
  }
  tvm::ffi::CUDADeviceGuard guard(target.device().device_id);
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, target.device().device_id));
  mtp_accept_metadata_kernel<<<batch, 1, 0, stream>>>(
      static_cast<const float *>(target.data_ptr()),
      static_cast<const float *>(proposal.data_ptr()),
      static_cast<const int64_t *>(drafts.data_ptr()),
      greedy ? static_cast<const int64_t *>(target.data_ptr()) : nullptr,
      static_cast<uint64_t *>(rng.data_ptr()), static_cast<const int *>(private_slots.data_ptr()),
      static_cast<int64_t *>(kept.data_ptr()), static_cast<int *>(counts.data_ptr()),
      static_cast<int *>(selected_slots.data_ptr()), static_cast<double *>(uniform.data_ptr()),
      state_capacity, batch);
  check_cuda(cudaGetLastError());
  if (greedy)
    return;
  mtp_accept_mass_kernel<<<dim3(batch, AcceptBlocks), 256, 0, stream>>>(
      static_cast<const float *>(target.data_ptr()),
      static_cast<const float *>(proposal.data_ptr()),
      static_cast<const int64_t *>(drafts.data_ptr()), static_cast<const int *>(counts.data_ptr()),
      static_cast<float *>(partials.data_ptr()));
  check_cuda(cudaGetLastError());
  mtp_accept_select_kernel<<<batch, 256, 0, stream>>>(
      static_cast<const float *>(target.data_ptr()),
      static_cast<const float *>(proposal.data_ptr()),
      static_cast<const int64_t *>(drafts.data_ptr()), static_cast<const int *>(counts.data_ptr()),
      static_cast<const double *>(uniform.data_ptr()),
      static_cast<const float *>(partials.data_ptr()), static_cast<int64_t *>(kept.data_ptr()));
  check_cuda(cudaGetLastError());
}
