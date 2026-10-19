#pragma once

#include "gdn_wy_native.cuh"

namespace gdn_wy {
using NativeOutputCorrection = decltype(cute::make_tiled_mma(
    cute::SM100_MMA_F16BF16_SS<NativeElement, NativeElement, float, 128, 64, cute::UMMA::Major::MN,
                               cute::UMMA::Major::K>{}));
using NativeOutputDeltaLayout = decltype(cute::UMMA::tile_to_mma_shape(
    cute::UMMA::Layout_MN_SW128_Atom<NativeElement>{},
    cute::partition_shape_A(NativeOutputCorrection{}, cute::Shape<cute::_128, cute::_64>{})));

struct alignas(128) NativeOutputStorage {
  NativeElement states[128 * 128];
  NativeElement queries[64 * 128];
  NativeElement delta[128 * 64];
  NativeElement coefficients[64 * 64];
  uint64_t barrier;
  uint32_t tmem;
  float prefix[64], decay[64];
};

template <class ALayout, class BLayout, int Inner, class Product = NativeProjection>
__device__ inline void native_output_product(NativeElement *a, NativeElement *b, uint32_t tmem,
                                             uint64_t *barrier, ALayout a_layout,
                                             BLayout b_layout) {
  using namespace cute;
  if (threadIdx.x < 32) {
    Product mma;
    auto partial = partition_fragment_C(mma, Shape<_128, _64>{});
    partial.data() = tmem;
    mma.accumulate_ = UMMA::ScaleOut::Zero;
    auto at = native_tensor(a, a_layout);
    auto bt = native_tensor(b, b_layout);
    auto af = mma.get_slice(0).make_fragment_A(at);
    auto bf = mma.get_slice(0).make_fragment_B(bt);
#pragma unroll
    for (int k = 0; k < Inner / 16; ++k) {
      gemm(mma, af(_, _, k), bf(_, _, k), partial);
      mma.accumulate_ = UMMA::ScaleOut::One;
    }
    cutlass::arch::umma_arrive(barrier);
  }
}

template <class Snapshot>
__global__ __launch_bounds__(256, 3) void native_output_qk_kernel(
    const __nv_bfloat16 *qkv, const float *gates, const int *metadata,
    const __nv_bfloat16 *transforms, const float *prefixes, const Snapshot *snapshots,
    __nv_bfloat16 *output, const __nv_bfloat16 *coefficients) {
  using namespace cute;
  int chunk = blockIdx.x, head = blockIdx.y;
  int first = metadata[chunk * 4], count = metadata[chunk * 4 + 1] - first;
  if (count <= 0)
    return;
  int thread = threadIdx.x, row = thread % 128, segment = thread / 128;
  extern __shared__ __align__(128) unsigned char memory[];
  auto &s = *reinterpret_cast<NativeOutputStorage *>(memory);
  TMEM::Allocator1Sm allocator;
  if (thread < 32) {
    allocator.allocate(64, &s.tmem);
    allocator.release_allocation_lock();
  }
  if (thread == 0) {
    initialize_barrier(s.barrier, 1);
    cutlass::arch::fence_barrier_init();
  }
  auto states = native_tensor(s.states, NativeStateLayout{});
  auto queries = native_tensor(s.queries, NativeWeightLayout{});
  auto deltas = native_tensor(s.delta, NativeOutputDeltaLayout{});
#pragma unroll
  for (int local = 0; local < 8; ++local) {
    int index = thread + local * 256, value = index / 16, key = index % 16 * 8;
    int64_t offset = (int64_t(chunk) * 48 + head) * 128 * 128 + index * 8;
    if constexpr (std::is_same_v<Snapshot, __nv_bfloat16>) {
      native_copy16(&native_element(states, value, key), snapshots + offset);
    } else {
      float4 a = *reinterpret_cast<const float4 *>(snapshots + offset);
      float4 b = *reinterpret_cast<const float4 *>(snapshots + offset + 4);
      *reinterpret_cast<uint4 *>(&native_element(states, value, key)) =
          make_uint4(native_pair(a.x, a.y), native_pair(a.z, a.w), native_pair(b.x, b.y),
                     native_pair(b.z, b.w));
    }
  }
#pragma unroll
  for (int local = 0; local < 4; ++local) {
    int index = thread + local * 256, token = index / 16, key = index % 16 * 8;
    native_copy16(&native_element(queries, token, key),
                  qkv + int64_t(first + token) * 10240 + head / 3 * 128 + key, token < count);
    native_copy16(&native_element(deltas, key, token),
                  transforms + transform_offset(chunk, head, 0, 1) + index * 8);
  }
  for (int index = thread * 8; index < 64 * 64; index += 256 * 8)
    native_copy16(s.coefficients + index,
                  coefficients + (int64_t(chunk) * 48 + head) * 4096 + index);
  if (thread < 64) {
    s.prefix[thread] = prefixes[(int64_t(chunk) * 48 + head) * 128 + thread];
    s.decay[thread] = thread < count ? gates[int64_t(first + thread) * 96 + 48 + head] : 1.f;
  }
  asm volatile("cp.async.commit_group; cp.async.wait_group 0;" ::: "memory");
  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  native_output_product<NativeStateLayout, NativeWeightLayout, 128>(
      s.states, s.queries, s.tmem, &s.barrier, NativeStateLayout{}, NativeWeightLayout{});
  wait_barrier(s.barrier, 0);
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  uint32_t address = s.tmem + uint32_t(row / 32 * 32) * 65536u + segment * 32;
  float initial[32];
  native_load32(address, initial);
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  native_output_product<NativeOutputDeltaLayout, NativeCoefficientLayout, 64,
                        NativeOutputCorrection>(s.delta, s.coefficients, s.tmem, &s.barrier,
                                                NativeOutputDeltaLayout{},
                                                NativeCoefficientLayout{});
  wait_barrier(s.barrier, 1);
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  float correction[32];
  native_load32(address, correction);
  auto *result = reinterpret_cast<__nv_bfloat16 *>(s.states);
#pragma unroll
  for (int local = 0; local < 32; ++local) {
    int token = segment * 32 + local;
    result[token * 136 + row] = __float2bfloat16_rn(
        fmaf(initial[local], s.prefix[token], correction[local]) * 0.08838834764831844f);
  }
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  __syncthreads();
#pragma unroll
  for (int local = 0; local < 4; ++local) {
    int index = thread + local * 256, token = index / 16, value = index % 16 * 8;
    if (token < count)
      *reinterpret_cast<uint4 *>(output + (int64_t(first + token) * 48 + head) * 128 + value) =
          *reinterpret_cast<const uint4 *>(result + token * 136 + value);
  }
  if (thread < 32)
    allocator.free(s.tmem, 64);
}
} // namespace gdn_wy
