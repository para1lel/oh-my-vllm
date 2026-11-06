#pragma once

#include "gdn_wy_native.cuh"
#include "gdn_wy_bf16_inverse.cuh"
#include <type_traits>

namespace gdn_wy {
using NativeTransform = decltype(cute::make_tiled_mma(
    cute::SM100_MMA_F16BF16_SS<NativeElement, NativeElement, float, 64, 128, cute::UMMA::Major::K,
                               cute::UMMA::Major::MN>{}));
using NativeInverseLayout = decltype(cute::UMMA::tile_to_mma_shape(
    cute::UMMA::Layout_K_SW128_Atom<NativeElement>{},
    cute::partition_shape_A(NativeTransform{}, cute::Shape<cute::_64, cute::_64>{})));
using NativeOperandLayout = decltype(cute::UMMA::tile_to_mma_shape(
    cute::UMMA::Layout_MN_SW128_Atom<NativeElement>{},
    cute::partition_shape_B(NativeTransform{}, cute::Shape<cute::_128, cute::_64>{})));
using PrepareProjection = decltype(cute::make_tiled_mma(
    cute::SM100_MMA_F16BF16_SS<NativeElement, NativeElement, float, 64, 64, cute::UMMA::Major::K,
                               cute::UMMA::Major::K>{}));
using PrepareKeyLayout = decltype(cute::UMMA::tile_to_mma_shape(
    cute::UMMA::Layout_K_SW128_Atom<NativeElement>{},
    cute::partition_shape_A(PrepareProjection{}, cute::Shape<cute::_64, cute::_128>{})));
static_assert(std::is_same_v<PrepareKeyLayout, NativeWeightLayout>);

struct NativeInverseStorage {
  float products[64 * ProductStride];

  union {
    float inverse[64 * TokenStride];

    struct {
      __nv_bfloat16 inverse[64 * TokenStride], raw[64 * TokenStride];
    } low;
  };

  float temporary[32 * 32];
  float beta[64], decay[64], prefix[64], final_decay[64];
};

struct alignas(128) NativePrepareStorage {
  union {
    NativeInverseStorage inverse;

    struct {
      NativeElement a[64 * 128], q[64 * 128];
    } keys;

    struct {
      NativeElement a[64 * 64];
      NativeElement b[128 * 64];
      __nv_bfloat16 result[64 * 136];
    } transform;
  } data;

  uint64_t barrier;
  uint32_t tmem;
  float beta[64], prefix[64];
};

#include "gdn_wy_prepare_stages.cuh"

__global__ __launch_bounds__(256, 4) void native_prepare_bf16_inverse_kernel(
    const __nv_bfloat16 *qkv, const float *gates, const int *metadata, __nv_bfloat16 *transforms,
    float *prefixes, __nv_bfloat16 *coefficients) {
  using namespace cute;
  int chunk = blockIdx.x, head = blockIdx.y;
  int first = metadata[chunk * 4], count = metadata[chunk * 4 + 1] - first;
  if (count <= 0)
    return;
  int thread = threadIdx.x;
  int row = thread % 128 / 32 * 16 + thread % 16;
  int segment = thread / 128 * 64 + thread % 32 / 16 * 32;
  extern __shared__ __align__(128) unsigned char memory[];
  auto &s = *reinterpret_cast<NativePrepareStorage *>(memory);
  TMEM::Allocator1Sm allocator;
  if (thread < 32) {
    allocator.allocate(128, &s.tmem);
    allocator.release_allocation_lock();
  }
  if (thread == 0) {
    initialize_barrier(s.barrier, 1);
    cutlass::arch::fence_barrier_init();
  }
  prepare_key_products(s, qkv, first, count, head, thread);
  uint32_t address = s.tmem + uint32_t(row / 16 * 32) * 65536u;
  float products[32];
  if (thread < 128)
    carry_load32(address + segment, products);
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  __syncthreads();
  prepare_inverse_products(s, gates, prefixes, products, chunk, head, first, count, row, segment,
                           thread);
  auto &inverse = s.data.inverse;
  gdn_wy_bf16_inverse::inverse_blocks(inverse);
  __syncthreads();
  prepare_coefficients(s, coefficients, address, chunk, head, row, segment, thread);
  prepare_transforms(s, qkv, transforms, address, first, count, chunk, head, row, segment, thread);
  if (thread < 32)
    allocator.free(s.tmem, 128);
}
} // namespace gdn_wy
