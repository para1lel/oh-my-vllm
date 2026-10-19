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
  auto a_keys = native_tensor(s.data.keys.a, PrepareKeyLayout{});
  auto b_keys = native_tensor(s.data.keys.a, NativeWeightLayout{});
  auto b_queries = native_tensor(s.data.keys.q, NativeWeightLayout{});
#pragma unroll
  for (int local = 0; local < 4; ++local) {
    int index = thread + local * 256, token = index / 16, key = index % 16 * 8;
    const auto *source = qkv + int64_t(first + token) * 10240 + 2048 + head / 3 * 128 + key;
    native_copy16(&native_element(a_keys, token, key), source, token < count);
    native_copy16(&native_element(b_queries, token, key), source - 2048, token < count);
  }
  asm volatile("cp.async.commit_group; cp.async.wait_group 0;" ::: "memory");
  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  if (thread < 32) {
    PrepareProjection mma;
    auto partial = partition_fragment_C(mma, Shape<_64, _64>{});
    partial.data() = s.tmem;
    auto af = mma.get_slice(0).make_fragment_A(a_keys);
    auto bf = mma.get_slice(0).make_fragment_B(b_keys);
    mma.accumulate_ = UMMA::ScaleOut::Zero;
#pragma unroll
    for (int inner = 0; inner < 8; ++inner) {
      gemm(mma, af(_, _, inner), bf(_, _, inner), partial);
      mma.accumulate_ = UMMA::ScaleOut::One;
    }
    partial.data() = s.tmem + 64;
    auto qf = mma.get_slice(0).make_fragment_B(b_queries);
    mma.accumulate_ = UMMA::ScaleOut::Zero;
#pragma unroll
    for (int inner = 0; inner < 8; ++inner) {
      gemm(mma, af(_, _, inner), qf(_, _, inner), partial);
      mma.accumulate_ = UMMA::ScaleOut::One;
    }
    cutlass::arch::umma_arrive(&s.barrier);
  }
  wait_barrier(s.barrier, 0);
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  uint32_t address = s.tmem + uint32_t(row / 16 * 32) * 65536u;
  float products[32];
  if (thread < 128)
    carry_load32(address + segment, products);
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  __syncthreads();
  auto &inverse = s.data.inverse;
#pragma unroll
  for (int local = 0; local < 32; ++local)
    if (thread < 128)
      inverse.products[row * ProductStride + segment + local] = products[local];
  if (thread < 64) {
    inverse.beta[thread] = thread < count ? gates[int64_t(first + thread) * 96 + head] : 0.f;
    inverse.decay[thread] = thread < count ? gates[int64_t(first + thread) * 96 + 48 + head] : 1.f;
  }
  __syncthreads();
  if (thread < 64) {
    float forward = inverse.decay[thread], reverse = inverse.decay[63 - thread];
#pragma unroll
    for (int distance = 1; distance < 32; distance *= 2) {
      float a = __shfl_up_sync(0xffffffffu, forward, distance);
      float b = __shfl_up_sync(0xffffffffu, reverse, distance);
      if (thread % 32 >= distance) {
        forward *= a;
        reverse *= b;
      }
    }
    inverse.prefix[thread] = forward;
    inverse.final_decay[63 - thread] = reverse;
  }
  __syncthreads();
  float forward = 1.f, reverse = 1.f;
  if (thread < 64) {
    forward = inverse.prefix[thread] * (thread >= 32 ? inverse.prefix[31] : 1.f);
    if (thread < 63)
      reverse = inverse.final_decay[thread + 1] * (thread < 31 ? inverse.final_decay[32] : 1.f);
  }
  __syncthreads();
  if (thread < 64) {
    inverse.prefix[thread] = forward;
    inverse.final_decay[thread] = reverse;
    prefixes[(int64_t(chunk) * 48 + head) * 128 + thread] = inverse.prefix[thread];
    prefixes[(int64_t(chunk) * 48 + head) * 128 + 64 + thread] = inverse.final_decay[thread];
    s.beta[thread] = inverse.beta[thread];
    s.prefix[thread] = forward;
  }
  int token = thread / 4, part = thread % 4;
  float ratios[16], ratio = 1.f;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    int previous = token - 1 - part * 16 - i;
    if (previous >= 0)
      ratio *= inverse.decay[previous + 1];
    ratios[i] = ratio;
  }
  float carry = 1.f;
#pragma unroll
  for (int prior = 0; prior < 3; ++prior) {
    float value = __shfl_sync(0xffffffffu, ratio, thread % 32 / 4 * 4 + prior);
    if (prior < part)
      carry *= value;
  }
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    int previous = token - 1 - part * 16 - i;
    if (previous >= 0)
      inverse.products[token * ProductStride + previous] *=
          inverse.beta[token] * (carry * ratios[i]);
  }
  __syncthreads();
  gdn_wy_bf16_inverse::inverse_blocks(inverse);
  __syncthreads();
  {
    auto staged_coefficients = native_tensor(reinterpret_cast<NativeElement *>(inverse.products),
                                             NativeCoefficientLayout{});
    if (thread < 128) {
      float qk[32];
      carry_load32(address + 64 + segment, qk);
      float source_prefix = inverse.prefix[row];
      bool quotient = source_prefix >= 0x1p-126f;
      float reciprocal = quotient ? 1.f / source_prefix : 0.f;
      float ratio = 1.f;
      if (!quotient)
        for (int token = row + 1; token < segment; ++token)
          ratio *= inverse.decay[token];
#pragma unroll
      for (int local = 0; local < 32; ++local) {
        int token = segment + local;
        if (quotient)
          ratio = inverse.prefix[token] * reciprocal;
        else if (token > row)
          ratio *= inverse.decay[token];
        native_element(staged_coefficients, token, row) =
            NativeElement(token >= row ? ratio * qk[local] : 0.f);
      }
    }
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads();
    for (int index = thread * 8; index < 64 * 64; index += 256 * 8)
      *reinterpret_cast<uint4 *>(coefficients + (int64_t(chunk) * 48 + head) * 4096 + index) =
          *reinterpret_cast<const uint4 *>(reinterpret_cast<NativeElement *>(inverse.products) +
                                           index);
    __syncthreads();
  }
  float triangular[16];
#pragma unroll
  for (int index = 0; index < 16; ++index) {
    int offset = thread + index * 256;
    triangular[index] =
        __bfloat162float(inverse.low.inverse[offset / 64 * TokenStride + offset % 64]);
  }
  __syncthreads();
  auto a_inverse = native_tensor(s.data.transform.a, NativeInverseLayout{});
  auto b_operand = native_tensor(s.data.transform.b, NativeOperandLayout{});
#pragma unroll
  for (int index = 0; index < 16; ++index) {
    int offset = thread + index * 256, token = offset / 64, previous = offset % 64;
    native_element(a_inverse, token, previous) = NativeElement(triangular[index]);
  }
  for (int kind = 0; kind < 2; ++kind) {
#pragma unroll
    for (int local = 0; local < 4; ++local) {
      int index = thread + local * 256, token = index / 16, column = index % 16 * 8;
      int input_column = kind == 0 ? 2048 + head / 3 * 128 + column : 4096 + head * 128 + column;
      native_copy16(&native_element(b_operand, column, token),
                    qkv + int64_t(first + token) * 10240 + input_column, token < count);
    }
    asm volatile("cp.async.commit_group; cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
#pragma unroll
    for (int local = 0; local < 4; ++local) {
      int index = thread + local * 256, token = index / 16, column = index % 16 * 8;
      auto *target = &native_element(b_operand, column, token);
      uint4 packed = *reinterpret_cast<const uint4 *>(target);
      uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
      for (int pair = 0; pair < 4; ++pair) {
        auto bits = *reinterpret_cast<const __nv_bfloat162 *>(words + pair);
        float2 value = __bfloat1622float2(bits);
        if (kind == 0) {
          value.x *= s.prefix[token];
          value.y *= s.prefix[token];
        }
        words[pair] = native_pair(value.x * s.beta[token], value.y * s.beta[token]);
      }
      *reinterpret_cast<uint4 *>(target) = make_uint4(words[0], words[1], words[2], words[3]);
    }
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    if (thread < 32) {
      NativeTransform mma;
      auto partial = partition_fragment_C(mma, Shape<_64, _128>{});
      partial.data() = s.tmem;
      auto af = mma.get_slice(0).make_fragment_A(a_inverse);
      auto bf = mma.get_slice(0).make_fragment_B(b_operand);
      mma.accumulate_ = UMMA::ScaleOut::Zero;
#pragma unroll
      for (int inner = 0; inner < 4; ++inner) {
        gemm(mma, af(_, _, inner), bf(_, _, inner), partial);
        mma.accumulate_ = UMMA::ScaleOut::One;
      }
      cutlass::arch::umma_arrive(&s.barrier);
    }
    wait_barrier(s.barrier, 1 - kind);
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    {
      float result[32];
      carry_load32(address + segment, result);
#pragma unroll
      for (int pair = 0; pair < 16; ++pair)
        *reinterpret_cast<__nv_bfloat162 *>(s.data.transform.result + row * 136 + segment +
                                            pair * 2) =
            __floats2bfloat162_rn(result[pair * 2], result[pair * 2 + 1]);
    }
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads();
    for (int index = thread * 8; index < 64 * 128; index += 256 * 8) {
      int token = index / 128, column = index % 128;
      *reinterpret_cast<uint4 *>(transforms + transform_offset(chunk, head, 0, kind) + index) =
          *reinterpret_cast<const uint4 *>(s.data.transform.result + token * 136 + column);
    }
    __syncthreads();
  }
  if (thread < 32)
    allocator.free(s.tmem, 128);
}
} // namespace gdn_wy
