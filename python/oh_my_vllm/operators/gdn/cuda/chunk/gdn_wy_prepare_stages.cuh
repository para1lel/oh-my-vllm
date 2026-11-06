#pragma once

__device__ __forceinline__ void prepare_key_products(NativePrepareStorage &s,
                                                     const __nv_bfloat16 *qkv, int first, int count,
                                                     int head, int thread) {
  using namespace cute;
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
}

__device__ __forceinline__ void prepare_inverse_products(NativePrepareStorage &s,
                                                         const float *gates, float *prefixes,
                                                         const float (&products)[32], int chunk,
                                                         int head, int first, int count, int row,
                                                         int segment, int thread) {
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
}

__device__ __forceinline__ void prepare_coefficients(NativePrepareStorage &s,
                                                     __nv_bfloat16 *coefficients, uint32_t address,
                                                     int chunk, int head, int row, int segment,
                                                     int thread) {
  auto &inverse = s.data.inverse;
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
            NativeElement(token >= row ? ratio * qk[local] * 0.08838834764831844f : 0.f);
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
}

template <class Inverse, class Operand>
__device__ __forceinline__ void prepare_transform_mma(NativePrepareStorage &s, Inverse &a_inverse,
                                                      Operand &b_operand, int thread) {
  using namespace cute;
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
}

__device__ __forceinline__ void prepare_transforms(NativePrepareStorage &s,
                                                   const __nv_bfloat16 *qkv,
                                                   __nv_bfloat16 *transforms, uint32_t address,
                                                   int first, int count, int chunk, int head,
                                                   int row, int segment, int thread) {
  using namespace cute;
  auto &inverse = s.data.inverse;
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
    native_element(a_inverse, token, previous) =
        NativeElement(triangular[index] * s.beta[previous]);
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
        words[pair] = native_pair(value.x, value.y);
      }
      *reinterpret_cast<uint4 *>(target) = make_uint4(words[0], words[1], words[2], words[3]);
    }
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    prepare_transform_mma(s, a_inverse, b_operand, thread);
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
}
