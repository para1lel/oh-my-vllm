#pragma once

namespace fa_ring_stages {
__device__ __forceinline__ void multiply(Shared &s, const Work &w) {
  auto sQ = make_tensor(make_smem_ptr(s.query_storage), QLayout{});
  QKMma qk;
  PVMma pv;
  auto qfrag = qk.get_slice(0).make_fragment_A(sQ);
  auto score = partition_fragment_C(qk, Shape<_128, _128>{});
  auto output = partition_fragment_C(pv, Shape<_128, _128>{});
  auto qk_chunk = [&](int tile, int chunk) {
    int slot = tile & 1, phase = (tile / 2) & 1;
    if (chunk == 0 && tile >= 2)
      wait_barrier(s.consumed[slot], phase ^ 1);
    wait_barrier(s.key_ready[chunk], tile & 1);
    fa_ring_after_sync();
    auto sK = make_tensor(make_smem_ptr(s.key_storage + chunk * cosize_v<KLayout>), KLayout{});
    auto kfrag = qk.get_slice(0).make_fragment_B(sK);
    score.data() = *s.tmem_base + slot * 128;
    qk.accumulate_ = chunk == 0 ? UMMA::ScaleOut::Zero : UMMA::ScaleOut::One;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      gemm(qk, qfrag(_, _, chunk * 8 + k), kfrag(_, _, k), score);
      qk.accumulate_ = UMMA::ScaleOut::One;
    }
    cutlass::arch::umma_arrive(s.key_consumed + chunk);
    if (chunk == 1)
      cutlass::arch::umma_arrive(s.score_ready + slot);
  };
  auto pv_chunk = [&](int tile, int chunk) {
    int slot = tile & 1, phase = (tile / 2) & 1;
    wait_barrier(s.value_ready[chunk], tile & 1);
    if (chunk == 0) {
      wait_barrier(s.probability_ready[slot], phase);
      wait_barrier(s.corrected[slot], phase);
    }
    fa_ring_after_sync();
    auto sP = make_tensor(make_smem_ptr(s.probability_storage), PLayout{});
    auto sV = make_tensor(make_smem_ptr(s.value_storage + chunk * cosize_v<VLayout>), VLayout{});
    auto pfrag = pv.get_slice(0).make_fragment_A(sP);
    auto vfrag = pv.get_slice(0).make_fragment_B(sV);
    pfrag.data() = *s.tmem_base + slot * 128 + 64;
    output.data() = *s.tmem_base + 256 + chunk * 128;
    pv.accumulate_ = UMMA::ScaleOut::One;
#pragma unroll
    for (int k = 0; k < 8; ++k)
      gemm(pv, pfrag(_, _, k), vfrag(_, _, k), output);
    cutlass::arch::umma_arrive(s.value_consumed + chunk);
    if (chunk == 1)
      cutlass::arch::umma_arrive(s.consumed + slot);
  };
  if (w.trips > 0) {
    qk_chunk(0, 0);
    qk_chunk(0, 1);
  }
  for (int tile = 0; tile < w.trips; ++tile) {
    if (tile + 1 < w.trips)
      qk_chunk(tile + 1, 0);
    pv_chunk(tile, 0);
    if (tile + 1 < w.trips)
      qk_chunk(tile + 1, 1);
    pv_chunk(tile, 1);
  }
  if (w.trips > 0)
    wait_barrier(s.consumed[(w.trips - 1) & 1], ((w.trips - 1) / 2) & 1);
  fa_ring_after_sync();
}

template <bool RoundMax>
__device__ __forceinline__ void load_scores(Shared &s, const Work &w, float (&values)[64],
                                            float maximum, uint32_t lane_base, int slot, int row,
                                            int base, int half, float &next_max) {
  float maxima[4] = {maximum, maximum, maximum, maximum};
  fa_ring_pair_load(lane_base + slot * 128, values);
  cutlass::arch::fence_view_async_tmem_load();
  bool complete = row < w.end && base + 128 <= w.length;
  if (complete) {
#pragma unroll
    for (int j = 0; j < 32; ++j) {
      float2 scaled = __fmul2_rn(make_float2(values[2 * j], values[2 * j + 1]),
                                 make_float2(0.09016844005556021f, 0.09016844005556021f));
      float v0 = scaled.x;
      float v1 = scaled.y;
      values[2 * j] = v0;
      values[2 * j + 1] = v1;
      maxima[(2 * j) % 4] = fmaxf(maxima[(2 * j) % 4], v0);
      maxima[(2 * j + 1) % 4] = fmaxf(maxima[(2 * j + 1) % 4], v1);
    }
  } else {
#pragma unroll
    for (int j = 0; j < 32; ++j) {
      float2 scaled = __fmul2_rn(make_float2(values[2 * j], values[2 * j + 1]),
                                 make_float2(0.09016844005556021f, 0.09016844005556021f));
      float v0 = row < w.end && base + half * 64 + 2 * j < w.length ? scaled.x : -INFINITY;
      float v1 = row < w.end && base + half * 64 + 2 * j + 1 < w.length ? scaled.y : -INFINITY;
      values[2 * j] = v0;
      values[2 * j + 1] = v1;
      maxima[(2 * j) % 4] = fmaxf(maxima[(2 * j) % 4], v0);
      maxima[(2 * j + 1) % 4] = fmaxf(maxima[(2 * j + 1) % 4], v1);
    }
  }
  next_max = fmaxf(fmaxf(maxima[0], maxima[1]), fmaxf(maxima[2], maxima[3]));
  next_max = fmaxf(next_max, __shfl_xor_sync(0xffffffff, next_max, 16));
  if constexpr (RoundMax)
    next_max = ceilf(next_max);
}

template <bool RoundMax> __device__ __forceinline__ void softmax(Shared &s, const Work &w) {
  int owner = (w.warp % 4) * 32 + (w.warp / 4) * 16 + w.lane % 16, half = w.lane / 16;
  uint32_t lane_base = *s.tmem_base + uint32_t((w.warp / 4) * 16) * 65536u;
  int row = w.start + (w.qbase + owner) / 6;
  float maximum = -INFINITY, denominator = 0;
  for (int tile = 0, base = w.first; tile < w.trips; ++tile, base += 128) {
    int slot = tile & 1, phase = (tile / 2) & 1;
    wait_barrier(s.score_ready[slot], phase);
    fa_ring_after_sync();
    float values[64], next_max = maximum;
    load_scores<RoundMax>(s, w, values, maximum, lane_base, slot, row, base, half, next_max);
    float safe = next_max == -INFINITY ? 0.f : next_max;
    float alpha = fa_exp2(maximum - safe), sum = 0;
    if (half == 0) {
      s.alpha_storage[slot * 128 + owner] = alpha;
      cutlass::arch::fence_view_async_shared();
      fa_ring_before_sync();
      Barrier::arrive(s.score_free + slot);
    }
    float packed[32];
    float2 sums[4] = {};
#pragma unroll
    for (int j = 0; j < 32; ++j) {
      float2 scaled =
          __fadd2_rn(make_float2(values[2 * j], values[2 * j + 1]), make_float2(-safe, -safe));
      float p0 = fa_exp2(scaled.x), p1 = fa_exp2(scaled.y);
      sums[j % 4] = __fadd2_rn(sums[j % 4], make_float2(p0, p1));
      uint32_t word;
      asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(word) : "f"(p0), "f"(p1));
      packed[j] = __uint_as_float(word);
    }
    float2 total = __fadd2_rn(__fadd2_rn(sums[0], sums[1]), __fadd2_rn(sums[2], sums[3]));
    sum = total.x + total.y;
    sum += __shfl_xor_sync(0xffffffff, sum, 16);
    fa_ring_pair_store(lane_base + slot * 128 + 64, packed);
    cutlass::arch::fence_view_async_tmem_store();
    fa_ring_before_sync();
    maximum = next_max;
    denominator = denominator * alpha + sum;
    Barrier::arrive(s.probability_ready + slot);
  }
  if (half == 0) {
    if (row < w.end) {
      int head = w.kh * 6 + (w.qbase + owner) % 6;
      w.lse[int64_t(row) * 24 + head] = maximum + log2f(denominator);
    }
    s.inverse_storage[owner] = denominator > 0 ? 1.f / denominator : 1.f;
  }
}

__device__ __forceinline__ void correct(Shared &s, const Work &w) {
  int owner = w.tid - 256;
  uint32_t address = *s.tmem_base + uint32_t(owner / 32 * 32) * 65536u + 256;
  for (int tile = 0; tile < w.trips; ++tile) {
    int slot = tile & 1, phase = (tile / 2) & 1;
    wait_barrier(s.score_free[slot], phase);
    if (tile > 0)
      wait_barrier(s.consumed[slot ^ 1], ((tile - 1) / 2) & 1);
    fa_ring_after_sync();
    float alpha = s.alpha_storage[slot * 128 + owner];
    if (__any_sync(0xffffffff, alpha != 1.f)) {
#pragma unroll 1
      for (int c = 0; c < 256; c += 64) {
        float block[64];
        fa_ring_tmem_load64(address + c, block);
        cutlass::arch::fence_view_async_tmem_load();
#pragma unroll
        for (int j = 0; j < 32; ++j) {
          float2 v =
              __fmul2_rn(make_float2(block[2 * j], block[2 * j + 1]), make_float2(alpha, alpha));
          block[2 * j] = v.x;
          block[2 * j + 1] = v.y;
        }
        fa_ring_tmem_store64(address + c, block);
      }
    }
    cutlass::arch::fence_view_async_tmem_store();
    fa_ring_before_sync();
    Barrier::arrive(s.corrected + slot);
  }
}

} // namespace fa_ring_stages
