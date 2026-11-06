#pragma once

namespace fa_prefill_stages {
__device__ __forceinline__ void multiply(Shared &s, const Work &w) {
  auto sQ = make_tensor(make_smem_ptr(s.query_storage), QLayout{});
  QKMma qk;
  PVMma pv;
  auto qfrag = qk.get_slice(0).make_fragment_A(sQ);
  auto score = partition_fragment_C(qk, Shape<_128, _64>{});
  auto output = partition_fragment_C(pv, Shape<_128, _256>{});
  output.data() = *s.tmem_base + 128;
  auto issue_qk = [&](int tile) {
    int slot = tile & 1, phase = (tile / 2) & 1;
    wait_barrier(s.key_ready[slot], phase);
    if (tile >= 2)
      wait_barrier(s.score_free[slot], phase ^ 1);
    fa_prefill_after_sync();
    auto sK = make_tensor(make_smem_ptr(s.key_storage + slot * cosize_v<KLayout>), KLayout{});
    auto kfrag = qk.get_slice(0).make_fragment_B(sK);
    score.data() = *s.tmem_base + slot * 64;
    qk.accumulate_ = UMMA::ScaleOut::Zero;
#pragma unroll
    for (int k = 0; k < 16; ++k) {
      gemm(qk, qfrag(_, _, k), kfrag(_, _, k), score);
      qk.accumulate_ = UMMA::ScaleOut::One;
    }
    cutlass::arch::umma_arrive(s.score_ready + slot);
  };
  if (w.trips > 0)
    issue_qk(0);
  for (int tile = 0; tile < w.trips; ++tile) {
    int slot = tile & 1, phase = (tile / 2) & 1;
    if (tile + 1 < w.trips)
      issue_qk(tile + 1);
    if (w.split_key_value)
      wait_barrier(s.value_ready[slot], phase);
    wait_barrier(s.probability_ready[slot], phase);
    wait_barrier(s.corrected[slot], phase);
    fa_prefill_after_sync();
    auto sP =
        make_tensor(make_smem_ptr(s.probability_storage + slot * cosize_v<PLayout>), PLayout{});
    auto sV = make_tensor(make_smem_ptr(s.value_storage + slot * cosize_v<VLayout>), VLayout{});
    auto pfrag = pv.get_slice(0).make_fragment_A(sP);
    auto vfrag = pv.get_slice(0).make_fragment_B(sV);
    pv.accumulate_ = UMMA::ScaleOut::One;
#pragma unroll
    for (int k = 0; k < 4; ++k)
      gemm(pv, pfrag(_, _, k), vfrag(_, _, k), output);
    cutlass::arch::umma_arrive(s.consumed + slot);
  }
  if (w.trips > 0)
    wait_barrier(s.consumed[(w.trips - 1) & 1], ((w.trips - 1) / 2) & 1);
  fa_prefill_after_sync();
}

__device__ __forceinline__ void softmax(Shared &s, const Work &w) {
  uint32_t lane_base = *s.tmem_base + uint32_t(w.warp * 32) * 65536u;
  int row = w.start + (w.qbase + w.tid) / 6;
  float maximum = -INFINITY, denominator = 0;
  for (int tile = 0, base = w.first; tile < w.trips; ++tile, base += 64) {
    int slot = tile & 1, phase = (tile / 2) & 1;
    if (tile >= 2)
      wait_barrier(s.consumed[slot], phase ^ 1);
    wait_barrier(s.score_ready[slot], phase);
    fa_prefill_after_sync();
    float values[64], next_max = maximum;
#pragma unroll
    for (int c = 0; c < 64; c += 32) {
      float block[32];
      fa_tmem_load32(lane_base + slot * 64 + c, block);
      cutlass::arch::fence_view_async_tmem_load();
#pragma unroll
      for (int j = 0; j < 32; ++j) {
        float s =
            row < w.end && base + c + j < w.length ? block[j] * 0.09016844005556021f : -INFINITY;
        values[c + j] = s;
        next_max = fmaxf(next_max, s);
      }
    }
    float safe = next_max == -INFINITY ? 0.f : next_max;
    float alpha = fa_exp2(maximum - safe), sum = 0;
    s.alpha_storage[slot * 128 + w.tid] = alpha;
    cutlass::arch::fence_view_async_shared();
    fa_prefill_before_sync();
    Barrier::arrive(s.score_free + slot);
    auto sP =
        make_tensor(make_smem_ptr(s.probability_storage + slot * cosize_v<PLayout>), PLayout{});
#pragma unroll
    for (int c = 0; c < 64; c += 8) {
      uint32_t packed[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        float p0 = fa_exp2(values[c + j * 2] - safe);
        sum += p0;
        float p1 = fa_exp2(values[c + j * 2 + 1] - safe);
        sum += p1;
        uint32_t word;
        asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(word) : "f"(p0), "f"(p1));
        packed[j] = word;
      }
      unsigned address = __cvta_generic_to_shared(&sP(make_coord(w.tid, c % 16), 0, c / 16));
      asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                   :
                   : "r"(address), "r"(packed[0]), "r"(packed[1]), "r"(packed[2]), "r"(packed[3])
                   : "memory");
    }
    maximum = next_max;
    denominator = denominator * alpha + sum;
    cutlass::arch::fence_view_async_shared();
    Barrier::arrive(s.probability_ready + slot);
  }
  if (row < w.end) {
    int head = w.kh * 6 + (w.qbase + w.tid) % 6;
    w.lse[(int64_t(row) * 24 + head) * w.splits + w.split] = maximum + log2f(denominator);
  }
  s.inverse_storage[w.tid] = denominator > 0 ? 1.f / denominator : 1.f;
}

__device__ __forceinline__ void correct(Shared &s, const Work &w) {
  int owner = w.tid - 128;
  uint32_t address = *s.tmem_base + uint32_t(owner / 32 * 32) * 65536u + 128;
  for (int tile = 0; tile < w.trips; ++tile) {
    int slot = tile & 1, phase = (tile / 2) & 1;
    wait_barrier(s.score_free[slot], phase);
    if (tile > 0)
      wait_barrier(s.consumed[slot ^ 1], ((tile - 1) / 2) & 1);
    fa_prefill_after_sync();
    float alpha = s.alpha_storage[slot * 128 + owner];
    if (__any_sync(0xffffffff, alpha != 1.f)) {
#pragma unroll 1
      for (int c = 0; c < 256; c += 32) {
        float block[32];
        fa_tmem_load32(address + c, block);
        cutlass::arch::fence_view_async_tmem_load();
#pragma unroll
        for (int j = 0; j < 32; ++j)
          block[j] *= alpha;
        fa_tmem_store32(address + c, block);
      }
    }
    cutlass::arch::fence_view_async_tmem_store();
    fa_prefill_before_sync();
    Barrier::arrive(s.corrected + slot);
  }
}

} // namespace fa_prefill_stages
