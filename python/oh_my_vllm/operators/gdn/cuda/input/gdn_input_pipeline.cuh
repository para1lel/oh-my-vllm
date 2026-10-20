#pragma once

template <int Rows, bool FuseConvolution, bool Streaming>
__global__ void gdn_input_pipeline_kernel(const __nv_fp8_e4m3 *input, const float *input_scale,
                                          const float *weight_scale, __nv_bfloat16 *output,
                                          const __grid_constant__ CUtensorMap weight_map,
                                          GDNInputConvolution convolution) {
  static_assert(Rows == 1 || Rows == 4 || Rows == 5);
  using namespace cute;
  using Element = cutlass::float_e4m3_t;
  using MMA = decltype(make_tiled_mma(
      SM100_MMA_F8F6F4_SS<Element, Element, float, 64, 8, UMMA::Major::K, UMMA::Major::K>{}));
  using WeightLayout = decltype(UMMA::tile_to_mma_shape(
      UMMA::Layout_K_SW128_Atom<Element>{}, partition_shape_A(MMA{}, Shape<_64, _512>{})));
  using InputLayout = decltype(UMMA::tile_to_mma_shape(
      UMMA::Layout_K_SW128_Atom<Element>{}, partition_shape_B(MMA{}, Shape<_8, _512>{})));
  extern __shared__ __align__(128) unsigned char storage[];
  auto *weights = reinterpret_cast<Element *>(storage);
  auto *inputs = weights + 3 * 32768;
  auto *barriers = reinterpret_cast<uint64_t *>(inputs + 2 * 4096);
  auto *tensor_memory = reinterpret_cast<uint32_t *>(barriers + 9);
  int thread = threadIdx.x, warp = thread / 32, lane = thread & 31;
  int first_column = blockIdx.x * 64;
  TMEM::Allocator1Sm allocator;
  if (warp == 1) {
    allocator.allocate(64, tensor_memory);
    allocator.release_allocation_lock();
  }
  if (thread == 0) {
    for (int index = 0; index < 9; ++index)
      initialize_barrier(barriers[index], index >= 7 ? 4 : 1);
    cutlass::arch::fence_barrier_init();
  }
  for (int index = thread; index < 512; index += 192) {
    int slot = index / 256, vector = index % 256;
    int row = vector / 32, column = vector % 32 * 16;
    auto tile = make_tensor(make_smem_ptr(inputs + slot * 4096), InputLayout{});
    *reinterpret_cast<uint4 *>(&tile(make_coord(row, column % 32), 0, column / 32)) =
        make_uint4(0, 0, 0, 0);
  }
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  auto prefetch = [&](int tile) {
    if (lane == 0) {
      int slot = tile % 3, group = tile * 4;
      cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(barriers + slot, 32768);
#pragma unroll
      for (int sub = 0; sub < 4; ++sub)
        SM90_TMA_LOAD_2D::copy(&weight_map, barriers + slot,
                               uint64_t(Streaming ? TMA::CacheHintSm90::EVICT_FIRST
                                                  : TMA::CacheHintSm90::EVICT_NORMAL),
                               weights + slot * 32768 + sub * 8192, (group + sub) * 128,
                               first_column);
    }
  };
  if (warp == 0) {
    prefetch(0);
    prefetch(1);
    prefetch(2);
    for (int tile = 0; tile < 10; ++tile) {
      int slot = tile & 1;
      if (tile >= 2) {
        wait_barrier(barriers[5 + slot], ((tile - 2) / 2) & 1);
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        if (tile + 1 < 10)
          prefetch(tile + 1);
      }
      auto input_tile = make_tensor(make_smem_ptr(inputs + slot * 4096), InputLayout{});
      for (int index = lane; index < Rows * 32; index += 32) {
        int row = index / 32, column = index % 32 * 16;
        unsigned destination =
            __cvta_generic_to_shared(&input_tile(make_coord(row, column % 32), 0, column / 32));
        const auto *source = input + int64_t(row) * 5120 + tile * 512 + column;
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                     :
                     : "r"(destination), "l"(source)
                     : "memory");
      }
      asm volatile("cp.async.commit_group; cp.async.wait_group 0;" ::: "memory");
      __syncwarp();
      cutlass::arch::fence_view_async_shared();
      __syncwarp();
      if (lane == 0)
        cutlass::arch::ClusterBarrier::arrive(barriers + 3 + slot);
    }
  } else if (warp == 1) {
    MMA mma;
    auto partial = partition_fragment_C(mma, Shape<_64, _8>{});
    for (int tile = 0; tile < 10; ++tile) {
      int slot = tile & 1;
      wait_barrier(barriers[tile % 3], (tile / 3) & 1);
      wait_barrier(barriers[3 + slot], (tile / 2) & 1);
      if (tile >= 2)
        wait_barrier(barriers[7 + slot], ((tile - 2) / 2) & 1);
      cutlass::arch::fence_view_async_shared();
      asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
      auto weight_tile = make_tensor(make_smem_ptr(weights + tile % 3 * 32768), WeightLayout{});
      auto input_tile = make_tensor(make_smem_ptr(inputs + slot * 4096), InputLayout{});
      auto weight_fragment = mma.get_slice(0).make_fragment_A(weight_tile);
      auto input_fragment = mma.get_slice(0).make_fragment_B(input_tile);
#pragma unroll
      for (int sub = 0; sub < 4; ++sub) {
        partial.data() = *tensor_memory + slot * 32 + sub * 8;
        mma.accumulate_ = UMMA::ScaleOut::Zero;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          gemm(mma, weight_fragment(_, _, sub * 4 + k), input_fragment(_, _, sub * 4 + k), partial);
          mma.accumulate_ = UMMA::ScaleOut::One;
        }
      }
      cutlass::arch::umma_arrive(barriers + 5 + slot);
    }
  } else {
    // A warp can load only its own 32-datapath TMEM slab.
    int reader = warp & 3;
    float accumulated[Rows] = {};
    for (int tile = 0; tile < 10; ++tile) {
      int slot = tile & 1, group = tile * 4;
      float factors[4][Rows];
#pragma unroll
      for (int sub = 0; sub < 4; ++sub)
#pragma unroll
        for (int row = 0; row < Rows; ++row)
          factors[sub][row] = input_scale[row * 40 + group + sub] *
                              weight_scale[int(blockIdx.x / 2) * 40 + group + sub];
      wait_barrier(barriers[5 + slot], (tile / 2) & 1);
      asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
      uint32_t address = *tensor_memory + slot * 32 + uint32_t(reader * 32) * 65536u;
      uint32_t values[4][8];
#pragma unroll
      for (int sub = 0; sub < 4; ++sub) {
        if constexpr (Rows == 1)
          SM100_TMEM_LOAD_32dp32b1x::copy(address + sub * 8, values[sub][0]);
        else if constexpr (Rows == 4)
          SM100_TMEM_LOAD_32dp32b4x::copy(address + sub * 8, values[sub][0], values[sub][1],
                                          values[sub][2], values[sub][3]);
        else
          SM100_TMEM_LOAD_32dp32b8x::copy(address + sub * 8, values[sub][0], values[sub][1],
                                          values[sub][2], values[sub][3], values[sub][4],
                                          values[sub][5], values[sub][6], values[sub][7]);
      }
      cutlass::arch::fence_view_async_tmem_load();
      asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
      __syncwarp();
      if (lane == 0)
        cutlass::arch::ClusterBarrier::arrive(barriers + 7 + slot);
#pragma unroll
      for (int sub = 0; sub < 4; ++sub)
#pragma unroll
        for (int row = 0; row < Rows; ++row)
          accumulated[row] =
              fmaf(__uint_as_float(values[sub][row]), factors[sub][row], accumulated[row]);
    }
#pragma unroll
    for (int row = 0; row < Rows; ++row)
      if (lane < 16)
        output[int64_t(row) * 16384 + first_column + reader * 16 + lane] =
            __float2bfloat16_rn(accumulated[row]);
    if constexpr (FuseConvolution) {
      static_assert(Rows == 1);
      if (first_column < 10240 && lane < 16) {
        int sequence = 0;
        gdn_require(convolution.starts[0] == 0 && convolution.starts[convolution.batch] == 1);
        while (sequence + 1 < convolution.batch && 0 >= convolution.starts[sequence + 1])
          ++sequence;
        int first = convolution.starts[sequence], last = convolution.starts[sequence + 1];
        int source = convolution.reads[sequence], target = convolution.writes[0];
        gdn_require(first >= 0 && first <= 0 && 0 < last && last <= 1);
        gdn_require(last - first <= 5);
        gdn_require(source >= 0 && source < convolution.slots && target >= -1 &&
                    target < convolution.slots);
        if (target >= 0)
          for (int request = 0; request < convolution.batch; ++request)
            gdn_require(target != convolution.reads[request]);
        int channel = first_column + reader * 16 + lane;
        __nv_bfloat16 projected = __float2bfloat16_rn(accumulated[0]);
        float sum = 0;
#pragma unroll
        for (int tap = 0; tap < 4; ++tap) {
          int position = tap - 3;
          auto value =
              position >= first
                  ? projected
                  : convolution
                        .state[(int64_t(source) * 10240 + channel) * 3 + position - first + 3];
          sum += __bfloat162float(value) * __bfloat162float(convolution.weight[channel * 4 + tap]);
          if (target >= 0 && tap > 0)
            convolution.state[(int64_t(target) * 10240 + channel) * 3 + tap - 1] = value;
        }
        convolution.output[channel] = __float2bfloat16_rn(sum * gdn_sigmoid(sum));
      }
    }
  }
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  __syncthreads();
  if (warp == 1) {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    allocator.free(*tensor_memory, 64);
  }
}
