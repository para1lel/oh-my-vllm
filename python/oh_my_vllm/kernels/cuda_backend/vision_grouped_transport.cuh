/***************************************************************************************************
 * Copyright (c) 2024 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include "common.cuh"
#include <cute/arch/copy_sm90_tma.hpp>

inline void vision_grouped_map(cute::TmaDescriptor &descriptor, const void *pointer, int tokens,
                               int groups, int tile_tokens) {
  cute::array<uint64_t, 5> dimensions{16, uint64_t(tokens), 5, uint64_t(16) * groups, 1};
  cute::array<uint64_t, 5> strides{2, 256, 32, uint64_t(tokens) * 256, 0};
  cute::array<uint32_t, 5> box{16, uint32_t(tile_tokens), 5, 1, 1};
  cute::array<uint32_t, 5> element_strides{1, 1, 1, 1, 1};
  CUresult status = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
      &descriptor, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, const_cast<void *>(pointer),
      dimensions.data(), strides.data() + 1, box.data(), element_strides.data(),
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_32B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK(status == CUDA_SUCCESS)
      << "grouped Vision tensor map encoding failed: " << int(status);
  if (!cute::detail::tma_gmem_prefix_is_contiguous(dimensions, strides, 16))
    reinterpret_cast<uint64_t *>(&descriptor)[1] &= ~(uint64_t(1) << 21);
}

template <class Base> struct VisionGroupedMainloop : Base {
  using Element = typename Base::Element;
  using Params = typename Base::Params;
  using Arguments = typename Base::Arguments;
  using TensorStorage = typename Base::TensorStorage;
  using PipelineQ = typename Base::PipelineQ;
  using PipelineKV = typename Base::PipelineKV;
  static_assert(Base::StageCountQ == 2 && Base::StageCountKV == 3);
  static_assert(Base::TransactionBytesLoadQ == 20480);
  static_assert(Base::TransactionBytesLoadK == 10240 && Base::TransactionBytesLoadV == 10240);

  template <class ProblemShape>
  static Params to_underlying_arguments(const ProblemShape &problem, const Arguments &arguments,
                                        void *workspace) {
    using namespace cute;
    Params params{};
    int tokens = get<0>(problem), groups = get<3, 1>(problem);
    vision_grouped_map(params.load.tma_load_q.tma_desc_, arguments.load.ptr_Q, tokens, groups, 128);
    vision_grouped_map(params.load.tma_load_k.tma_desc_, arguments.load.ptr_K, tokens, groups, 64);
    vision_grouped_map(params.load.tma_load_v.tma_desc_, arguments.load.ptr_V, tokens, groups, 64);
    float scale_softmax = arguments.scale_softmax;
    if (scale_softmax == 0.f)
      scale_softmax = 1.f / (float)std::sqrt(get<2>(problem));
    float log2_e = static_cast<float>(std::log2(std::exp(1.0)));
    params.scale_softmax = arguments.scale_q * arguments.scale_k * scale_softmax;
    params.scale_softmax_log2 = arguments.scale_q * arguments.scale_k * log2_e * scale_softmax;
    params.scale_output = arguments.scale_v * arguments.inv_scale_o;
    return params;
  }

  template <class BlkCoord, class ProblemShape, class ParamsProblemShape>
  CUTLASS_DEVICE void load(const BlkCoord &block, const ProblemShape &problem, const Params &params,
                           const ParamsProblemShape &params_problem, TensorStorage &storage,
                           PipelineQ &pipeline_q, typename PipelineQ::PipelineState &q_state,
                           PipelineKV &pipeline_kv, typename PipelineKV::PipelineState &kv_state) {
    using namespace cute;
    static_assert(
        !cutlass::fmha::collective::is_variable_length_v<tuple_element_t<0, ParamsProblemShape>>);
    int count = typename Base::Mask{}.get_trip_count(block, typename Base::TileShape{}, problem);
    int head = int(get<2, 1>(block)) * 16 + int(get<2, 0>(block));
    int q0 = int(get<0>(block)) * 256;
    uint32_t elected = elect_one_sync();

    pipeline_q.producer_acquire(q_state);
    if (elected)
      SM90_TMA_LOAD_4D::copy(params.load.tma_load_q.get_tma_descriptor(),
                             pipeline_q.producer_get_barrier(q_state),
                             uint64_t(TMA::CacheHintSm90::EVICT_NORMAL),
                             storage.smem_q.data() + q_state.index() * 128 * 80, 0, q0, 0, head);
    ++q_state;

    pipeline_kv.producer_acquire(kv_state);
    if (elected)
      SM90_TMA_LOAD_4D::copy(params.load.tma_load_k.get_tma_descriptor(),
                             pipeline_kv.producer_get_barrier(kv_state),
                             uint64_t(TMA::CacheHintSm90::EVICT_NORMAL),
                             storage.smem_k.data() + kv_state.index() * 64 * 80, 0, 0, 0, head);
    ++kv_state;

    pipeline_q.producer_acquire(q_state);
    if (elected)
      SM90_TMA_LOAD_4D::copy(
          params.load.tma_load_q.get_tma_descriptor(), pipeline_q.producer_get_barrier(q_state),
          uint64_t(TMA::CacheHintSm90::EVICT_NORMAL),
          storage.smem_q.data() + q_state.index() * 128 * 80, 0, q0 + 128, 0, head);
    ++q_state;

    pipeline_kv.producer_acquire(kv_state);
    if (elected)
      SM90_TMA_LOAD_4D::copy(params.load.tma_load_v.get_tma_descriptor(),
                             pipeline_kv.producer_get_barrier(kv_state),
                             uint64_t(TMA::CacheHintSm90::EVICT_NORMAL),
                             storage.smem_v.data() + kv_state.index() * 64 * 80, 0, 0, 0, head);
    ++kv_state;

    for (int tile = 1; tile < count; ++tile) {
      pipeline_kv.producer_acquire(kv_state);
      if (elected)
        SM90_TMA_LOAD_4D::copy(
            params.load.tma_load_k.get_tma_descriptor(), pipeline_kv.producer_get_barrier(kv_state),
            uint64_t(TMA::CacheHintSm90::EVICT_NORMAL),
            storage.smem_k.data() + kv_state.index() * 64 * 80, 0, tile * 64, 0, head);
      ++kv_state;

      pipeline_kv.producer_acquire(kv_state);
      if (elected)
        SM90_TMA_LOAD_4D::copy(
            params.load.tma_load_v.get_tma_descriptor(), pipeline_kv.producer_get_barrier(kv_state),
            uint64_t(TMA::CacheHintSm90::EVICT_NORMAL),
            storage.smem_v.data() + kv_state.index() * 64 * 80, 0, tile * 64, 0, head);
      ++kv_state;
    }
  }
};

template <class Base> struct VisionGroupedEpilogue : Base {
  using Params = typename Base::Params;
  using Arguments = typename Base::Arguments;
  using TensorStorage = typename Base::TensorStorage;
  using Pipeline = typename Base::Pipeline;
  using Base::Base;

  template <class ProblemShape>
  static Params to_underlying_arguments(const ProblemShape &problem, const Arguments &arguments,
                                        void *workspace = nullptr) {
    using namespace cute;
    Params params{};
    vision_grouped_map(params.tma_store_o.tma_desc_, arguments.ptr_O, get<0>(problem),
                       get<3, 1>(problem), 8);
    params.ptr_LSE = arguments.ptr_LSE;
    params.dLSE = arguments.dLSE;
    return params;
  }

  template <class BlkCoord, class ProblemShape, class ParamsProblemShape>
  CUTLASS_DEVICE void store(const BlkCoord &block, const ProblemShape &problem,
                            const Params &params, const ParamsProblemShape &params_problem,
                            TensorStorage &storage, Pipeline &pipeline,
                            typename Pipeline::PipelineState &consumer_state) {
    using namespace cute;
    static_assert(
        !cutlass::fmha::collective::is_variable_length_v<tuple_element_t<0, ParamsProblemShape>>);
    int head = int(get<2, 1>(block)) * 16 + int(get<2, 0>(block));
    int q0 = int(get<0>(block)) * 256;
    uint32_t elected = elect_one_sync();
    auto release_state = consumer_state;

    pipeline.consumer_wait(consumer_state);
    ++consumer_state;
    if (elected) {
#pragma unroll
      for (int tile = 0; tile < 16; ++tile)
        SM90_TMA_STORE_4D::copy(params.tma_store_o.get_tma_descriptor(),
                                storage.smem_o.data() + tile * 8 * 80, 0, q0 + tile * 8, 0, head);
    }
    tma_store_arrive();

    pipeline.consumer_wait(consumer_state);
    ++consumer_state;
    if (elected) {
#pragma unroll
      for (int tile = 0; tile < 16; ++tile)
        SM90_TMA_STORE_4D::copy(params.tma_store_o.get_tma_descriptor(),
                                storage.smem_o.data() + 128 * 80 + tile * 8 * 80, 0,
                                q0 + 128 + tile * 8, 0, head);
    }
    tma_store_arrive();

    tma_store_wait<1>();
    pipeline.consumer_release(release_state);
    ++release_state;
    tma_store_wait<0>();
    pipeline.consumer_release(release_state);
    ++release_state;
  }
};

inline void vision_packed_output_map(cute::TmaDescriptor &descriptor, const void *pointer,
                                     int tokens, int groups) {
  cute::array<uint64_t, 5> dimensions{16, uint64_t(tokens), 4, 16, uint64_t(groups)};
  cute::array<uint64_t, 5> strides{2, 2304, 32, 144, uint64_t(tokens) * 2304};
  cute::array<uint32_t, 5> box{16, 8, 4, 1, 1};
  cute::array<uint32_t, 5> elements{1, 1, 1, 1, 1};
  auto status = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
      &descriptor, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 5, const_cast<void *>(pointer),
      dimensions.data(), strides.data() + 1, box.data(), elements.data(),
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_32B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK(status == CUDA_SUCCESS) << int(status);
  if (!cute::detail::tma_gmem_prefix_is_contiguous(dimensions, strides, 16))
    reinterpret_cast<uint64_t *>(&descriptor)[1] &= ~(uint64_t(1) << 21);
}

template <class Base> struct VisionPackedEpilogue : Base {
  struct Params : Base::Params {
    typename Base::ElementOut *output = nullptr;
  };

  using Arguments = typename Base::Arguments;
  using TensorStorage = typename Base::TensorStorage;
  using Pipeline = typename Base::Pipeline;
  using Base::Base;

  template <class ProblemShape>
  static Params to_underlying_arguments(const ProblemShape &problem, const Arguments &arguments,
                                        void *workspace = nullptr) {
    Params params{};
    vision_packed_output_map(params.tma_store_o.tma_desc_, arguments.ptr_O, cute::get<0>(problem),
                             cute::get<3, 1>(problem));
    params.ptr_LSE = arguments.ptr_LSE;
    params.dLSE = arguments.dLSE;
    params.output = arguments.ptr_O;
    return params;
  }

  template <class BlkCoord, class ProblemShape, class ParamsProblemShape>
  CUTLASS_DEVICE void store(const BlkCoord &block, const ProblemShape &problem,
                            const Params &params, const ParamsProblemShape &params_problem,
                            TensorStorage &storage, Pipeline &pipeline,
                            typename Pipeline::PipelineState &consumer_state) {
    using namespace cute;
    static_assert(
        !cutlass::fmha::collective::is_variable_length_v<tuple_element_t<0, ParamsProblemShape>>);
    int head = int(get<2, 0>(block)), frame = int(get<2, 1>(block));
    int q0 = int(get<0>(block)) * 256, tokens = int(get<0>(problem));
    int lane = threadIdx.x % 32;
    uint32_t elected = elect_one_sync();
    auto release_state = consumer_state;
    auto sO = make_tensor(make_smem_ptr(storage.smem_o.data()), typename Base::SmemLayoutO{});
#pragma unroll
    for (int stage = 0; stage < 2; ++stage) {
      pipeline.consumer_wait(consumer_state);
      ++consumer_state;
      for (int row = lane; row < 128; row += 32) {
        int query = q0 + stage * 128 + row;
        if (query < tokens) {
          uint4 value = *reinterpret_cast<const uint4 *>(&sO(row, 64, stage));
          auto *destination =
              params.output + (int64_t(frame) * tokens + query) * 1152 + head * 72 + 64;
          *reinterpret_cast<uint4 *>(destination) = value;
        }
      }
      if (elected) {
#pragma unroll
        for (int tile = 0; tile < 16; ++tile)
          SM90_TMA_STORE_5D::copy(params.tma_store_o.get_tma_descriptor(),
                                  storage.smem_o.data() + stage * 128 * 80 + tile * 8 * 80, 0,
                                  q0 + stage * 128 + tile * 8, 0, head, frame);
      }
      tma_store_arrive();
    }
    tma_store_wait<1>();
    pipeline.consumer_release(release_state);
    ++release_state;
    tma_store_wait<0>();
    pipeline.consumer_release(release_state);
    ++release_state;
  }
};
