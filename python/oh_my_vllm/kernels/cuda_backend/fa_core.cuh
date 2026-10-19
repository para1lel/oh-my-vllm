#pragma once

#include "fa_prefill.cuh"
#include "fa_prefill_cluster.cuh"
#include "fa_prefill_ring.cuh"
#include "fa_q1_warp_transpose.cuh"
#include "fa_warp_localcopy.cuh"

inline int fa_attention_core_launch(TensorView query, TensorView cache, TensorView tables,
                                    TensorView lengths, TensorView sequence_rows,
                                    TensorView partial, TensorView lse, int max_queries,
                                    int first_token, cudaStream_t stream,
                                    __nv_bfloat16 *attended = nullptr, int max_context = 0,
                                    int min_context = 0) {
  int rows = query.size(0), batch = tables.size(0), blocks = cache.size(0), splits = lse.size(2);
  if (batch == 2 && rows > max_queries && first_token == 0 && min_context >= 32768 &&
      splits >= 64 && max_queries >= 2 && max_queries <= 5) {
    splits = max_queries == 2 ? 34 : 36;
    auto launch = [&](auto query_warps) {
      constexpr int QueryWarps = decltype(query_warps)::value;
      constexpr int SharedBytes = (QueryWarps * 16 * 256 + 2 * 64 * 256) * 2;
      static std::once_flag configured;
      std::call_once(configured, [] {
        check_cuda(cudaFuncSetAttribute(fa_warp_localcopy_kernel<QueryWarps>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, SharedBytes));
      });
      fa_warp_localcopy_kernel<QueryWarps><<<dim3(batch, 4, splits), 128, SharedBytes, stream>>>(
          static_cast<const __nv_bfloat16 *>(query.data_ptr()),
          static_cast<const __nv_bfloat16 *>(cache.data_ptr()),
          static_cast<const int32_t *>(tables.data_ptr()),
          static_cast<const int32_t *>(lengths.data_ptr()),
          static_cast<const int32_t *>(sequence_rows.data_ptr()),
          static_cast<float *>(partial.data_ptr()), static_cast<float *>(lse.data_ptr()),
          int(tables.size(1)), splits, first_token, max_queries, rows, blocks);
    };
    if (max_queries == 2)
      launch(std::integral_constant<int, 1>{});
    else
      launch(std::integral_constant<int, 2>{});
    check_cuda(cudaGetLastError());
    return splits;
  }
  if (rows <= 32) {
    int split_limit = max_queries == 1 && ((batch >= 4 && max_context >= 32768) ||
                                           (batch == 2 && first_token == 0 && splits >= 128))
                          ? 128
                          : 64;
    splits = std::min(splits, split_limit);
  }
  if (rows <= 32 && splits > 16 && tables.size(1) <= 3)
    splits = tables.size(1) == 1 ? 16 : 32;
  else if (rows <= 32 && batch > 1 && max_queries > 1)
    splits = std::min(splits, 16);
  if (rows > 32) {
    splits = splits >= 4 && tables.size(1) >= 64 ? 4 : 1;
    if (max_queries >= 512 && max_queries < 1024 && batch == 1) {
      splits = 1;
      const auto &maps = fa_ring_pipeline_tensor_maps(cache);
      int tiles = (max_queries * 6 + 127) / 128;
      auto launch = [&](auto mode) {
        constexpr bool ExportBFloat16 = decltype(mode)::value;
        static std::once_flag configured;
        std::call_once(configured, [] {
          check_cuda(cudaFuncSetAttribute(fa_ring_kernel<ExportBFloat16>,
                                          cudaFuncAttributeMaxDynamicSharedMemorySize,
                                          FARingSharedBytes));
        });
        fa_ring_kernel<ExportBFloat16><<<dim3(batch * tiles, 4), 480, FARingSharedBytes, stream>>>(
            static_cast<const __nv_bfloat16 *>(query.data_ptr()),
            static_cast<const __nv_bfloat16 *>(cache.data_ptr()),
            static_cast<const int32_t *>(tables.data_ptr()),
            static_cast<const int32_t *>(lengths.data_ptr()),
            static_cast<const int32_t *>(sequence_rows.data_ptr()),
            static_cast<float *>(partial.data_ptr()), static_cast<float *>(lse.data_ptr()),
            int(tables.size(1)), tiles, max_queries, rows, blocks, first_token, maps.key,
            maps.value, fa_ring_pipeline_l2_bytes(query.device().device_id), maps.key16,
            maps.value16, attended);
      };
      if (attended != nullptr)
        launch(std::true_type{});
      else
        launch(std::false_type{});
      check_cuda(cudaGetLastError());
      return 1;
    }
    if (max_queries >= 1024) {
      splits = 1;
      const auto &maps = fa_prefill_cluster_tensor_maps(cache);
      int tiles = (max_queries * 6 + 255) / 256;
      auto launch = [&](auto mode, auto schedule) {
        constexpr bool ExportBFloat16 = decltype(mode)::value;
        constexpr bool Decouple = decltype(schedule)::value;
        static std::once_flag configured;
        std::call_once(configured, [] {
          check_cuda(cudaFuncSetAttribute(fa_prefill_cluster_kernel<ExportBFloat16, Decouple>,
                                          cudaFuncAttributeMaxDynamicSharedMemorySize,
                                          FAPrefillClusterSharedBytes));
          check_cuda(cudaFuncSetAttribute(fa_prefill_cluster_kernel<ExportBFloat16, Decouple>,
                                          cudaFuncAttributePreferredSharedMemoryCarveout, 100));
        });
        cudaLaunchConfig_t config{};
        config.gridDim = dim3(2 * batch * tiles, 4);
        config.blockDim = dim3(Decouple ? 480 : 448);
        config.dynamicSmemBytes = FAPrefillClusterSharedBytes;
        config.stream = stream;
        cudaLaunchAttribute attribute{};
        attribute.id = cudaLaunchAttributeClusterDimension;
        attribute.val.clusterDim = {2, 1, 1};
        config.attrs = &attribute;
        config.numAttrs = 1;
        check_cuda(cudaLaunchKernelEx(&config, fa_prefill_cluster_kernel<ExportBFloat16, Decouple>,
                                      static_cast<const __nv_bfloat16 *>(query.data_ptr()),
                                      static_cast<const __nv_bfloat16 *>(cache.data_ptr()),
                                      static_cast<const int32_t *>(tables.data_ptr()),
                                      static_cast<const int32_t *>(lengths.data_ptr()),
                                      static_cast<const int32_t *>(sequence_rows.data_ptr()),
                                      static_cast<float *>(partial.data_ptr()),
                                      static_cast<float *>(lse.data_ptr()), int(tables.size(1)),
                                      tiles, max_queries, rows, blocks, first_token, maps.key,
                                      maps.value, maps.key16, maps.value16, attended));
      };
      auto export_result = [&](auto schedule) {
        if (attended != nullptr)
          launch(std::true_type{}, schedule);
        else
          launch(std::false_type{}, schedule);
      };
      if (max_queries >= 2048 || max_context >= 12544)
        export_result(std::true_type{});
      else
        export_result(std::false_type{});
      check_cuda(cudaGetLastError());
      return splits;
    }
    const auto &maps = fa_prefill_pipeline_tensor_maps(cache);
    int tiles = (max_queries * 6 + 127) / 128;
    auto launch = [&](auto mode) {
      constexpr bool ExportBFloat16 = decltype(mode)::value;
      static std::once_flag configured;
      std::call_once(configured, [] {
        check_cuda(cudaFuncSetAttribute(fa_prefill_kernel<ExportBFloat16>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        FAPrefillSharedBytes));
      });
      fa_prefill_kernel<ExportBFloat16>
          <<<dim3(batch * tiles, 4, splits), 320, FAPrefillSharedBytes, stream>>>(
              static_cast<const __nv_bfloat16 *>(query.data_ptr()),
              static_cast<const __nv_bfloat16 *>(cache.data_ptr()),
              static_cast<const int32_t *>(tables.data_ptr()),
              static_cast<const int32_t *>(lengths.data_ptr()),
              static_cast<const int32_t *>(sequence_rows.data_ptr()),
              static_cast<float *>(partial.data_ptr()), static_cast<float *>(lse.data_ptr()),
              tables.size(1), tiles, max_queries, rows, blocks, first_token, maps.key, maps.value,
              fa_prefill_pipeline_l2_bytes(query.device().device_id), attended);
    };
    if (attended != nullptr && splits == 1)
      launch(std::true_type{});
    else
      launch(std::false_type{});
    check_cuda(cudaGetLastError());
  } else if (max_queries == 1 && rows <= 32 && splits <= 128) {
    static std::once_flag warp_q1_configured;
    std::call_once(warp_q1_configured, [] {
      check_cuda(cudaFuncSetAttribute(fa_q1_warp_partial_kernel,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      FAWarpTransposeSharedBytes));
    });
    fa_q1_warp_partial_kernel<<<dim3(batch, 4, splits), 128, FAWarpTransposeSharedBytes, stream>>>(
        static_cast<const __nv_bfloat16 *>(query.data_ptr()),
        static_cast<const __nv_bfloat16 *>(cache.data_ptr()),
        static_cast<const int32_t *>(tables.data_ptr()),
        static_cast<const int32_t *>(lengths.data_ptr()),
        static_cast<const int32_t *>(sequence_rows.data_ptr()),
        static_cast<float *>(partial.data_ptr()), static_cast<float *>(lse.data_ptr()),
        tables.size(1), splits, first_token, rows, blocks);
    check_cuda(cudaGetLastError());
  } else {
    auto attend = [&](auto query_tile) {
      constexpr int QueryTile = decltype(query_tile)::value;
      constexpr int shared_bytes =
          (QueryTile * 256 + 2 * 64 * 256 + QueryTile * 64) * 2 + 8 * QueryTile * 4;
      static std::once_flag configured;
      std::call_once(configured, [] {
        check_cuda(cudaFuncSetAttribute(fa_partial_kernel<QueryTile>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
      });
      int tiles = (max_queries * 6 + QueryTile - 1) / QueryTile;
      fa_partial_kernel<QueryTile><<<dim3(batch * tiles, 4, splits), 128, shared_bytes, stream>>>(
          static_cast<const __nv_bfloat16 *>(query.data_ptr()),
          static_cast<const __nv_bfloat16 *>(cache.data_ptr()), tables.data_ptr(),
          lengths.data_ptr(), sequence_rows.data_ptr(), static_cast<float *>(partial.data_ptr()),
          static_cast<float *>(lse.data_ptr()), tables.size(1), splits, first_token, false, false,
          false, tiles, max_queries, rows, blocks);
      check_cuda(cudaGetLastError());
    };
    if (max_queries * 6 <= 16)
      attend(std::integral_constant<int, 16>{});
    else
      attend(std::integral_constant<int, 32>{});
  }
  return splits;
}
