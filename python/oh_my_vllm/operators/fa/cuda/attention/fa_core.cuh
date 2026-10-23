#pragma once

#include "fa_prefill.cuh"
#include "fa_prefill_ring.cuh"

inline int fa_attention_core_launch(TensorView query, TensorView cache, TensorView tables,
                                    TensorView lengths, TensorView sequence_rows,
                                    TensorView partial, TensorView lse, int max_queries,
                                    int first_token, cudaStream_t stream, __nv_bfloat16 *attended,
                                    int max_context) {
  int rows = query.size(0), batch = tables.size(0), blocks = cache.size(0), splits = lse.size(2);
  if (rows <= 32) {
    int split_limit = max_queries <= 5 ? (batch >= 3 ? 16 : batch == 2 ? 32 : 64) : 64;
    splits = std::min(splits, split_limit);
  }
  if (rows <= 32 && splits > 16 && tables.size(1) <= 3)
    splits = tables.size(1) == 1 ? 16 : 32;
  if (rows > 32) {
    splits = splits >= 4 && tables.size(1) >= 64 ? 4 : 1;
    if (max_queries >= 512) {
      splits = 1;
      const auto &maps = fa_ring_pipeline_tensor_maps(cache);
      int tiles = (max_queries * 6 + 127) / 128;
      auto launch = [&](auto mode, auto rounding) {
        constexpr bool ExportBFloat16 = decltype(mode)::value;
        constexpr bool RoundMax = decltype(rounding)::value;
        static std::once_flag configured;
        std::call_once(configured, [] {
          check_cuda(cudaFuncSetAttribute(fa_ring_kernel<ExportBFloat16, RoundMax>,
                                          cudaFuncAttributeMaxDynamicSharedMemorySize,
                                          FARingSharedBytes));
        });
        fa_ring_kernel<ExportBFloat16, RoundMax>
            <<<dim3(batch * tiles, 4), 480, FARingSharedBytes, stream>>>(
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
      if (max_queries >= 1024) {
        if (attended != nullptr)
          launch(std::true_type{}, std::true_type{});
        else
          launch(std::false_type{}, std::true_type{});
      } else if (attended != nullptr) {
        launch(std::true_type{}, std::false_type{});
      } else {
        launch(std::false_type{}, std::false_type{});
      }
      check_cuda(cudaGetLastError());
      return 1;
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
  } else {
    auto attend = [&](auto query_tile) {
      constexpr int QueryTile = decltype(query_tile)::value;
      constexpr int shared_bytes =
          (QueryTile * 256 + 2 * 64 * 256 + QueryTile * 64) * 2 + 8 * QueryTile * 4 + 16;
      static std::once_flag configured;
      std::call_once(configured, [] {
        check_cuda(cudaFuncSetAttribute(fa_partial_kernel<QueryTile>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
        check_cuda(cudaFuncSetAttribute(fa_partial_kernel<QueryTile>,
                                        cudaFuncAttributePreferredSharedMemoryCarveout, 100));
      });
      int tiles = (max_queries * 6 + QueryTile - 1) / QueryTile;
      fa_partial_kernel<QueryTile><<<dim3(batch * tiles, 4, splits), 128, shared_bytes, stream>>>(
          static_cast<const __nv_bfloat16 *>(query.data_ptr()),
          static_cast<const __nv_bfloat16 *>(cache.data_ptr()),
          static_cast<const int32_t *>(tables.data_ptr()),
          static_cast<const int32_t *>(lengths.data_ptr()),
          static_cast<const int32_t *>(sequence_rows.data_ptr()),
          static_cast<float *>(partial.data_ptr()), static_cast<float *>(lse.data_ptr()),
          tables.size(1), splits, first_token, tiles, max_queries, rows, blocks,
          fa_prefill_pipeline_tensor_maps(cache).key);
      check_cuda(cudaGetLastError());
    };
    if (max_queries == 1)
      attend(std::integral_constant<int, 16>{});
    else
      attend(std::integral_constant<int, 32>{});
  }
  return splits;
}
