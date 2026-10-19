#include "../../python/oh_my_vllm/kernels/cuda_backend/gdn_value_pair.cuh"
#include "../../python/oh_my_vllm/kernels/cuda_backend/gdn_wy_launch.cuh"

__global__ void gdn_core_pack_kernel(const __nv_bfloat16 *q, const __nv_bfloat16 *k,
                                     const __nv_bfloat16 *v, const __nv_bfloat16 *a,
                                     const __nv_bfloat16 *b, const float *a_log,
                                     const float *dt_bias, __nv_bfloat16 *qkv, __nv_bfloat16 *ba,
                                     float *gates, bool verify) {
  int row = blockIdx.x, group = blockIdx.y * 4 + threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  float values[4], square = 0.f;
#pragma unroll
  for (int element = 0; element < 4; ++element) {
    int column = group * 128 + lane * 4 + element;
    const auto *input = group < 16 ? q : group < 32 ? k : v;
    int width = group < 32 ? 2048 : 6144;
    int offset = group < 16 ? column : group < 32 ? column - 2048 : column - 4096;
    values[element] = __bfloat162float(input[int64_t(row) * width + offset]);
    square += values[element] * values[element];
  }
  float inverse = !verify && group < 32 ? rsqrtf(warp_sum(square) + 1e-6f) : 1.f;
#pragma unroll
  for (int element = 0; element < 4; ++element)
    qkv[int64_t(row) * 10240 + group * 128 + lane * 4 + element] =
        __float2bfloat16_rn(values[element] * inverse);
  if (blockIdx.y == 0 && threadIdx.x < 48) {
    int head = threadIdx.x;
    auto projected_b = b[int64_t(row) * 48 + head];
    auto projected_a = a[int64_t(row) * 48 + head];
    ba[int64_t(row) * 96 + head] = projected_b;
    ba[int64_t(row) * 96 + 48 + head] = projected_a;
    if (!verify) {
      float activation = __bfloat162float(projected_a) + dt_bias[head];
      float softplus = activation > 20.f ? activation : log1pf(expf(activation));
      gates[int64_t(row) * 96 + head] = gdn_sigmoid(__bfloat162float(projected_b));
      gates[int64_t(row) * 96 + 48 + head] = expf(-expf(a_log[head]) * softplus);
    }
  }
}

void gdn_core(TensorView q, TensorView k, TensorView v, TensorView a, TensorView b,
              TensorView a_log, TensorView dt_bias, TensorView pool, TensorView starts,
              TensorView reads, TensorView writes, TensorView qkv, TensorView ba, TensorView gates,
              TensorView output, TensorView chunk_metadata, TensorView chunk_transforms,
              TensorView chunk_prefixes, TensorView chunk_snapshots, TensorView chunk_coefficients,
              bool verify, int max_queries) {
  TVM_FFI_ICHECK(q.ndim() == 3 && reads.ndim() == 1 && pool.ndim() == 4);
  int rows = q.size(0), batch = reads.size(0), slots = pool.size(0);
  TVM_FFI_ICHECK(rows > 0 && rows <= 32768 && batch > 0 && batch <= rows);
  TVM_FFI_ICHECK(slots > batch && slots <= 65536);
  TVM_FFI_ICHECK(max_queries > 0 && max_queries <= rows && (!verify || max_queries <= 5));
  for (auto tensor : {q, k})
    tensor_contract(tensor, q, kDLBfloat, 16, {rows, 16, 128});
  for (auto tensor : {a, b})
    tensor_contract(tensor, q, kDLBfloat, 16, {rows, 48});
  for (auto tensor : {v, output})
    tensor_contract(tensor, q, kDLBfloat, 16, {rows, 48, 128});
  for (auto tensor : {a_log, dt_bias})
    tensor_contract(tensor, q, kDLFloat, 32, {48});
  tensor_contract(pool, q, kDLBfloat, 16, {slots, 48, 128, 128});
  tensor_contract(starts, q, kDLInt, 32, {batch + 1});
  tensor_contract(reads, q, kDLInt, 32, {batch});
  tensor_contract(writes, q, kDLInt, 32, {rows});
  tensor_contract(qkv, q, kDLBfloat, 16, {rows, 10240});
  tensor_contract(ba, q, kDLBfloat, 16, {rows, 96});
  tensor_contract(gates, q, kDLFloat, 32, {verify ? 0 : rows, 96});
  bool chunked = gdn_wy::selected(rows, batch, max_queries, verify);
  int capacity = chunked ? gdn_wy::chunk_capacity(max_queries) : 0;
  gdn_wy::workspace_contract(q, chunk_metadata, chunk_transforms, chunk_prefixes, chunk_snapshots,
                             chunk_coefficients, batch * capacity);
  auto tensors = {q,
                  k,
                  v,
                  a,
                  b,
                  a_log,
                  dt_bias,
                  pool,
                  starts,
                  reads,
                  writes,
                  qkv,
                  ba,
                  gates,
                  output,
                  chunk_metadata,
                  chunk_transforms,
                  chunk_prefixes,
                  chunk_snapshots,
                  chunk_coefficients};
  int left = 0;
  for (auto first : tensors) {
    int right = 0;
    for (auto second : tensors)
      if (right++ > left)
        disjoint(first, second);
    ++left;
  }
  tvm::ffi::CUDADeviceGuard guard(q.device().device_id);
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, q.device().device_id));
  auto *packed = static_cast<__nv_bfloat16 *>(qkv.data_ptr());
  auto *projected = static_cast<__nv_bfloat16 *>(ba.data_ptr());
  auto *coefficients = static_cast<float *>(gates.data_ptr());
  auto *state = static_cast<__nv_bfloat16 *>(pool.data_ptr());
  auto *result = static_cast<__nv_bfloat16 *>(output.data_ptr());
  auto *sequence_starts = static_cast<const int *>(starts.data_ptr());
  auto *read_slots = static_cast<const int *>(reads.data_ptr());
  auto *write_slots = static_cast<const int *>(writes.data_ptr());
  auto *logarithms = static_cast<const float *>(a_log.data_ptr());
  auto *biases = static_cast<const float *>(dt_bias.data_ptr());
  gdn_core_pack_kernel<<<dim3(rows, 20), 128, 0, stream>>>(
      static_cast<const __nv_bfloat16 *>(q.data_ptr()),
      static_cast<const __nv_bfloat16 *>(k.data_ptr()),
      static_cast<const __nv_bfloat16 *>(v.data_ptr()),
      static_cast<const __nv_bfloat16 *>(a.data_ptr()),
      static_cast<const __nv_bfloat16 *>(b.data_ptr()), logarithms, biases, packed, projected,
      coefficients, verify);
  check_cuda(cudaGetLastError());
  if (chunked) {
    gdn_wy::native_launch(packed, coefficients, sequence_starts, read_slots, write_slots, state,
                          result, static_cast<int *>(chunk_metadata.data_ptr()),
                          static_cast<__nv_bfloat16 *>(chunk_transforms.data_ptr()),
                          static_cast<float *>(chunk_prefixes.data_ptr()),
                          static_cast<__nv_bfloat16 *>(chunk_snapshots.data_ptr()),
                          static_cast<__nv_bfloat16 *>(chunk_coefficients.data_ptr()), rows, slots,
                          batch, capacity, stream);
  } else if (verify) {
    auto scan = [&](auto tile, auto threads) {
      constexpr int ValueTile = decltype(tile)::value, Threads = decltype(threads)::value;
      launch_kernel(gdn_scan_quant_kernel<false, ValueTile, Threads, false, false, false, true>,
                    dim3(batch, 48, 128 / ValueTile), dim3(Threads), 0, stream, packed, nullptr,
                    projected, nullptr, nullptr, logarithms, biases, nullptr, state,
                    sequence_starts, read_slots, write_slots, nullptr, nullptr, rows, slots, true,
                    result, coefficients);
    };
    if (batch == 4 && rows == 20)
      scan(std::integral_constant<int, 32>{}, std::integral_constant<int, 128>{});
    else
      scan(std::integral_constant<int, 16>{}, std::integral_constant<int, 256>{});
  } else {
    auto paired = [&](auto tile, auto kernel) {
      constexpr int ValueTile = decltype(tile)::value;
      launch_kernel(kernel, dim3(batch, 48, 128 / ValueTile), dim3(128),
                    2 * GDNPrefetchSlotBytes<ValueTile>, stream, packed, state, sequence_starts,
                    read_slots, write_slots, result, coefficients, rows, slots);
    };
    if (gdn_prefill_two_pairs(rows, max_queries, q.device().device_id))
      paired(std::integral_constant<int, 32>{}, gdn_value_two_pair_kernel<32, 128>);
    else
      paired(std::integral_constant<int, 16>{}, gdn_value_pair_kernel<16, 128>);
  }
  check_cuda(cudaGetLastError());
}
