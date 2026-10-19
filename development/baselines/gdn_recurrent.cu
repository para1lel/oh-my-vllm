#include "../../python/oh_my_vllm/kernels/cuda_backend/gdn_recurrent_core.cuh"

void gdn_recurrent_core(TensorView q, TensorView k, TensorView v, TensorView a, TensorView b,
                        TensorView a_log, TensorView dt_bias, TensorView starts, TensorView reads,
                        TensorView writes, TensorView pool, TensorView output, int max_queries,
                        int value_tile) {
  TVM_FFI_ICHECK(q.ndim() == 3 && reads.ndim() == 1 && pool.ndim() == 4);
  int rows = q.size(0), batch = reads.size(0), slots = pool.size(0);
  TVM_FFI_ICHECK(rows > 0 && rows <= 32768 && batch > 0 && batch <= rows);
  TVM_FFI_ICHECK(slots > batch && slots <= 65536 && max_queries > 0 && max_queries <= 5);
  for (auto tensor : {q, k})
    tensor_contract(tensor, q, kDLBfloat, 16, {rows, 16, 128});
  for (auto tensor : {v, output})
    tensor_contract(tensor, q, kDLBfloat, 16, {rows, 48, 128});
  for (auto tensor : {a, b})
    tensor_contract(tensor, q, kDLBfloat, 16, {rows, 48});
  for (auto tensor : {a_log, dt_bias})
    tensor_contract(tensor, q, kDLFloat, 32, {48});
  tensor_contract(starts, q, kDLInt, 32, {batch + 1});
  tensor_contract(reads, q, kDLInt, 32, {batch});
  tensor_contract(writes, q, kDLInt, 32, {rows});
  tensor_contract(pool, q, kDLBfloat, 16, {slots, 48, 128, 128});
  auto tensors = {q, k, v, a, b, a_log, dt_bias, starts, reads, writes, pool, output};
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
  gdn_recurrent::launch_validated_metadata(
      static_cast<const __nv_bfloat16 *>(q.data_ptr()),
      static_cast<const __nv_bfloat16 *>(k.data_ptr()),
      static_cast<const __nv_bfloat16 *>(v.data_ptr()),
      static_cast<const __nv_bfloat16 *>(a.data_ptr()),
      static_cast<const __nv_bfloat16 *>(b.data_ptr()),
      static_cast<const float *>(a_log.data_ptr()), static_cast<const float *>(dt_bias.data_ptr()),
      static_cast<const int *>(starts.data_ptr()), static_cast<const int *>(reads.data_ptr()),
      static_cast<const int *>(writes.data_ptr()), static_cast<__nv_bfloat16 *>(pool.data_ptr()),
      static_cast<__nv_bfloat16 *>(output.data_ptr()), rows, batch, slots, max_queries, value_tile,
      stream);
}
