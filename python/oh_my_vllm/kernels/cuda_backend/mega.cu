#include "common.cuh"
#include "tile_runtime.cuh"
#include <cooperative_groups.h>
#include <tvm/ffi/container/array.h>
#include <vector>

using tvm::ffi::Array;
using tvm::ffi::Tensor;

__global__ void mega_kernel(const int64_t *descriptors, int tasks, const int *edges, int edge_count,
                            int *state, int *queue, int capacity, int64_t *control,
                            const int64_t *metadata, const int64_t *addresses, int buffers,
                            int rows) {
  auto grid = cooperative_groups::this_grid();
  if (blockIdx.x == 0 && threadIdx.x == 0)
    tile::initialize(descriptors, tasks, state, queue, capacity, control, metadata, rows, buffers,
                     edge_count);
  grid.sync();
  __shared__ int task;
  while (true) {
    if (threadIdx.x == 0) {
      task = tile::pop(queue, capacity, control, tasks);
      if (task >= 0) {
        int expected = 0;
        if (task >= tasks ||
            !tile::atomic(state[int64_t(task) * 2 + 1])
                 .compare_exchange_strong(expected, 1, cuda::memory_order_acq_rel)) {
          tile::fail(control, tile::InvalidTask);
          task = -2;
        }
      }
    }
    __syncthreads();
    if (task == -2)
      break;
    if (task == -1) {
      __nanosleep(64);
      __syncthreads();
      continue;
    }
    const int64_t *descriptor = descriptors + int64_t(task) * tile::Fields;
    tile::run_task(descriptor, addresses, metadata[1]);
    // Task bodies must finish asynchronous copies and MMA before this publication barrier.
    __syncthreads();
    if (threadIdx.x == 0)
      tile::finish(descriptor, edges, state, queue, capacity, control, tasks, task);
    __syncthreads();
  }
  grid.sync();
  if (blockIdx.x == 0 && threadIdx.x == 0 && control[tile::Error] == tile::Success)
    control[tile::CompletedEpoch] = metadata[0];
}

int64_t mega_max_blocks() {
  int device = 0, cooperative = 0, sms = 0, blocks = 0;
  check_cuda(cudaGetDevice(&device));
  check_cuda(cudaDeviceGetAttribute(&cooperative, cudaDevAttrCooperativeLaunch, device));
  TVM_FFI_ICHECK(cooperative) << "tile execution requires cooperative launch";
  check_cuda(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
  check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, mega_kernel, tile::Threads, 0));
  TVM_FFI_ICHECK(blocks > 0);
  return int64_t(sms) * blocks;
}

void mega_bind(Array<Tensor> buffers, TensorView addresses, int64_t rows) {
  TVM_FFI_ICHECK(!buffers.empty() && rows > 0 && rows <= INT32_MAX);
  tensor_contract(addresses, addresses, kDLInt, 64, {int64_t(buffers.size())});
  std::vector<int64_t> pointers;
  for (size_t index = 0; index < buffers.size(); ++index) {
    TensorView value = buffers[index];
    tensor_contract(value, addresses, kDLFloat, 32, {rows});
    disjoint(value, addresses);
    for (size_t other = 0; other < index; ++other)
      disjoint(value, buffers[other]);
    pointers.push_back(reinterpret_cast<int64_t>(value.data_ptr()));
  }
  tvm::ffi::CUDADeviceGuard guard(addresses.device().device_id);
  check_cuda(cudaMemcpy(addresses.data_ptr(), pointers.data(), pointers.size() * sizeof(int64_t),
                        cudaMemcpyHostToDevice));
}

void mega_launch(TensorView descriptors, TensorView edges, TensorView state, TensorView queue,
                 TensorView control, TensorView metadata, TensorView addresses, int64_t rows,
                 int64_t workers) {
  TVM_FFI_ICHECK(descriptors.ndim() == 2 && descriptors.size(0) > 0 &&
                 descriptors.size(0) <= INT32_MAX && rows > 0 && rows <= INT32_MAX);
  int tasks = descriptors.size(0);
  TVM_FFI_ICHECK(edges.ndim() == 1 && edges.size(0) <= INT32_MAX);
  TVM_FFI_ICHECK(queue.ndim() == 1 && queue.size(0) > 0 && queue.size(0) <= INT32_MAX);
  TVM_FFI_ICHECK(addresses.ndim() == 1 && addresses.size(0) > 0 && addresses.size(0) <= INT32_MAX);
  tensor_contract(descriptors, descriptors, kDLInt, 64, {tasks, tile::Fields});
  tensor_contract(edges, descriptors, kDLInt, 32, {edges.size(0)});
  tensor_contract(state, descriptors, kDLInt, 32, {tasks, 2});
  tensor_contract(queue, descriptors, kDLInt, 32, {queue.size(0)});
  tensor_contract(control, descriptors, kDLInt, 64, {9});
  tensor_contract(metadata, descriptors, kDLInt, 64, {2});
  tensor_contract(addresses, descriptors, kDLInt, 64, {addresses.size(0)});
  std::vector<TensorView> tensors = {descriptors, edges,    state,    queue,
                                     control,     metadata, addresses};
  for (size_t left = 0; left < tensors.size(); ++left)
    for (size_t right = 0; right < left; ++right)
      disjoint(tensors[left], tensors[right]);
  tvm::ffi::CUDADeviceGuard guard(descriptors.device().device_id);
  TVM_FFI_ICHECK(workers > 0 && workers <= mega_max_blocks());
  auto stream =
      static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, descriptors.device().device_id));
  auto task_data = static_cast<const int64_t *>(descriptors.data_ptr());
  auto edge_data = static_cast<const int *>(edges.data_ptr());
  auto state_data = static_cast<int *>(state.data_ptr());
  auto queue_data = static_cast<int *>(queue.data_ptr());
  auto control_data = static_cast<int64_t *>(control.data_ptr());
  auto step_data = static_cast<const int64_t *>(metadata.data_ptr());
  auto pointers = static_cast<const int64_t *>(addresses.data_ptr());
  int edge_count = edges.size(0), capacity = queue.size(0), buffers = addresses.size(0),
      row_count = rows;
  void *arguments[] = {&task_data, &tasks,        &edge_data, &edge_count, &state_data, &queue_data,
                       &capacity,  &control_data, &step_data, &pointers,   &buffers,    &row_count};
  check_cuda(cudaLaunchCooperativeKernel(reinterpret_cast<void *>(mega_kernel), workers,
                                         tile::Threads, arguments, 0, stream));
}
