#pragma once

#include <cuda/atomic>
#include <cstdint>

namespace tile {
constexpr int Threads = 256;
constexpr int Fields = 10;

enum Control { CompletedEpoch, SubmittedEpoch, Error, Finished, Head, Tail, Size, Lock, Running };

enum Failure { Success, QueueFull, StaleEpoch, InvalidTask, InvalidMetadata, Incomplete };

__device__ inline auto atomic(int64_t &value) {
  return cuda::atomic_ref<int64_t, cuda::thread_scope_device>(value);
}

__device__ inline auto atomic(int &value) {
  return cuda::atomic_ref<int, cuda::thread_scope_device>(value);
}

__device__ inline void fail(int64_t *control, int code) {
  int64_t expected = 0;
  atomic(control[Error]).compare_exchange_strong(expected, code, cuda::memory_order_acq_rel);
}

__device__ inline void lock(int64_t *control) {
  int64_t expected = 0;
  while (!atomic(control[Lock]).compare_exchange_weak(expected, 1, cuda::memory_order_acquire)) {
    expected = 0;
    __nanosleep(64);
  }
}

__device__ inline void unlock(int64_t *control) {
  atomic(control[Lock]).store(0, cuda::memory_order_release);
}

__device__ inline bool push(int *queue, int capacity, int64_t *control, int task) {
  lock(control);
  bool success = atomic(control[Error]).load(cuda::memory_order_acquire) == Success;
  if (success && control[Size] == capacity) {
    fail(control, QueueFull);
    success = false;
  }
  if (success) {
    queue[control[Tail]] = task;
    control[Tail] = (control[Tail] + 1) % capacity;
    ++control[Size];
  }
  unlock(control);
  return success;
}

__device__ inline int pop(int *queue, int capacity, int64_t *control, int tasks) {
  lock(control);
  int task = -1;
  if (atomic(control[Error]).load(cuda::memory_order_acquire) != Success ||
      control[Finished] == tasks) {
    task = -2;
  } else if (control[Size]) {
    task = queue[control[Head]];
    control[Head] = (control[Head] + 1) % capacity;
    --control[Size];
    ++control[Running];
  } else if (control[Running] == 0) {
    fail(control, Incomplete);
    task = -2;
  }
  unlock(control);
  return task;
}

__device__ inline bool valid_task(const int64_t *task, int rows, int buffers, int edges,
                                  int tasks) {
  return task[0] >= 0 && task[0] <= 2 && task[1] >= 0 && task[1] < buffers &&
         (task[0] != 2 || (task[2] >= 0 && task[2] < buffers)) && task[3] >= 0 &&
         task[3] < buffers && task[4] >= 0 && task[4] <= rows && task[5] > 0 &&
         task[5] <= rows - task[4] && task[6] >= 0 && task[6] <= edges && task[7] >= 0 &&
         task[7] <= edges - task[6] && task[9] >= 0 && task[9] < tasks &&
         isfinite(__int_as_float(static_cast<int>(task[8])));
}

__device__ inline void initialize(const int64_t *descriptors, int tasks, int *state, int *queue,
                                  int capacity, int64_t *control, const int64_t *metadata, int rows,
                                  int buffers, int edges) {
  // A failed submission cannot make its tensor state safe by resetting the queue.
  if (control[Error] != Success)
    return;
  control[SubmittedEpoch] = metadata[0];
  for (int field = Error; field <= Running; ++field)
    control[field] = 0;
  if (metadata[0] <= control[CompletedEpoch] || metadata[0] <= 0) {
    control[Error] = StaleEpoch;
    return;
  }
  if (metadata[1] < 0 || metadata[1] > rows) {
    control[Error] = InvalidMetadata;
    return;
  }
  for (int index = 0; index < tasks; ++index) {
    const int64_t *task = descriptors + int64_t(index) * Fields;
    if (!valid_task(task, rows, buffers, edges, tasks)) {
      control[Error] = InvalidTask;
      return;
    }
    state[int64_t(index) * 2] = task[9];
    state[int64_t(index) * 2 + 1] = 0;
    if (task[9] == 0 && !push(queue, capacity, control, index))
      return;
  }
}

__device__ inline void run_task(const int64_t *task, const int64_t *addresses, int rows) {
  const float *input = reinterpret_cast<const float *>(addresses[task[1]]);
  const float *other = task[0] == 2 ? reinterpret_cast<const float *>(addresses[task[2]]) : input;
  float *output = reinterpret_cast<float *>(addresses[task[3]]);
  int64_t end = min(task[4] + task[5], int64_t(rows));
  float factor = __int_as_float(static_cast<int>(task[8]));
  for (int64_t index = task[4] + threadIdx.x; index < end; index += blockDim.x) {
    float value = input[index];
    output[index] = task[0] == 2 ? value + other[index] : task[0] == 1 ? value * factor : value;
  }
}

__device__ inline void finish(const int64_t *task, const int *edges, int *state, int *queue,
                              int capacity, int64_t *control, int tasks, int index) {
  for (int offset = 0; offset < task[7]; ++offset) {
    int child = edges[task[6] + offset];
    if (child < 0 || child >= tasks) {
      fail(control, InvalidTask);
      break;
    }
    // The last parent acquires the writes published by all preceding parents.
    int previous = atomic(state[int64_t(child) * 2]).fetch_sub(1, cuda::memory_order_acq_rel);
    if (previous <= 0) {
      fail(control, InvalidTask);
      break;
    }
    if (previous == 1 && !push(queue, capacity, control, child))
      break;
  }
  atomic(state[int64_t(index) * 2 + 1]).store(2, cuda::memory_order_release);
  lock(control);
  ++control[Finished];
  --control[Running];
  unlock(control);
}
} // namespace tile
