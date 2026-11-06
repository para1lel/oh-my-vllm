#pragma once

inline void check_blas(cublasStatus_t status) {
  TVM_FFI_ICHECK(status == CUBLAS_STATUS_SUCCESS) << "cuBLAS status " << int(status);
}

struct BlasOwner {
  cublasHandle_t handle = nullptr;
  cublasHandle_t limited = nullptr;
  int device;

  explicit BlasOwner(int device) : device(device) {
    int previous = 0;
    check_cuda(cudaGetDevice(&previous));
    check_cuda(cudaSetDevice(device));
    cublasStatus_t status = cublasCreate(&handle);
    if (status == CUBLAS_STATUS_SUCCESS)
      status = cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
    if (status == CUBLAS_STATUS_SUCCESS)
      status = cublasCreate(&limited);
    if (status == CUBLAS_STATUS_SUCCESS)
      status = cublasSetMathMode(limited, CUBLAS_DEFAULT_MATH);
    // The released BF16 multi-row projection uses a smaller SM target. This
    // avoids the split-K kernel that the current library selects by default.
    if (status == CUBLAS_STATUS_SUCCESS)
      status = cublasSetSmCountTarget(limited, 8);
    if (status != CUBLAS_STATUS_SUCCESS) {
      if (limited)
        cublasDestroy(limited);
      if (handle)
        cublasDestroy(handle);
      check_cuda(cudaSetDevice(previous));
      check_blas(status);
    }
    check_cuda(cudaSetDevice(previous));
  }

  ~BlasOwner() {
    int previous = 0;
    if (cudaGetDevice(&previous) == cudaSuccess && cudaSetDevice(device) == cudaSuccess) {
      cublasDestroy(handle);
      cublasDestroy(limited);
      cudaSetDevice(previous);
    }
  }
};

inline BlasOwner &owner(int device, cudaStream_t stream) {
  static thread_local std::unordered_map<int, std::unique_ptr<BlasOwner>> owners;
  auto found = owners.find(device);
  if (found == owners.end()) {
    cudaStreamCaptureStatus capture;
    check_cuda(cudaStreamIsCapturing(stream, &capture));
    TVM_FFI_ICHECK(capture == cudaStreamCaptureStatusNone) << "prepare HC before Graph capture";
    found = owners.emplace(device, std::make_unique<BlasOwner>(device)).first;
  }
  return *found->second;
}

inline cublasHandle_t handle(int device, cudaStream_t stream) {
  return owner(device, stream).handle;
}

inline cublasHandle_t limited_handle(int device, cudaStream_t stream) {
  return owner(device, stream).limited;
}

inline void bf16_multiply(int device, const void *input, const void *weight, void *output, int rows,
                          int outputs, int width, cudaStream_t stream) {
  auto blas = rows == 1 ? handle(device, stream) : limited_handle(device, stream);
  check_blas(cublasSetStream(blas, stream));
  float alpha = 1.f, beta = 0.f;
  check_blas(cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, outputs, rows, width, &alpha, weight,
                          CUDA_R_16BF, width, input, CUDA_R_16BF, width, &beta, output, CUDA_R_16BF,
                          outputs, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}
