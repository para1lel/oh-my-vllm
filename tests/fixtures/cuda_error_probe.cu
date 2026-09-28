// Appended to the production CUDA translation unit only by the isolated
// KRN-09 regression process. These functions are never in the runtime module.
__global__ void diagnostic_trap_kernel() { asm volatile("trap;"); }

void diagnostic_prior_launch_then_rms(TensorView x, TensorView weight,
                                      TensorView gate, TensorView out) {
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, x.device().device_id));
  diagnostic_trap_kernel<<<0, 1, 0, stream>>>();
  rms(x, weight, gate, out, 1e-6, false);
}

void diagnostic_prior_execution_then_rms(TensorView x, TensorView weight,
                                         TensorView gate, TensorView out) {
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, x.device().device_id));
  diagnostic_trap_kernel<<<1, 1, 0, stream>>>();
  rms(x, weight, gate, out, 1e-6, false);
}

void diagnostic_current_execution(TensorView x) {
  auto stream = stream_for(x, "diagnostic_current_execution");
  diagnostic_trap_kernel<<<1, 1, 0, stream>>>();
  finish_cuda_launch(stream, "diagnostic_current_execution");
}

int64_t diagnostic_peek_error() { return cudaPeekAtLastError(); }
int64_t diagnostic_clear_error() { return cudaGetLastError(); }
