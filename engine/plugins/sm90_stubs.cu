// Hopper (FC_SM=90) build: the 2:4-sparse FP8 kernels (sparse_fp8.cu) and the NVFP4 kernels (nvfp4_ffn.cu) are
// sm_120-family only (Hopper has no FP4 tensor cores; the sparse collective is sm120-specific here). These stubs keep
// the plugins linkable and make any use of those paths fail loudly instead of falling back.
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdio>

static int refuse(const char* what) {
  fprintf(stderr, "fastconformer-trt: %s is not available in the Hopper (FC_SM=90) build\n", what);
  return 99;
}
extern "C" {
void* sparse_fp8_prepare(const void*, int, int, cudaStream_t) { refuse("2:4 sparse FP8"); return nullptr; }
void* sparse_fp8_prepare_l(const void*, int, int, int, cudaStream_t) { refuse("2:4 sparse FP8"); return nullptr; }
int sparse_ffn_run(void*, void*, const void*, void*, const void*, void*, int, float, float, float, float, int, cudaStream_t) {
  return refuse("2:4 sparse FP8");
}
int sparse_linear_run(void*, const void*, const void*, void*, int, float, float, int, cudaStream_t) {
  return refuse("2:4 sparse FP8");
}
int sparse_qkv_run(void**, const void*, const float*, void**, int, int, int, const float*, cudaStream_t) {
  return refuse("2:4 sparse FP8");
}
size_t nvfp4_sf_bytes(int, int) { return 0; }
int nvfp4_quant(const void*, void*, void*, int, int, float, int, cudaStream_t) { return refuse("NVFP4"); }
int nvfp4_gemm1(const void*, const void*, const void*, const void*, void*, void*, int, int, int, float, const float*,
                cudaStream_t) { return refuse("NVFP4"); }
int nvfp4_gemm2(const void*, const void*, const void*, const void*, const void*, void*, int, int, int, float, float,
                cudaStream_t) { return refuse("NVFP4"); }
}
