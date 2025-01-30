#include <iostream>
#include <cassert>
#include <cmath>
#include <random>
#include <cuda_runtime.h>

/*
Compile: nvcc test.cu -lineinfo
Profile:
ncu --set full --metrics smsp__inst_executed_op_ldgsts.sum, -fo __sample_profile --kernel-name ReferenceGemm_kernel ./a.out

NVBit:
nvcc test.cu -lineinfo -lnvbi -L./nbbit_release/core
LD_PRELOAD=./nvbit_release/tools/mem_trace/mem_trace.so ./a.out
*/

__global__ void ReferenceGemm_kernel(
    int M, int N, int K, float alpha, const float *A, int lda,
    const float *B, int ldb, float beta, float *C, int ldc) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < M && j < N) {
    float accumulator = 0.0f;
    for (int k = 0; k < K; ++k) {
      accumulator += A[k + i * lda] * B[j + k * ldb];
    }
    C[j + i * ldc] = alpha * accumulator + beta * C[j + i * ldc];
  }
}

int main() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 1.0);

  int M = 128, N = 256, K = 8192;
  float alpha = 1.0, beta = 0.0;

  int size_A = M * K, size_B = N * K, size_C = M * N;
  float *h_A = new float[size_A];
  float *h_B = new float[size_B];
  float *h_C = new float[size_C];
  int lda = K, ldb = N, ldc = N;

  for (int i = 0; i < size_A; i++) h_A[i] = dis(gen);
  for (int i = 0; i < size_B; i++) h_B[i] = dis(gen);
  for (int i = 0; i < size_C; i++) h_C[i] = 0.0f;

  float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
  cudaError_t err;
  
  err = cudaMalloc((void**)&d_A, size_A * sizeof(float));
  assert(err == cudaSuccess);
  
  err = cudaMalloc((void**)&d_B, size_B * sizeof(float));
  assert(err == cudaSuccess);
  
  err = cudaMalloc((void**)&d_C, size_C * sizeof(float));
  assert(err == cudaSuccess);

  err = cudaMemcpy(d_A, h_A, size_A * sizeof(float), cudaMemcpyHostToDevice);
  assert(err == cudaSuccess);

  err = cudaMemcpy(d_B, h_B, size_B * sizeof(float), cudaMemcpyHostToDevice);
  assert(err == cudaSuccess);

  dim3 block(16, 16, 1);
  dim3 grid((M + 15) / 16, (N + 15) / 16, 1);

  ReferenceGemm_kernel<<<grid, block>>>(M, N, K, alpha, d_A, lda, d_B, ldb, beta, d_C, ldc);
  cudaDeviceSynchronize();

  err = cudaMemcpy(h_C, d_C, size_C * sizeof(float), cudaMemcpyDeviceToHost);
  assert(err == cudaSuccess);

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      float result = 0.0f;
      for (int k = 0; k < K; k++) {
        result += h_A[i * lda + k] * h_B[j + k * ldb];
      }
      if (std::abs(h_C[i * ldc + j] - result) > 1e-4) {
        printf("Mismatch at (%d, %d): %f vs %f\n", i, j, h_C[i * ldc + j], result);
        assert(false);
      }
    }
  }

  delete[] h_A;
  delete[] h_B;
  delete[] h_C;
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  return 0;
}
