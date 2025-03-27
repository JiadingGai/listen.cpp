#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

// g++ -o cuda_graph_ptx cuda_graph_ptx.cpp -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcuda

/*
// The ptx code below implements the following cuda kernel
__global__ void double_values(float *data, int n) {
  int bidx = blockIdx.x;
  int tidx = threadIdx.x;

  int global_offset = bidx * 256 + tidx;
  if (global_offset < n) {
    data[global_offset] *= 2.0f;
  }
}
 */

// Error checking macro for Driver API calls
#define CHECK_CUDA(call) do { \
  CUresult err = call; \
  if (err != CUDA_SUCCESS) { \
    const char *errStr; \
    cuGetErrorString(err, &errStr); \
    fprintf(stderr, "CUDA Error: %s at %s: %d\n", errStr, __FILE__, __LINE__); \
    exit(EXIT_FAILURE); \
  } \
} while (0)

const char *ptxCode =
  ".version 8.4\n"
  ".target sm_80\n"
  ".address_size 64\n"
  ".visible .entry double_values(.param .u64 data, .param .u32 n) {\n"
  "    .reg .pred %p<2>;\n"
  "    .reg .f32  %f<3>;\n"
  "    .reg .b32  %r<5>;\n"  // Increased to 5 registers for %r4
  "    .reg .b64  %rd<5>;\n"
  "    ld.param.u64 %rd1, [data];\n"
  "    ld.param.u32 %r2, [n];\n"
  "    mov.u32 %r3, %ctaid.x;\n"
  "    mov.u32 %r4, %tid.x;\n"          // Get threadIdx.x
  "    mad.lo.s32 %r1, %r3, 256, %r4;\n" // global_offset = bidx * 256 + tidx
  "    setp.ge.s32 %p1, %r1, %r2;\n"
  "    @%p1 bra L_exit;\n"
  "    cvta.to.global.u64 %rd2, %rd1;\n"
  "    mul.wide.s32 %rd3, %r1, 4;\n"
  "    add.s64 %rd4, %rd2, %rd3;\n"
  "    ld.global.f32 %f1, [%rd4];\n"
  "    add.f32 %f2, %f1, %f1;\n"
  "    st.global.f32 [%rd4], %f2;\n"
  "L_exit:\n"
  "    ret;\n"
  "}\n";

int main() {
  // Initialize CUDA Driver API
  CHECK_CUDA(cuInit(0));

  // Get device and create context
  CUdevice device;
  CUcontext context;
  CHECK_CUDA(cuDeviceGet(&device, 0));
  CHECK_CUDA(cuCtxCreate(&context, 0, device));

  // Allocate device memory
  CUdeviceptr d_data;
  const int N = 1024;
  CHECK_CUDA(cuMemAlloc(&d_data, N * sizeof(float)));

  // Initialize data with 1.0f
  float *h_data = (float *) malloc(N * sizeof(float));
  for (int i = 0; i < N; i++) {
    h_data[i] = 1.0f;
  }
  CHECK_CUDA(cuMemcpyHtoD(d_data, h_data, N * sizeof(float)));
  free(h_data);

  // Load PTX code into a module
  CUmodule module;
  CHECK_CUDA(cuModuleLoadData(&module, ptxCode));

  // Get kernel function from module
  CUfunction kernel;
  CHECK_CUDA(cuModuleGetFunction(&kernel, module, "double_values"));

  // Create a CUDA graph
  CUgraph graph;
  CHECK_CUDA(cuGraphCreate(&graph, 0));

  // Set up kernel parameters
  void *kernelParams[] = { &d_data, (void*)&N };
  CUgraphNode kernelNode;
  CUDA_KERNEL_NODE_PARAMS kernelNodeParams = {0};
  kernelNodeParams.func = kernel;
  kernelNodeParams.gridDimX = N / 256;
  kernelNodeParams.gridDimY = 1;
  kernelNodeParams.gridDimZ = 1;
  kernelNodeParams.blockDimX = 256;
  kernelNodeParams.blockDimY = 1;
  kernelNodeParams.blockDimZ = 1;
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = kernelParams;
  kernelNodeParams.extra = nullptr;

  // Add kernel node to graph
  CHECK_CUDA(cuGraphAddKernelNode(&kernelNode, graph, nullptr, 0, &kernelNodeParams));

  // Instantiate the graph
  CUgraphExec graphExec;
  CHECK_CUDA(cuGraphInstantiate(&graphExec, graph, 0ULL));

  // Launch the graph
  // 0 uses default stream
  CHECK_CUDA(cuGraphLaunch(graphExec, 0)); 

  // Synchronize context
  CHECK_CUDA(cuCtxSynchronize());

  // Verify results
  h_data = (float *) malloc(N * sizeof(float));
  CHECK_CUDA(cuMemcpyDtoH(h_data, d_data, N * sizeof(float)));
  for (int i = 0; i < 10; i++) {
    if (h_data[i] != 2.0f) {
      printf("Verification failed at index %d: %f\n", i, h_data[i]);
      break;
    }
  }
  printf("Graph executed successfully!\n");
  free(h_data);

  // Clean up
  CHECK_CUDA(cuGraphExecDestroy(graphExec));
  CHECK_CUDA(cuGraphDestroy(graph));
  CHECK_CUDA(cuMemFree(d_data));
  CHECK_CUDA(cuModuleUnload(module));
  CHECK_CUDA(cuCtxDestroy(context));

  return 0;
}
