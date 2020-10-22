
#include <stdio.h>

#include "../common/book.h"

__global__ void addKernel(const int *a, const int *b, int *c);

int main() {

  const int arraySize = 5;
  const int nBytes = arraySize*sizeof(int);
  const int a[arraySize] = {1, 2, 3, 4, 5};
  const int b[arraySize] = {10, 20, 30, 40, 50};
  int c[arraySize] = {0};


  // Choose which GPU to run on, change this on a multi-GPU system.
  HANDLE_ERROR(cudaSetDevice(0));

  // Allocate GPU buffers for three vectors (two input, one output).
  int *dev_a = 0;
  int *dev_b = 0;
  int *dev_c = 0;
  HANDLE_ERROR(cudaMalloc((void**) &dev_a, nBytes));
  HANDLE_ERROR(cudaMalloc((void**) &dev_b, nBytes));
  HANDLE_ERROR(cudaMalloc((void**) &dev_c, nBytes));

  // Copy input vectors from host memory to GPU buffers.
  HANDLE_ERROR(cudaMemcpy(dev_a, a, nBytes, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_b, b, nBytes, cudaMemcpyHostToDevice));

  // Launch a kernel on the GPU with one thread for each element.
  addKernel <<<1, arraySize>>> (dev_a, dev_b, dev_c);

  // cudaDeviceSynchronize waits for the kernel to finish, and returns any errors encountered during the launch.
  HANDLE_ERROR(cudaDeviceSynchronize());

  // Copy output vector from GPU buffer to host memory.
  HANDLE_ERROR(cudaMemcpy(c, dev_c, nBytes, cudaMemcpyDeviceToHost));
 printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n", c[0], c[1], c[2], c[3], c[4]);

  // cudaDeviceReset must be called before exiting in order for profiling and
  // tracing tools such as Nsight and Visual Profiler to show complete traces.
  HANDLE_ERROR(cudaDeviceReset());

  return 0;
}

__global__ void addKernel(const int *a, const int *b, int *c) {
  int i = threadIdx.x;
  c[i] = a[i] + b[i];
}
