/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.
 * Any use, reproduction, disclosure, or distribution of this software
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA)
 * associated with this source code for terms and conditions that govern
 * your use of this NVIDIA software.
 *
 */

#include "../common/book.h"

#define imin(a,b) (a < b ? a : b)

const int N = 33*1024;
const int nThreadsPerBlock = 256;
const int nBlocks = imin(32, (N + nThreadsPerBlock - 1) / nThreadsPerBlock);
const int nDataBytes = N*sizeof(float);
const int nGridBytes = nBlocks*sizeof(float);

__global__ void dot(float *a, float *b, float *c);

int main(void) {

  // Allocate CPU memory.
  float *a, *b, c, *partial_c;
  a = (float*) malloc(nDataBytes);
  b = (float*) malloc(nDataBytes);
  partial_c = (float*) malloc(nGridBytes);

  // Fill in the host memory with data.
  for (int i = 0; i < N; i++) {
    a[i] = i;
    b[i] = 2*i;
  }

  // Allocate GPU memory.
  float *dev_a, *dev_b, *dev_partial_c;
  HANDLE_ERROR(cudaMalloc((void**) &dev_a, nDataBytes));
  HANDLE_ERROR(cudaMalloc((void**) &dev_b, nDataBytes));
  HANDLE_ERROR(cudaMalloc((void**) &dev_partial_c, nGridBytes));

  // Copy the arrays a and b to the GPU.
  HANDLE_ERROR(cudaMemcpy(dev_a, a, nDataBytes, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_b, b, nDataBytes, cudaMemcpyHostToDevice));
  dot<<<nBlocks, nThreadsPerBlock>>>(dev_a, dev_b, dev_partial_c);

  // Copy the array c back from the GPU to the CPU.
  HANDLE_ERROR(cudaMemcpy(partial_c, dev_partial_c, nGridBytes, cudaMemcpyDeviceToHost));

  // Complete the sum on the CPU.
  c = 0;
  for (int i = 0; i < nBlocks; i++) { c += partial_c[i]; }

#define sum_squares(x)  (x*(x+1)*(2*x+1)/6)
  printf("Does GPU value %.6g = %.6g?\n", c, 2*sum_squares((float) (N - 1)));

  // Free the GPU memory.
  HANDLE_ERROR(cudaFree(dev_a));
  HANDLE_ERROR(cudaFree(dev_b));
  HANDLE_ERROR(cudaFree(dev_partial_c));

  // Free the CPU memory.
  free(a);
  free(b);
  free(partial_c);
}


__global__ void dot(float *a, float *b, float *c) {

  __shared__ float cache[nThreadsPerBlock];

  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  int cacheIndex = threadIdx.x;

  // Set the cache values.
  float temp = 0;
  while (tid < N) {
    temp += a[tid]*b[tid];
    tid += blockDim.x*gridDim.x;
  }
  cache[cacheIndex] = temp;

  // Synchronize threads in this block
  __syncthreads();

  // For reductions, nThreadsPerBlock must be a power of 2 because of the following code
  int i = blockDim.x / 2;
  while (i != 0) {
    if (cacheIndex < i) { cache[cacheIndex] += cache[cacheIndex + i]; }
    __syncthreads();
    i /= 2;
  }

  if (cacheIndex == 0) { c[blockIdx.x] = cache[0]; }
}
