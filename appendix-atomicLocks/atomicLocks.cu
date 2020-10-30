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
#include "lock.h"

#define imin(in0, in1) (in0 < in1 ? in0 : in1)
#define sum_squares(x) (x*(x + 1)*(2*x + 1) / 6)

const int N = 33*1024*1024;
const int nDataBytes = N*sizeof(float);
const int threadsPerBlock = 256;
const int nBlocks = imin(32, (N + threadsPerBlock - 1) / threadsPerBlock);

__global__ void gpuComputeDotProduct(Lock lock, float *in0, float *in1, float *out);

int main(void) {

  // Allocate CPU memory.
  float* in0 = (float*) malloc(nDataBytes);
  float* in1 = (float*) malloc(nDataBytes);
  float dotProduct = 0;

  // Populate the host memory with data.
  for (int i = 0; i < N; i++) {
    in0[i] = i;
    in1[i] = 2*i;
  }

  // Allocate GPU memory.
  float *gpuIn0, *gpuIn1, *gpuDotProduct;
  HANDLE_ERROR(cudaMalloc((void**) &gpuIn0, nDataBytes));
  HANDLE_ERROR(cudaMalloc((void**) &gpuIn1, nDataBytes));
  HANDLE_ERROR(cudaMalloc((void**) &gpuDotProduct, sizeof(float)));

  // Copy the input arrays and the initial value (0) of the output variable to the GPU.
  HANDLE_ERROR(cudaMemcpy(gpuIn0, in0, nDataBytes, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(gpuIn1, in1, nDataBytes, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(gpuDotProduct, &dotProduct, sizeof(float), cudaMemcpyHostToDevice));

  // Define a lock to be shared across GPU threads when updating the gpuDotProduct return value.
  Lock lock;
  gpuComputeDotProduct<<<nBlocks, threadsPerBlock>>>(lock, gpuIn0, gpuIn1, gpuDotProduct);

  // Copy the result back from the GPU to the CPU.
  HANDLE_ERROR(cudaMemcpy(&dotProduct, gpuDotProduct, sizeof(float), cudaMemcpyDeviceToHost));
  printf("GPU dot product = %.6g\nShould be = %.6g\n", dotProduct, 2*sum_squares((float) (N - 1)));

  // Free memory.
  HANDLE_ERROR(cudaFree(gpuIn0));
  HANDLE_ERROR(cudaFree(gpuIn1));
  HANDLE_ERROR(cudaFree(gpuDotProduct));
  free(in0);
  free(in1);

  return 0;
}

__global__ void gpuComputeDotProduct(Lock lock, float *in0, float *in1, float *out) {

  __shared__ float cache[threadsPerBlock];
  int tid = blockDim.x*blockIdx.x + threadIdx.x;
  int cacheIndex = threadIdx.x;

  float localDotProduct = 0;
  while (tid < N) {
    localDotProduct += in0[tid]*in1[tid];
    tid += gridDim.x*blockDim.x;
  }

  // Set the cache values, and let all threads before moving on to the reduction.
  cache[cacheIndex] = localDotProduct;
  __syncthreads();

  // For reductions, threadsPerBlock must be a power of 2 because of the following code
  int i = blockDim.x/2;
  while (i != 0) {
    if (cacheIndex < i) { cache[cacheIndex] += cache[cacheIndex + i]; }
    __syncthreads();
    i /= 2;
  }

  // This time, get a lock before atomically updating the output value, then release afterwards.
  // For comparision, check the value that you get if you comment out the lock() and unlock() calls.
  if (cacheIndex == 0) {
    lock.lock();
    *out += cache[0];
    lock.unlock();
  }
}
