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

#define imin(a, b) (a < b ? a : b)

const int dataLength = 33*1024*1024;
const int threadsPerBlock = 256;
const int nBlocks = imin(32, (dataLength + threadsPerBlock - 1) / threadsPerBlock);

__global__ void gpuComputeThreadwiseDotProduct(float *in0, float *in1, float *out);
float runDotProductTest(bool useZeroCopyMemory);

int main(void) {

  int gpuID;
  cudaDeviceProp gpuProperties;
  HANDLE_ERROR(cudaGetDevice(&gpuID));
  HANDLE_ERROR(cudaGetDeviceProperties(&gpuProperties, gpuID));
  if (gpuProperties.canMapHostMemory != true) {
    printf("Device can not map memory.\n");
    return 0;
  }
  HANDLE_ERROR(cudaSetDeviceFlags(cudaDeviceMapHost));

  float elapsedTime;

  // Perform the test by using malloc to allocate memory on the CPU and transferring to the GPU as usual.
  elapsedTime = runDotProductTest(false);
  printf("Time using cudaMalloc: %3.1f ms\n", elapsedTime);

  // Perform the test by using cudaHostAlloc to allocate zero-copy memory on the CPU.
  elapsedTime = runDotProductTest(true);
  printf("Time using zero-copy memory with cudaHostAlloc: %3.1f ms\n", elapsedTime);
}

float runDotProductTest(bool useZeroCopyMemory) {

  cudaEvent_t start, stop;
  HANDLE_ERROR(cudaEventCreate(&start));
  HANDLE_ERROR(cudaEventCreate(&stop));

  float *a, *b, *aTimesB, dotProduct;
  float *gpuA, *gpuB, *gpuATimesB;
  long nBytesData = dataLength*sizeof(float);

  if (useZeroCopyMemory) {

    // Allocate zero-copy memory on the CPU.
    HANDLE_ERROR(cudaHostAlloc((void**) &a, nBytesData, cudaHostAllocWriteCombined | cudaHostAllocMapped));
    HANDLE_ERROR(cudaHostAlloc((void**) &b, nBytesData, cudaHostAllocWriteCombined | cudaHostAllocMapped));
    HANDLE_ERROR(cudaHostAlloc((void**) &aTimesB, nBlocks*sizeof(float), cudaHostAllocMapped));

    // Set the GPU pointers.
    HANDLE_ERROR(cudaHostGetDevicePointer(&gpuA, a, 0));
    HANDLE_ERROR(cudaHostGetDevicePointer(&gpuB, b, 0));
    HANDLE_ERROR(cudaHostGetDevicePointer(&gpuATimesB, aTimesB, 0));

  } else {

    // Allocate CPU memory.
    a = (float*) malloc(nBytesData);
    b = (float*) malloc(nBytesData);
    aTimesB = (float*) malloc(nBlocks*sizeof(float));

    // Allocate GPU memory.
    HANDLE_ERROR(cudaMalloc((void**) &gpuA, nBytesData));
    HANDLE_ERROR(cudaMalloc((void**) &gpuB, nBytesData));
    HANDLE_ERROR(cudaMalloc((void**) &gpuATimesB, nBlocks*sizeof(float)));
  }

  // Fill in the host memory with data.
  for (int i = 0; i < dataLength; i++) {
    a[i] = i;
    b[i] = i*2;
  }

  HANDLE_ERROR(cudaEventRecord(start, 0));

  if (!useZeroCopyMemory) {
    // Copy arrays a and b to the GPU.
    HANDLE_ERROR(cudaMemcpy(gpuA, a, nBytesData, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(gpuB, b, nBytesData, cudaMemcpyHostToDevice));
  }

  // Compute the dot product on the GPU.
  gpuComputeThreadwiseDotProduct<<<nBlocks, threadsPerBlock>>>(gpuA, gpuB, gpuATimesB);

  if (!useZeroCopyMemory) {
    // Copy array c back to the CPU.
    HANDLE_ERROR(cudaMemcpy(aTimesB, gpuATimesB, nBlocks*sizeof(float), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaThreadSynchronize());
  }

  HANDLE_ERROR(cudaEventRecord(stop, 0));
  HANDLE_ERROR(cudaEventSynchronize(stop));
  float elapsedTime;
  HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));

  // Complete the sum on the CPU.
  dotProduct = 0;
  for (int i = 0; i < nBlocks; i++) { dotProduct += aTimesB[i]; }

  if (useZeroCopyMemory) {
    HANDLE_ERROR(cudaFreeHost(a)); // instead of cudaFree, since the memory is on the CPU
    HANDLE_ERROR(cudaFreeHost(b));
    HANDLE_ERROR(cudaFreeHost(aTimesB));
  } else {
    HANDLE_ERROR(cudaFree(gpuA));
    HANDLE_ERROR(cudaFree(gpuB));
    HANDLE_ERROR(cudaFree(gpuATimesB));
    free(a);
    free(b);
    free(aTimesB);
  }

  HANDLE_ERROR(cudaEventDestroy(start));
  HANDLE_ERROR(cudaEventDestroy(stop));

  printf("Value calculated: %f\n", dotProduct);
  return elapsedTime;
}

__global__ void gpuComputeThreadwiseDotProduct(float *in0, float *in1, float *out) {

  __shared__ float cache[threadsPerBlock];
  int tid = blockDim.x*blockIdx.x + threadIdx.x;
  int stride = gridDim.x*blockDim.x;
  int cacheIndex = threadIdx.x;

  float threadwiseDotProduct = 0;
  while (tid < dataLength) {
    threadwiseDotProduct += in0[tid]*in1[tid];
    tid += stride;
  }

  // Set the cache values and synchronize across threads before proceeding.
  cache[cacheIndex] = threadwiseDotProduct;
  __syncthreads();

  // for reductions, threadsPerBlock must be a power of 2 because of the following code
  int i = blockDim.x/2;
  while (i != 0) {
    if (cacheIndex < i)
      cache[cacheIndex] += cache[cacheIndex + i];
    __syncthreads();
    i /= 2;
  }

  if (cacheIndex == 0) { out[blockIdx.x] = cache[0]; }
}
