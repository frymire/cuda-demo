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

#define chunkSize 1024*1024
#define nBytesChunk chunkSize*sizeof(int)
#define nData 20*chunkSize
#define nBytesData nData*sizeof(int)
#define nThreadsPerBlock 256
#define nBlocks chunkSize/nThreadsPerBlock

__global__ void gpuComputeChunk(int* a, int* b, int* c);

int main(void) {

  cudaDeviceProp gpuProperties;
  int whichDevice;
  HANDLE_ERROR(cudaGetDevice(&whichDevice));
  HANDLE_ERROR(cudaGetDeviceProperties(&gpuProperties, whichDevice));
  if (!gpuProperties.deviceOverlap) {
    printf("Device will not handle overlaps, so no speed up from streams.\n");
    return 0;
  }

  cudaEvent_t start, stop;
  float elapsedTime;

  cudaStream_t stream0, stream1;
  int *cpuA, *cpuB, *cpuC;
  int *gpuA0, *gpuB0, *gpuC0;
  int *gpuA1, *gpuB1, *gpuC1;

  // Start the timers.
  HANDLE_ERROR(cudaEventCreate(&start));
  HANDLE_ERROR(cudaEventCreate(&stop));

  // Initialize the streams.
  HANDLE_ERROR(cudaStreamCreate(&stream0));
  HANDLE_ERROR(cudaStreamCreate(&stream1));

  // Allocate GPU memory.
  HANDLE_ERROR(cudaMalloc((void**) &gpuA0, nBytesChunk));
  HANDLE_ERROR(cudaMalloc((void**) &gpuB0, nBytesChunk));
  HANDLE_ERROR(cudaMalloc((void**) &gpuC0, nBytesChunk));
  HANDLE_ERROR(cudaMalloc((void**) &gpuA1, nBytesChunk));
  HANDLE_ERROR(cudaMalloc((void**) &gpuB1, nBytesChunk));
  HANDLE_ERROR(cudaMalloc((void**) &gpuC1, nBytesChunk));

  // Allocate host locked memory, used to stream.
  HANDLE_ERROR(cudaHostAlloc((void**) &cpuA, nBytesData, cudaHostAllocDefault));
  HANDLE_ERROR(cudaHostAlloc((void**) &cpuB, nBytesData, cudaHostAllocDefault));
  HANDLE_ERROR(cudaHostAlloc((void**) &cpuC, nBytesData, cudaHostAllocDefault));

  for (int i = 0; i < nData; i++) {
    cpuA[i] = rand();
    cpuB[i] = rand();
  }

  HANDLE_ERROR(cudaEventRecord(start, 0));

  // Loop over the data in bite-sized chunks.
  for (int i = 0; i < nData; i += 2*chunkSize) {

    // Bad design here, done as a demo. Queuing all tasks for stream0 before all tasks for stream1.
    // (Actually, this gives the same performance on a 2080 Ti. Is the scheduler automatically rescheduling these?)

    // Queue all tasks for stream0  (bad move).
    HANDLE_ERROR(cudaMemcpyAsync(gpuA0, cpuA + i, nBytesChunk, cudaMemcpyHostToDevice, stream0));
    HANDLE_ERROR(cudaMemcpyAsync(gpuB0, cpuB + i, nBytesChunk, cudaMemcpyHostToDevice, stream0));
    gpuComputeChunk<<<nBlocks, nThreadsPerBlock, 0, stream0>>>(gpuA0, gpuB0, gpuC0);
    HANDLE_ERROR(cudaMemcpyAsync(cpuC + i, gpuC0, nBytesChunk, cudaMemcpyDeviceToHost, stream0));

    // Queue all tasks for stream1 (bad move).
    HANDLE_ERROR(cudaMemcpyAsync(gpuA1, cpuA + i + chunkSize, nBytesChunk, cudaMemcpyHostToDevice, stream1));
    HANDLE_ERROR(cudaMemcpyAsync(gpuB1, cpuB + i + chunkSize, nBytesChunk, cudaMemcpyHostToDevice, stream1));
    gpuComputeChunk<<<nBlocks, nThreadsPerBlock, 0, stream1>>>(gpuA1, gpuB1, gpuC1);
    HANDLE_ERROR(cudaMemcpyAsync(cpuC + i + chunkSize, gpuC1, nBytesChunk, cudaMemcpyDeviceToHost, stream1));
  }

  // Synch the streams to wait for the computations to finish.
  HANDLE_ERROR(cudaStreamSynchronize(stream0));
  HANDLE_ERROR(cudaStreamSynchronize(stream1));

  // Measure time.
  HANDLE_ERROR(cudaEventRecord(stop, 0));
  HANDLE_ERROR(cudaEventSynchronize(stop));
  HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
  printf("Time taken: %3.1f ms\n", elapsedTime);

  // Clean up the streams and memory.
  HANDLE_ERROR(cudaFreeHost(cpuA));
  HANDLE_ERROR(cudaFreeHost(cpuB));
  HANDLE_ERROR(cudaFreeHost(cpuC));
  HANDLE_ERROR(cudaFree(gpuA0));
  HANDLE_ERROR(cudaFree(gpuB0));
  HANDLE_ERROR(cudaFree(gpuC0));
  HANDLE_ERROR(cudaFree(gpuA1));
  HANDLE_ERROR(cudaFree(gpuB1));
  HANDLE_ERROR(cudaFree(gpuC1));
  HANDLE_ERROR(cudaStreamDestroy(stream0));
  HANDLE_ERROR(cudaStreamDestroy(stream1));

  return 0;
}

__global__ void gpuComputeChunk(int* a, int* b, int* c) {

  int i0 = blockIdx.x*blockDim.x + threadIdx.x;

  if (i0 < chunkSize) {
    int i1 = (i0 + 1) % nThreadsPerBlock;
    int i2 = (i0 + 2) % nThreadsPerBlock;
    c[i0] = (a[i0] + a[i1] + a[i2] + b[i0] + b[i1] + b[i2]) / 6.0f;
  }
}
