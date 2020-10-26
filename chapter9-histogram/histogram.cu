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

// 100 mega-entries
#define nData 100*1024*1024
#define nBins 256
#define nHistogramBytes nBins*sizeof(int)

__global__ void gpuComputeHistogram(unsigned char* data, long sizeData, unsigned int* histogram);

int main(void) {

  // Allocate random data on the CPU.
  unsigned char* data = (unsigned char*) big_random_block(nData);


  // Compute the histogram on the CPU.

  clock_t cpuOnlyStart = clock();

  unsigned int cpuOnlyHistogram[nBins];
  for (int i = 0; i < nBins; i++) cpuOnlyHistogram[i] = 0;
  for (int i = 0; i < nData; i++) cpuOnlyHistogram[data[i]]++;

  float elapsedTime = (float) (clock() - cpuOnlyStart) / (float) CLOCKS_PER_SEC*1000.0f;
  printf("Time to generate: %3.1f ms\n", elapsedTime);

  long histogramSum = 0;
  for (int i = 0; i < nBins; i++) { histogramSum += cpuOnlyHistogram[i]; }
  printf("Histogram sum: %ld (should be %ld)\n\n", histogramSum, nData);


  // Compute the histogram on the GPU.

  // Start the timer.
  cudaEvent_t start, stop;
  HANDLE_ERROR(cudaEventCreate(&start));
  HANDLE_ERROR(cudaEventCreate(&stop));
  HANDLE_ERROR(cudaEventRecord(start, 0));

  // Allocate GPU memory.
  unsigned char* gpuData;
  unsigned int* gpuHistogram;
  HANDLE_ERROR(cudaMalloc((void**) &gpuData, nData));
  HANDLE_ERROR(cudaMalloc((void**) &gpuHistogram, nHistogramBytes));
  HANDLE_ERROR(cudaMemcpy(gpuData, data, nData, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemset(gpuHistogram, 0, nHistogramBytes));

  // Compute the histogram on the GPU, with each thread within a block corresponding to a single bin.
  // Setting nBlocks to 2x the number of streaming multi-processors gave the best performance.
  cudaDeviceProp  prop;
  HANDLE_ERROR(cudaGetDeviceProperties(&prop, 0));
  int nBlocks = 2*prop.multiProcessorCount;
  gpuComputeHistogram<<<nBlocks, nBins>>>(gpuData, nData, gpuHistogram);

  unsigned int cpuHistogram[nBins];
  HANDLE_ERROR(cudaMemcpy(cpuHistogram, gpuHistogram, nHistogramBytes, cudaMemcpyDeviceToHost));

  // Compute the timing.
  HANDLE_ERROR(cudaEventRecord(stop, 0));
  HANDLE_ERROR(cudaEventSynchronize(stop));
  HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
  printf("Time to generate: %3.1f ms\n", elapsedTime);

  // Verify the total number of histogram entries.
  histogramSum = 0;
  for (int i = 0; i < nBins; i++) { histogramSum += cpuHistogram[i]; }
  printf("Histogram sum: %ld (should be %ld)\n", histogramSum, nData);

  // Verify that we have the correct counts by counting down the histogram elements on the CPU.
  for (int i = 0; i < nData; i++) cpuHistogram[data[i]]--;
  for (int i = 0; i < nBins; i++) {
    if (cpuHistogram[i] != 0) printf("Failure at %d\n", i);
  }

  HANDLE_ERROR(cudaEventDestroy(start));
  HANDLE_ERROR(cudaEventDestroy(stop));
  cudaFree(gpuHistogram);
  cudaFree(gpuData);
  free(data);
  return 0;
}

__global__ void gpuComputeHistogram(unsigned char *data, long dataLength, unsigned int *histogram) {

  // Each thread maps to a single bin entry, so we can initialize with one write per thread.
  __shared__ unsigned int localHistogram[nBins];
  localHistogram[threadIdx.x] = 0;
  __syncthreads();

  // Calculate the stride offset and starting index for this thread, and perform the updates for this bin.
  int stride = gridDim.x*blockDim.x;
  int i = blockDim.x*blockIdx.x + threadIdx.x;
  while (i < dataLength) {
    atomicAdd(&localHistogram[data[i]], 1);
    i += stride;
  }

  // Wait until the local histograms are completed. 
  __syncthreads();

  // Atomically add the local histogram counts for each bin (thread) to the global histogram.
  atomicAdd(&(histogram[threadIdx.x]), localHistogram[threadIdx.x]);
}
