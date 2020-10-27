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
#define N (33*1024*1024)
const int threadsPerBlock = 256;
const int nBlocks = imin(32, ( N/2 + threadsPerBlock - 1 ) / threadsPerBlock);

struct DataStruct {
  int deviceID;
  int size;
  float *a;
  float *b;
  float returnValue;
};

unsigned WINAPI taskGPU(void *gpuTask);
__global__ void dot(int size, float *a, float *b, float *c);

int main(void) {

  int deviceCount;
  HANDLE_ERROR(cudaGetDeviceCount(&deviceCount));
  if (deviceCount < 2) {
    printf("We need at least two compute 1.0 or greater devices, but only found %d\n", deviceCount);
    return 0;
  }

  float *a = (float*) malloc(sizeof(float)*N);
  HANDLE_NULL(a);
  float *b = (float*) malloc(sizeof(float)*N);
  HANDLE_NULL(b);

  // fill in the host memory with data
  for (int i = 0; i < N; i++) {
    a[i] = i;
    b[i] = 2*i;
  }

  // prepare for multithread
  DataStruct data[2];
  data[0].deviceID = 0;
  data[0].size = N/2;
  data[0].a = a;
  data[0].b = b;

  data[1].deviceID = 1;
  data[1].size = N/2;
  data[1].a = a + N/2;
  data[1].b = b + N/2;

  CUTThread thread = start_thread(taskGPU, &(data[0]));
  taskGPU(&(data[1]));
  end_thread(thread);

  free(a);
  free(b);

  printf("Value calculated: \t%f\n", data[0].returnValue + data[1].returnValue);
  printf("Should be:\t\t%f.\n", 27621693407370839851008.0f);
  return 0;
}

unsigned WINAPI taskGPU(void* gpuData) {

  DataStruct* data = (DataStruct*) gpuData;
  cudaDeviceProp gpuProperties;
  HANDLE_ERROR(cudaGetDeviceProperties(&gpuProperties, data->deviceID));
  printf("Tasking GPU#%d (%s)...\n", data->deviceID, gpuProperties.name);
  HANDLE_ERROR(cudaSetDevice(data->deviceID));

  // Allocate CPU memory.
  float* a = data->a;
  float* b = data->b;
  float* partialC = (float*) malloc(nBlocks*sizeof(float));

  // Allocate GPU memory.
  int size = data->size;
  int nBytesData = size*sizeof(float);
  float *gpuA, *gpuB, *gpuPartialC;
  HANDLE_ERROR(cudaMalloc((void**) &gpuA, nBytesData));
  HANDLE_ERROR(cudaMalloc((void**) &gpuB, nBytesData));
  HANDLE_ERROR(cudaMalloc((void**) &gpuPartialC, nBlocks*sizeof(float)));

  // Copy arrays a and b to the GPU.
  HANDLE_ERROR(cudaMemcpy(gpuA, a, nBytesData, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(gpuB, b, nBytesData, cudaMemcpyHostToDevice));

  dot<<<nBlocks, threadsPerBlock>>>(size, gpuA, gpuB, gpuPartialC);

  // Copy array c from the GPU to the CPU.
  HANDLE_ERROR(cudaMemcpy(partialC, gpuPartialC, nBlocks*sizeof(float), cudaMemcpyDeviceToHost));

  // Complete the dot product calculation on the CPU.
  float dotProduct = 0;
  for (int i = 0; i < nBlocks; i++) { dotProduct += partialC[i]; }
  data->returnValue = dotProduct;

  HANDLE_ERROR(cudaFree(gpuA));
  HANDLE_ERROR(cudaFree(gpuB));
  HANDLE_ERROR(cudaFree(gpuPartialC));
  free(partialC);

  return 0;
}

__global__ void dot(int size, float *a, float *b, float *c) {

  __shared__ float cache[threadsPerBlock];
  int tid = blockDim.x*blockIdx.x + threadIdx.x;
  int cacheIndex = threadIdx.x;

  float temp = 0;
  while (tid < size) {
    temp += a[tid]* b[tid];
    tid += gridDim.x*blockDim.x;
  }

  // set the cache values
  cache[cacheIndex] = temp;

  // synchronize threads in this block
  __syncthreads();

  // for reductions, threadsPerBlock must be a power of 2 because of the following code
  int i = blockDim.x/2;
  while (i != 0) {
    if (cacheIndex < i)
      cache[cacheIndex] += cache[cacheIndex + i];
    __syncthreads();
    i /= 2;
  }

  if (cacheIndex == 0) { c[blockIdx.x] = cache[0]; }
}
