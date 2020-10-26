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

#define SIZE 64*1024*1024

float cuda_malloc_test(int size, bool up);
float cuda_host_alloc_test(int size, bool up);

int main(void) {

  float elapsedTime;
  const float MB = (float) 100*SIZE*sizeof(int)/1024/1024;
  
  // Try it with cudaMalloc.
  elapsedTime = cuda_malloc_test(SIZE, true);
  printf("Copy up with cudaMalloc: %3.1f ms, %3.1f MB/s\n", elapsedTime, MB/(elapsedTime/1000));

  elapsedTime = cuda_malloc_test(SIZE, false);
  printf("Copy down with cudaMalloc: %3.1f ms, %3.1f MB/s\n", elapsedTime, MB/(elapsedTime/1000));

  // Now try it with cudaHostAlloc to use page-locked memory on the CPU.
  elapsedTime = cuda_host_alloc_test(SIZE, true);
  printf("Copy up with cudaHostMalloc: %3.1f ms, %3.1f MB/s\n", elapsedTime, MB/(elapsedTime/1000));

  elapsedTime = cuda_host_alloc_test(SIZE, false);
  printf("Copy down with cudaHostMalloc: %3.1f ms, %3.1f MB/s\n", elapsedTime, MB/(elapsedTime/1000));
}

float cuda_malloc_test(int size, bool up) {

  cudaEvent_t start, stop;
  int *a, *dev_a;
  float elapsedTime;

  HANDLE_ERROR(cudaEventCreate(&start));
  HANDLE_ERROR(cudaEventCreate(&stop));

  a = (int*) malloc(size * sizeof(*a));
  HANDLE_NULL(a);
  HANDLE_ERROR(cudaMalloc((void**) &dev_a, size * sizeof(*dev_a)));

  HANDLE_ERROR(cudaEventRecord(start, 0));
  for (int i = 0; i<100; i++) {
    if (up)
      HANDLE_ERROR(cudaMemcpy(dev_a, a, size * sizeof(*dev_a), cudaMemcpyHostToDevice));
    else
      HANDLE_ERROR(cudaMemcpy(a, dev_a, size * sizeof(*dev_a), cudaMemcpyDeviceToHost));
  }
  HANDLE_ERROR(cudaEventRecord(stop, 0));
  HANDLE_ERROR(cudaEventSynchronize(stop));
  HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));

  free(a);
  HANDLE_ERROR(cudaFree(dev_a));
  HANDLE_ERROR(cudaEventDestroy(start));
  HANDLE_ERROR(cudaEventDestroy(stop));

  return elapsedTime;
}

float cuda_host_alloc_test(int size, bool up) {

  cudaEvent_t start, stop;
  int *a, *dev_a;
  float elapsedTime;

  HANDLE_ERROR(cudaEventCreate(&start));
  HANDLE_ERROR(cudaEventCreate(&stop));

  HANDLE_ERROR(cudaHostAlloc((void**) &a, size*sizeof(*a), cudaHostAllocDefault));
  HANDLE_ERROR(cudaMalloc((void**) &dev_a, size * sizeof(*dev_a)));

  HANDLE_ERROR(cudaEventRecord(start, 0));

  for (int i = 0; i<100; i++) {
    if (up)
      HANDLE_ERROR(cudaMemcpy(dev_a, a, size * sizeof(*a), cudaMemcpyHostToDevice));
    else
      HANDLE_ERROR(cudaMemcpy(a, dev_a, size * sizeof(*a), cudaMemcpyDeviceToHost));
  }

  HANDLE_ERROR(cudaEventRecord(stop, 0));
  HANDLE_ERROR(cudaEventSynchronize(stop));
  HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));

  HANDLE_ERROR(cudaFreeHost(a));
  HANDLE_ERROR(cudaFree(dev_a));
  HANDLE_ERROR(cudaEventDestroy(start));
  HANDLE_ERROR(cudaEventDestroy(stop));

  return elapsedTime;
}
