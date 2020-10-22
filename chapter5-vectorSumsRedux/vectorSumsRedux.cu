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

#define N (33*1024)

__global__ void add(int *a, int *b, int *c) {
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	while (tid < N) {
		c[tid] = a[tid] + b[tid];
		tid += blockDim.x*gridDim.x;
	}
}

int main(void) {

	const int nBytes = N*sizeof(int);
	int *a, *b, *c;
	
	// Allocate the memory on the CPU.
	a = (int*) malloc(nBytes);
	b = (int*) malloc(nBytes);
	c = (int*) malloc(nBytes);

  // Fill the arrays a and b on the CPU.
  for (int i = 0; i < N; i++) {
    a[i] = i;
    b[i] = 2*i;
  }
  
  // Allocate the memory on the GPU.
  int *dev_a, *dev_b, *dev_c;
	HANDLE_ERROR(cudaMalloc((void**) &dev_a, nBytes));
	HANDLE_ERROR(cudaMalloc((void**) &dev_b, nBytes));
	HANDLE_ERROR(cudaMalloc((void**) &dev_c, nBytes));
	
	// Copy the arrays a and b to the GPU.
	HANDLE_ERROR(cudaMemcpy(dev_a, a, nBytes, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_b, b, nBytes, cudaMemcpyHostToDevice));

  // Run the kernel on the GPU.
	add<<<128, 128>>>(dev_a, dev_b, dev_c);

	// Copy the array c back from the GPU to the CPU.
	HANDLE_ERROR(cudaMemcpy(c, dev_c, nBytes, cudaMemcpyDeviceToHost));

	// Verify that the GPU did the work we requested.
	bool success = true;
	for (int i = 0; i < N; i++) {
		if ((a[i] + b[i]) != c[i]) {
			printf("Error: %d + %d != %d\n", a[i], b[i], c[i]);
			success = false;
		}
	}
	if (success) printf("We did it!\n");

	// Free the memory we allocated on the GPU.
	HANDLE_ERROR(cudaFree(dev_a));
	HANDLE_ERROR(cudaFree(dev_b));
	HANDLE_ERROR(cudaFree(dev_c));

	// Free the memory we allocated on the CPU.
	free(a);
	free(b);
	free(c);

	return 0;
}
