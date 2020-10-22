#include <stdlib.h>

#include "../common/book.h"

//#define N 10
#define N (32*1024)

__global__ void vectorAdd(int *a, int *b, int *c) {
	int threadID = blockIdx.x;
	while (threadID < N) {
		c[threadID] = a[threadID] + b[threadID];
		threadID += gridDim.x;
	}
}

int main(void) {

	// Allocate CPU memory.
	int *a, *b, *c;
	a = (int *)malloc(N * sizeof(int));
	b = (int *)malloc(N * sizeof(int));
	c = (int *)malloc(N * sizeof(int));

	// Fill arrays 'a' and 'b' on the CPU.
	for (int i = 0; i < N; i++) {
		a[i] = i;
		b[i] = 2 * i;
	}

	// Allocate GPU memory.
	int *dev_a, *dev_b, *dev_c;
	HANDLE_ERROR(cudaMalloc((void**)&dev_a, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_b, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_c, N * sizeof(int)));

	// Copy the arrays to the GPU.
	HANDLE_ERROR(cudaMemcpy(dev_a, a, N*sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_b, b, N*sizeof(int), cudaMemcpyHostToDevice));

	vectorAdd<<<128, 1>>> (dev_a, dev_b, dev_c);

	// copy the array 'c' back from the GPU to the CPU
	HANDLE_ERROR(cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost));

	// Display the result.
	//for (int i = 0; i < N; i++) {
	//	printf("%d + %d = %d\n", a[i], b[i], c[i]);
	//}

	// Verify the result.
	bool success = true;
	for (int i = 0; i < N; i++) {
		if ((a[i] + b[i]) != c[i]) {
			printf("Error:  %d + %d != %d\n", a[i], b[i], c[i]);
			success = false;
		}
	}
	if (success) printf("The CPU and GPU vectors match.\n");

	// Free the GPU memory.
	HANDLE_ERROR(cudaFree(dev_a));
	HANDLE_ERROR(cudaFree(dev_b));
	HANDLE_ERROR(cudaFree(dev_c));

	// Free the CPU memory.
	free(a);
	free(b);
	free(c);

	return 0;
}
