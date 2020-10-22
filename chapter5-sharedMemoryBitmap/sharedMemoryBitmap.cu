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

#include "cuda.h"
#include "../common/book.h"
#include "../common/cpu_bitmap.h"

#define dim 1024
#define pi 3.1415926535897932f
#define period 128.0f

__global__ void kernel(unsigned char *bitmapData);

// globals needed by the update routine
struct DataBlock {
  unsigned char *dev_bitmap;
};

int main(void) {

  DataBlock data;
  CPUBitmap bitmap(dim, dim, &data);
  unsigned char *dev_bitmap;

  HANDLE_ERROR(cudaMalloc((void**) &dev_bitmap, bitmap.image_size()));
  data.dev_bitmap = dev_bitmap;

  dim3 nBlocks(dim/16, dim/16);
  dim3 nThreads(16, 16);
  kernel<<<nBlocks, nThreads>>>(dev_bitmap);

  HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaFree(dev_bitmap));

  bitmap.display_and_exit();
}

__global__ void kernel(unsigned char *bitmapData) {

  // Map from threadIdx/BlockIdx to pixel position.
  int x = blockDim.x*blockIdx.x + threadIdx.x;
  int y = blockDim.y*blockIdx.y + threadIdx.y;
  int offset = (blockDim.x*gridDim.x)*y + x;

  __shared__ float shared[16][16];

  // now calculate the value at that position
  shared[threadIdx.x][threadIdx.y] =
    255 * (sinf(x*2.0f*pi/ period) + 1.0f) * (sinf(y*2.0f*pi/ period) + 1.0f) / 4.0f;

  // NOTE: Comment this out to see the importance of thread synchronization.
  __syncthreads();

  bitmapData[offset*4 + 0] = shared[15 - threadIdx.x][15 - threadIdx.y] / 2;
  bitmapData[offset*4 + 1] = 0;
  bitmapData[offset*4 + 2] = shared[15 - threadIdx.x][15 - threadIdx.y];
  bitmapData[offset*4 + 3] = 255;
}
