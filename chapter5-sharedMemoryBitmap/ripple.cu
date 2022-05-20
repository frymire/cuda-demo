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
#include "../common/cpu_anim.h"

const int dim = 1024;
const float PI = 3.1415926535897932f;

__global__ void kernel(unsigned char *bitmapData, int ticks);

struct DataBlock {
  unsigned char *dev_bitmap;
  CPUAnimBitmap *bitmap;
};

// Define a function to pass to the bitmap anim_and_exit function that generates a frame to animate.
void generate_frame(DataBlock *d, int ticks) {
  dim3 blocks(dim/16, dim/16);
  dim3 threads(16, 16);
  kernel<<<blocks, threads>>>(d->dev_bitmap, ticks);
  HANDLE_ERROR(cudaMemcpy(d->bitmap->get_ptr(), d->dev_bitmap, d->bitmap->image_size(), cudaMemcpyDeviceToHost));
}

// Define a function to pass to the bitmap anim_and_exit function that frees the GPU memory.
void cleanup(DataBlock *d) { HANDLE_ERROR(cudaFree(d->dev_bitmap)); }

int main(void) {
  DataBlock data;
  CPUAnimBitmap bitmap(dim, dim, &data);
  data.bitmap = &bitmap;
  HANDLE_ERROR(cudaMalloc((void**) &data.dev_bitmap, bitmap.image_size()));
  bitmap.anim_and_exit((void(*)(void*, int)) generate_frame, (void(*)(void*)) cleanup);
}

__global__ void kernel(unsigned char *bitmapData, int ticks) {

  // Map from threadIdx/BlockIdx to pixel position.
  int x = blockDim.x*blockIdx.x + threadIdx.x;
  int y = blockDim.y*blockIdx.y + threadIdx.y;
  int offset = (gridDim.x*blockDim.x)*y + x;

  // Calculate the value at (x, y).
  float fx = x - dim/2;
  float fy = y - dim/2;
  float d = sqrtf(fx*fx + fy*fy);
  unsigned char grey = (unsigned char) (128.0f + 127.0f * cos(d/10.0f - ticks/7.0f) / (d/10.0f + 1.0f));
  bitmapData[offset*4 + 0] = grey;
  bitmapData[offset*4 + 1] = grey;
  bitmapData[offset*4 + 2] = grey;
  bitmapData[offset*4 + 3] = 255;
}
