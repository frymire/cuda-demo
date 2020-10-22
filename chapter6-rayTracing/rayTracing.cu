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

#define DIM 1024
#define rnd( x ) (x * rand() / RAND_MAX)
#define INF 2e10f
#define SPHERES 20

struct Sphere {

  float r, b, g;
  float radius;
  float x, y, z;

  __device__ float hit(float ox, float oy, float *n) {
    float dx = ox - x;
    float dy = oy - y;
    if (dx*dx + dy*dy < radius*radius) {
      float dz = sqrtf(radius*radius - dx*dx - dy*dy);
      *n = dz / sqrtf(radius*radius);
      return dz + z;
    }
    return -INF;
  }

};

__constant__ Sphere s[SPHERES];
__global__ void renderSpheres(unsigned char *bitmapData);

// Globals needed by the update routine
struct DataBlock {
  unsigned char *dev_bitmap;
};

int main(void) {

  // Capture the start time
  cudaEvent_t start, stop;
  HANDLE_ERROR(cudaEventCreate(&start));
  HANDLE_ERROR(cudaEventCreate(&stop));
  HANDLE_ERROR(cudaEventRecord(start, 0));

  DataBlock data;
  CPUBitmap bitmap(DIM, DIM, &data);
  unsigned char *dev_bitmap;

  // Allocate memory on the GPU for the output bitmap
  HANDLE_ERROR(cudaMalloc((void**) &dev_bitmap, bitmap.image_size()));

  // Allocate temp memory, initialize it, copy to constant memory on the GPU, then free our temp memory.
  Sphere *temp_s = (Sphere*) malloc(sizeof(Sphere) * SPHERES);
  for (int i = 0; i<SPHERES; i++) {
    temp_s[i].r = rnd(1.0f);
    temp_s[i].g = rnd(1.0f);
    temp_s[i].b = rnd(1.0f);
    temp_s[i].x = rnd(1000.0f) - 500;
    temp_s[i].y = rnd(1000.0f) - 500;
    temp_s[i].z = rnd(1000.0f) - 500;
    temp_s[i].radius = rnd(100.0f) + 20;
  }
  HANDLE_ERROR(cudaMemcpyToSymbol(s, temp_s, sizeof(Sphere) * SPHERES));
  free(temp_s);

  // Generate a bitmap from our sphere data.
  dim3 grids(DIM/16, DIM/16);
  dim3 threads(16, 16);
  renderSpheres<<<grids, threads>>>(dev_bitmap);

  // copy our bitmap back from the GPU for display
  HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost));

  // Get stop time, and display the timing results.
  HANDLE_ERROR(cudaEventRecord(stop, 0));
  HANDLE_ERROR(cudaEventSynchronize(stop));
  float elapsedTime;
  HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
  printf("Time to generate: %3.1f ms\n", elapsedTime);

  HANDLE_ERROR(cudaEventDestroy(start));
  HANDLE_ERROR(cudaEventDestroy(stop));
  HANDLE_ERROR(cudaFree(dev_bitmap));

  bitmap.display_and_exit();
}

__global__ void renderSpheres(unsigned char *bitmapData) {

  // Map from threadIdx/BlockIdx to pixel position.
  int x = blockDim.x*blockIdx.x + threadIdx.x;
  int y = blockDim.y*blockIdx.y + threadIdx.y;
  int offset = (gridDim.x*blockDim.x)*y + x;
  float ox = (x - DIM/2);
  float oy = (y - DIM/2);

  float r = 0, g = 0, b = 0;
  float maxZ = -INF;
  for (int i = 0; i < SPHERES; i++) {
    float n;
    float t = s[i].hit(ox, oy, &n);
    if (t > maxZ) {
      float fscale = n;
      r = s[i].r * fscale;
      g = s[i].g * fscale;
      b = s[i].b * fscale;
      maxZ = t;
    }
  }

  bitmapData[offset*4 + 0] = (int) (255*r);
  bitmapData[offset*4 + 1] = (int) (255*g);
  bitmapData[offset*4 + 2] = (int) (255*b);
  bitmapData[offset*4 + 3] = 255;
}
