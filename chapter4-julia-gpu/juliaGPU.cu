#include <stdio.h>

#include "../common/book.h"
#include "../common/cpu_bitmap.h"

const int dim = 1000;

struct cuComplex {
  float r;
  float i;
  __device__ cuComplex(float a, float b) : r(a), i(b) {}
  __device__ float magnitude2(void) { return r * r + i * i; }
  __device__ cuComplex operator*(const cuComplex& a) { return cuComplex(r * a.r - i * a.i, i * a.r + r * a.i); }
  __device__ cuComplex operator+(const cuComplex& a) { return cuComplex(r + a.r, i + a.i); }
};

__device__ int julia(int x, int y);
__global__ void fillBitmapWithJuliaValues(unsigned char* bitmapData);

// Globals needed by the update routine
struct DataBlock { unsigned char* dev_bitmap; };

int main(void) {

  DataBlock data;
  CPUBitmap bitmap(dim, dim, &data);
  unsigned char* dev_bitmap;

  printf("Image size = %d\n", bitmap.image_size());

  HANDLE_ERROR(cudaMalloc((void**)&dev_bitmap, bitmap.image_size()));
  data.dev_bitmap = dev_bitmap;

  // Split the work over a DIM x DIM grid, with 1 thread per block. (Probably very inefficient.)
  dim3 grid(dim, dim);
  fillBitmapWithJuliaValues<<<grid, 1>>>(dev_bitmap);

  HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaFree(dev_bitmap));

  bitmap.display_and_exit();
  return 0;
}


__global__ void fillBitmapWithJuliaValues(unsigned char* bitmapData) {

  // Map from blockIdx to pixel position.
  int x = blockIdx.x;
  int y = blockIdx.y;
  int offset = gridDim.x * y + x;

  // If the point is in the Julia set, set the corresponding pixel to red, otherwise black.
  bitmapData[offset * 4 + 0] = 255 * julia(x, y);
  bitmapData[offset * 4 + 1] = 0;
  bitmapData[offset * 4 + 2] = 0;
  bitmapData[offset * 4 + 3] = 255;
}


__device__ int julia(int x, int y) {

  const float scale = 1.5;
  float jx = scale * (float)(dim / 2 - x) / (dim / 2);
  float jy = scale * (float)(dim / 2 - y) / (dim / 2);

  cuComplex c(-0.8, 0.156);
  cuComplex a(jx, jy);

  // If the sequence diverges prior to 200 iterations, return 0.
  int i = 0;
  for (i = 0; i < 200; i++) {
    a = a * a + c;
    if (a.magnitude2() > 1000) return 0;
  }

  return 1;
}
