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

#define DIM 1024
#define PI 3.1415926535897932f
#define MAX_TEMP 1.0f
#define MIN_TEMP 0.0001f
#define SPEED 0.25f

// Globals needed by the update routine.
struct DataBlock {

  cudaEvent_t start, stop;
  float totalTime;
  float nFrames;

  float* dev_constantImageData;
  float* dev_pingImageData; // ping-pong buffer for the image data
  float* dev_pongImageData;
  unsigned char* dev_outputBitmap;
  CPUAnimBitmap* host_animator;
};

void setConstantTemperatureData(float* pixelTemperatures);
void setInitialTemperatureData(float* pixelTemperatures);
void gpuUpdateFrame(DataBlock *d, int ticks);
__global__ void resetConstantTemperaturePixels(float* pixelTemperatures);
__global__ void computeNextFrame(float *outputFrame, bool usePing);
void gpuExitAnimation(DataBlock *d);

// Define 2-D float textures.
texture<float, 2> texConstantImage;
texture<float, 2> texPingImage;
texture<float, 2> texPongImage;

int main(void) {

  DataBlock data;
  HANDLE_ERROR(cudaEventCreate(&data.start));
  HANDLE_ERROR(cudaEventCreate(&data.stop));
  data.totalTime = 0;
  data.nFrames = 0;
  CPUAnimBitmap theAnimator(DIM, DIM, &data);
  data.host_animator = &theAnimator;

  int nImageBytes = theAnimator.image_size();
  HANDLE_ERROR(cudaMalloc((void**) &data.dev_constantImageData, nImageBytes));
  HANDLE_ERROR(cudaMalloc((void**) &data.dev_pingImageData, nImageBytes));
  HANDLE_ERROR(cudaMalloc((void**) &data.dev_pongImageData, nImageBytes));
  HANDLE_ERROR(cudaMalloc((void**) &data.dev_outputBitmap, nImageBytes));

  cudaChannelFormatDesc textureDescriptor = cudaCreateChannelDesc<float>();
  HANDLE_ERROR(cudaBindTexture2D(NULL, texConstantImage, data.dev_constantImageData, textureDescriptor, DIM, DIM, sizeof(float)*DIM));
  HANDLE_ERROR(cudaBindTexture2D(NULL, texPingImage, data.dev_pingImageData, textureDescriptor, DIM, DIM, sizeof(float)*DIM));
  HANDLE_ERROR(cudaBindTexture2D(NULL, texPongImage, data.dev_pongImageData, textureDescriptor, DIM, DIM, sizeof(float)*DIM));

  float *pixelTemperatures = (float*) malloc(nImageBytes);

  // Set temperature values that will be fixed for the whole run.
  setConstantTemperatureData(pixelTemperatures);
  HANDLE_ERROR(cudaMemcpy(data.dev_constantImageData, pixelTemperatures, nImageBytes, cudaMemcpyHostToDevice));

  // Picking up from where we left off, initialize the ping buffer to set temperature values for the frame at time 0.
  setInitialTemperatureData(pixelTemperatures);
  HANDLE_ERROR(cudaMemcpy(data.dev_pingImageData, pixelTemperatures, nImageBytes, cudaMemcpyHostToDevice));
  free(pixelTemperatures);

  // When you run it, the temperature in the constant regions does not change, 
  // while that in the region set by the setInitialTemperatureData() does.
  theAnimator.anim_and_exit( (void(*)(void*, int)) gpuUpdateFrame, (void(*)(void*)) gpuExitAnimation );
}

void setConstantTemperatureData(float* pixelTemperatures) {

  // Set a couple of block to the maximum and minimum temperature, respectively.
  for (int i = 0; i < DIM*DIM; i++) {
    pixelTemperatures[i] = 0;
    int x = i % DIM; // row major
    int y = i / DIM; // row major
    if ((x > 300) && (x < 600) && (y > 310) && (y < 601)) pixelTemperatures[i] = MAX_TEMP;
    if ((x > 400) && (x < 500) && (y > 800) && (y < 900)) pixelTemperatures[i] = MIN_TEMP;
  }

  // Set a few individual points to specific temperatures.
  pixelTemperatures[DIM*100 + 100] = (MAX_TEMP + MIN_TEMP)/2;
  pixelTemperatures[DIM*700 + 100] = MIN_TEMP;
  pixelTemperatures[DIM*300 + 300] = MIN_TEMP;
  pixelTemperatures[DIM*200 + 700] = MIN_TEMP;
}

void setInitialTemperatureData(float* pixelTemperatures) {
  for (int x = 0; x < 200; x++) {
    for (int y = 800; y < DIM; y++) {
      pixelTemperatures[DIM*y + x] = MAX_TEMP;
    }
  }
}

// Define a function to be executed for each frame update in the call to bitmap.anim_and_exit().
void gpuUpdateFrame(DataBlock *d, int ticks) {

  HANDLE_ERROR(cudaEventRecord(d->start, 0));

  dim3 blocks(DIM/16, DIM/16);
  dim3 threads(16, 16);
 
  // Define which half of the ping-pong buffer to use for the given iteration.
  volatile bool usePingAsInput = true;

  // Perform 90 iterations per screen refresh
  for (int i = 0; i < 90; i++) { 
    float *input, *output;
    if (usePingAsInput) {
      input = d->dev_pingImageData;
      output = d->dev_pongImageData;
    }
    else {
      input = d->dev_pongImageData;
      output = d->dev_pingImageData;
    }
    resetConstantTemperaturePixels<<<blocks, threads>>>(input);
    computeNextFrame<<<blocks, threads>>>(output, usePingAsInput);
    usePingAsInput = !usePingAsInput;
  }
  float_to_color<<<blocks, threads>>>(d->dev_outputBitmap, d->dev_pingImageData);

  // Copy the resulting image from the GPU back to the CPU.
  CPUAnimBitmap* host_animator = d->host_animator;
  HANDLE_ERROR(cudaMemcpy(host_animator->get_ptr(), d->dev_outputBitmap, host_animator->image_size(), cudaMemcpyDeviceToHost));

  HANDLE_ERROR(cudaEventRecord(d->stop, 0));
  HANDLE_ERROR(cudaEventSynchronize(d->stop));
  float elapsedTime;
  HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, d->start, d->stop));
  d->totalTime += elapsedTime;
  d->nFrames++;
  printf("Average time per frame: %3.1f ms\n", d->totalTime/d->nFrames);
}

__global__ void resetConstantTemperaturePixels(float *pixelTemperatures) {

  // Map from threadIdx/BlockIdx to pixel position.
  int x = blockDim.x*blockIdx.x + threadIdx.x;
  int y = blockDim.y*blockIdx.y + threadIdx.y;
  int offset = (gridDim.x*blockDim.x)*y + x;

  // Reset any pixels with a non-zero temperature in the constant texture to their original values.
  // Note that the texture isn't actually constant. We just reset the data here for each iteration.
  float t = tex2D(texConstantImage, x, y);
  if (t != 0) { pixelTemperatures[offset] = t; }
}

__global__ void computeNextFrame(float *outputTemperatures, bool usePing) {

  // Map from threadIdx/BlockIdx to pixel position.
  int x = blockDim.x*blockIdx.x + threadIdx.x;
  int y = blockDim.y*blockIdx.y + threadIdx.y;
  int offset = (gridDim.x*blockDim.x)*y + x;

  // Get the neighboring values from the appropriate texture.
  float up, left, middle, right, down;
  if (usePing) {
    up = tex2D(texPingImage, x, y - 1);
    left = tex2D(texPingImage, x - 1, y);
    middle = tex2D(texPingImage, x, y);
    right = tex2D(texPingImage, x + 1, y);
    down = tex2D(texPingImage, x, y + 1);
  }
  else {
    up = tex2D(texPongImage, x, y - 1);
    left = tex2D(texPongImage, x - 1, y);
    middle = tex2D(texPongImage, x, y);
    right = tex2D(texPongImage, x + 1, y);
    down = tex2D(texPongImage, x, y + 1);
  }

  // Compute the new temperature for the current pixel.
  outputTemperatures[offset] = middle + SPEED*(up + down + left + right - 4*middle);
}

// Define a function to clean up GPU memory at the end of the call to bitmap.anim_and_exit().
void gpuExitAnimation(DataBlock *d) {
  cudaUnbindTexture(texPingImage);
  cudaUnbindTexture(texPongImage);
  cudaUnbindTexture(texConstantImage);
  HANDLE_ERROR(cudaFree(d->dev_pingImageData));
  HANDLE_ERROR(cudaFree(d->dev_pongImageData));
  HANDLE_ERROR(cudaFree(d->dev_constantImageData));
  HANDLE_ERROR(cudaEventDestroy(d->start));
  HANDLE_ERROR(cudaEventDestroy(d->stop));
}
