
// This is a combination of all of the code in Chapter 3 of the "CUDA By Example" book,
// refactored and commented so I can understand it.

#include <stdio.h>
#include "../common/book.h"

// Get and print out some properties for a GPU
void printGPUProperties(int deviceID) {

  cudaDeviceProp prop;
  HANDLE_ERROR(cudaGetDeviceProperties(&prop, deviceID));

  printf("\n   --- General Information for device %d ---\n", deviceID);
  printf("Name: %s\n", prop.name);
  printf("Compute capability: %d.%d\n", prop.major, prop.minor);
  printf("Clock rate: %d\n", prop.clockRate);
  printf("Device copy overlap: ");
  if (prop.deviceOverlap)
    printf("Enabled\n");
  else
    printf("Disabled\n");
  printf("Kernel execution timeout : ");
  if (prop.kernelExecTimeoutEnabled)
    printf("Enabled\n");
  else
    printf("Disabled\n");

  printf("\n   --- Memory Information for device %d ---\n", deviceID);
  printf("Total global mem: %zd\n", prop.totalGlobalMem);
  printf("Total constant Mem: %zd\n", prop.totalConstMem);
  printf("Max mem pitch: %zd\n", prop.memPitch);
  printf("Texture Alignment: %zd\n", prop.textureAlignment);

  printf("\n   --- MP Information for device %d ---\n", deviceID);
  printf("Multiprocessor count: %d\n", prop.multiProcessorCount);
  printf("Shared mem per mp: %zd\n", prop.sharedMemPerBlock);
  printf("Registers per mp: %d\n", prop.regsPerBlock);
  printf("Threads in warp: %d\n", prop.warpSize);
  printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
  printf("Max thread dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
  printf("Max grid dimensions: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
  printf("\n");
}


// Return the ID of a GPU with CUDA compute capability version 1.3 or better.
int get13CompatibleGPU() {

  // Just to show that something is happening, we'll set the current device to be the second GPU. 
  // Note that this will only work if you have at least 2 GPUs. If you don't, just comment this out.
  HANDLE_ERROR(cudaSetDevice(1));

  // Get and print the ID of the current device, just to demonstrate that we can read its initial setting.
  int currentDeviceID;
  HANDLE_ERROR(cudaGetDevice(&currentDeviceID));
  printf("ID of the current CUDA device: %d\n", currentDeviceID);

  // Declare an empty device property struct that we'll use to define requirements for the tasks we want to do.  Set 
  // it to require CUDA compute capability version 1.3 or later to ensure that we can do double-precision arithmetic. 
  cudaDeviceProp requiredProperties;
  memset(&requiredProperties, 0, sizeof(cudaDeviceProp));
  requiredProperties.major = 1;
  requiredProperties.minor = 3;

  // Tell CUDA to look for a device that satisfies the required properties.
  HANDLE_ERROR(cudaChooseDevice(&currentDeviceID, &requiredProperties));
  printf("ID of CUDA device closest to revision 1.3: %d\n", currentDeviceID);

  // Return the ID of the device we found.
  return currentDeviceID;
}


// A kernel to add two integers together on the GPU
__global__ void add(int i1, int i2, int* pSum) {

  // Add the two input integers passed in, and store the result in
  // the location on the device specified by the pointer provided.
  *pSum = i1 + i2;
}

// Demonstrate that we can send two integers to the GPU, add them there, and copy the result back.
int main(void) {

  // Find out how many GPUs there are.
  int deviceCount;
  HANDLE_ERROR(cudaGetDeviceCount(&deviceCount));
  printf("\n%d device(s) found.\n\n", deviceCount);

  // Get and print out some properties for each GPU
  for (int i = 0; i < deviceCount; i++) printGPUProperties(i);

  // Try to find a GPU with CUDA compute capability version 1.3, and set it as the current GPU.
  int deviceID = get13CompatibleGPU();
  HANDLE_ERROR(cudaSetDevice(deviceID));

  // Declare a pointer to an int that we'll use to reference the result on the device
  int* devpSum;

  // Try to allocate an int on the GPU, and set the pointer to reference it
  HANDLE_ERROR(cudaMalloc((void**)&devpSum, sizeof(int)));

  // Call the kernel to add 2 + 7 on the GPU
  add << <1, 1 >> > (2, 7, devpSum);

  // NOTE: One mustn't deference a pointer allocated with cudaMalloc on the host side.  That is, don't do this...
  //int badMove = *devpSum;
  // The code will hang at runtime, because the host is trying to dereference an address in the device's memory.
  // We have to use cudaMemcpy to transfer between the contents of device memory back to the host.	
  // Declare a host variable to store the result, and copy the contents of the device pointer to it.
  int sum;
  HANDLE_ERROR(cudaMemcpy(&sum, devpSum, sizeof(int), cudaMemcpyDeviceToHost));

  // Now that we have the result back on the host, we can access and print it
  printf("\n2 + 7 = %d\n\n", sum);

  // Free the memory that we allocated dynamically on the GPU
  cudaFree(devpSum);

  return 0;
}
