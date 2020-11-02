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
#include "lock.h"

#define nBytesData 100*1024*1024
#define nData (nBytesData / sizeof(unsigned int))
#define nBins 1024

struct Entry {
  unsigned int key;
  void* value;
  Entry* next;
};

struct HashTable {
  Entry** bins;
  Entry* entryPool;
};

void initializeTable(HashTable &table);
__global__ void gpuAddEntriesToTable(HashTable table, Lock* lock, unsigned int* keys, void** values);
__device__ __host__ size_t computeHash(unsigned int key) { return key % nBins; }
void verifyTable(const HashTable &gpuTable);
void copyTableToHost(const HashTable &cpuTable, HashTable &gpuTable);
void freeTable(HashTable &table);

int main(void) {

  unsigned int* data = (unsigned int*) big_random_block(nBytesData);
  unsigned int* gpuKeys;
  void** gpuValues;
  HANDLE_ERROR(cudaMalloc((void**) &gpuKeys, nBytesData));
  HANDLE_ERROR(cudaMalloc((void**) &gpuValues, nBytesData));
  HANDLE_ERROR(cudaMemcpy(gpuKeys, data, nBytesData, cudaMemcpyHostToDevice));
  // A real implementation would copy the values to gpuValues here.

  HashTable gpuTable;
  initializeTable(gpuTable);

  Lock cpuLocks[nBins];
  Lock* gpuLocks;
  HANDLE_ERROR(cudaMalloc((void**) &gpuLocks, nBins*sizeof(Lock)));
  HANDLE_ERROR(cudaMemcpy(gpuLocks, cpuLocks, nBins*sizeof(Lock), cudaMemcpyHostToDevice));

  cudaEvent_t start, stop;
  HANDLE_ERROR(cudaEventCreate(&start));
  HANDLE_ERROR(cudaEventCreate(&stop));
  HANDLE_ERROR(cudaEventRecord(start, 0));

  gpuAddEntriesToTable<<<60, 256>>>(gpuTable, gpuLocks, gpuKeys, gpuValues);

  HANDLE_ERROR(cudaEventRecord(stop, 0));
  HANDLE_ERROR(cudaEventSynchronize(stop));
  float elapsedTime;
  HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
  printf("Time to hash: %3.1f ms\n", elapsedTime);

  verifyTable(gpuTable);

  HANDLE_ERROR(cudaEventDestroy(start));
  HANDLE_ERROR(cudaEventDestroy(stop));
  freeTable(gpuTable);
  HANDLE_ERROR(cudaFree(gpuLocks));
  HANDLE_ERROR(cudaFree(gpuKeys));
  HANDLE_ERROR(cudaFree(gpuValues));
  free(data);
  return 0;
}

void initializeTable(HashTable &table) {
  HANDLE_ERROR(cudaMalloc((void**) &table.bins, nBins*sizeof(Entry*)));
  HANDLE_ERROR(cudaMemset(table.bins, 0, nBins*sizeof(Entry*)));
  HANDLE_ERROR(cudaMalloc((void**) &table.entryPool, nData*sizeof(Entry)));
}

__global__ void gpuAddEntriesToTable(HashTable table, Lock* gpuLocks, unsigned int* keys, void** values) {

  int tid = blockDim.x*blockIdx.x + threadIdx.x;
  int stride = gridDim.x*blockDim.x;

  while (tid < nData) {
    unsigned int key = keys[tid];
    size_t hashValue = computeHash(key);
    for (int i = 0; i < 32; i++) {
      if ((tid % 32) == i) {

        Entry* p_entry = &(table.entryPool[tid]);
        p_entry->key = key;
        p_entry->value = values[tid];

        // Compare this to code missing lock() and unlock() calls.
        gpuLocks[hashValue].lock();
        p_entry->next = table.bins[hashValue];
        table.bins[hashValue] = p_entry;
        gpuLocks[hashValue].unlock();
      }
    }
    tid += stride;
  }
}

void verifyTable(const HashTable &gpuTable) {

  HashTable cpuTable;
  copyTableToHost(gpuTable, cpuTable);

  int entryCount = 0;
  for (size_t i = 0; i < nBins; i++) {
    Entry* p_entry = cpuTable.bins[i];
    while (p_entry != NULL) {
      entryCount++;
      if (computeHash(p_entry->key) != i) {
        printf("%d hashed to %ld, but was located at %ld\n", p_entry->key, computeHash(p_entry->key), i);
      }
      p_entry = p_entry->next;
    }
  }

  if (entryCount != nData) {
    printf("%d elements found in hash table. Should be %ld\n", entryCount, nData);
  } else {
    printf("All %d elements found in hash table.\n", entryCount);
  }

  free(cpuTable.entryPool);
  free(cpuTable.bins);
}

void copyTableToHost(const HashTable &gpuTable, HashTable &cpuTable) {

  cpuTable.bins = (Entry**) calloc(nBins, sizeof(Entry*));
  cpuTable.entryPool = (Entry*) malloc(nData*sizeof(Entry));

  HANDLE_ERROR(cudaMemcpy(cpuTable.bins, gpuTable.bins, nBins*sizeof(Entry*), cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(cpuTable.entryPool, gpuTable.entryPool, nData*sizeof(Entry), cudaMemcpyDeviceToHost));

  for (int i = 0; i < nBins; i++) {
    if (cpuTable.bins[i] != NULL) {
      cpuTable.bins[i] = (Entry*) ((size_t) cpuTable.bins[i] - (size_t) gpuTable.entryPool + (size_t) cpuTable.entryPool);
    }
  }

  for (int i = 0; i < nData; i++) {
    if (cpuTable.entryPool[i].next != NULL) {
      cpuTable.entryPool[i].next = (Entry*) ((size_t) cpuTable.entryPool[i].next - (size_t) gpuTable.entryPool + (size_t) cpuTable.entryPool);
    }
  }
}

void freeTable(HashTable &table) {
  HANDLE_ERROR(cudaFree(table.entryPool));
  HANDLE_ERROR(cudaFree(table.bins));
}
