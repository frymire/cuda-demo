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

#define nBytesData (100*1024*1024)
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
  Entry* nextEntryAvailableInPool;
};

size_t computeHash(unsigned int key) { return key % nBins; }
void initializeTable(HashTable &table, int elements);
void addEntryToTable(HashTable &table, unsigned int key, void *value);
void verifyTable(const HashTable &table);
void freeTable(HashTable &table);

int main(void) {

  unsigned int* data = (unsigned int*) big_random_block(nBytesData);

  HashTable table;
  initializeTable(table, nData);

  clock_t start = clock();
  for (int i = 0; i < nData; i++) { addEntryToTable(table, data[i], (void*) NULL); }
  float elapsedTime = (float) (clock() - start) / (float) CLOCKS_PER_SEC*1000.0f;
  printf("Time to hash: %3.1f ms\n", elapsedTime);
  
  verifyTable(table);

  freeTable(table);
  free(data);
  return 0;
}

void initializeTable(HashTable &table, int nElements) {
  table.bins = (Entry**) calloc(nBins, sizeof(Entry*));
  table.entryPool = (Entry*) malloc(nElements*sizeof(Entry));
  table.nextEntryAvailableInPool = table.entryPool;
}

void addEntryToTable(HashTable &table, unsigned int key, void *value) {
  Entry* p_entry = table.nextEntryAvailableInPool++;
  p_entry->key = key;
  p_entry->value = value;
  size_t hashValue = computeHash(key);
  p_entry->next = table.bins[hashValue];
  table.bins[hashValue] = p_entry;
}

void verifyTable(const HashTable &table) {

  int entryCount = 0;

  for (size_t i = 0; i < nBins; i++) {

    Entry* pCurrentEntry = table.bins[i];
    while (pCurrentEntry != NULL) {
      entryCount++;
      if (computeHash(pCurrentEntry->key) != i) {
        printf("%d hashed to %ld, but was located at %ld\n", pCurrentEntry->key, computeHash(pCurrentEntry->key), i);
      }
      pCurrentEntry = pCurrentEntry->next;
    }
  }

  if (entryCount != nData) {
    printf("%d elements found in hash table.  Should be %ld\n", entryCount, nData);
  } else {
    printf("All %d elements found in hash table.\n", entryCount);
  }
}

void freeTable(HashTable &table) {
  free(table.bins);
  free(table.entryPool);
}
