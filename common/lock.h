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

#ifndef __LOCK_H__
#define __LOCK_H__

struct Lock {

  int* mutex;

  Lock(void) {
    HANDLE_ERROR(cudaMalloc((void**) &mutex, sizeof(int)));
    HANDLE_ERROR(cudaMemset(mutex, 0, sizeof(int)));
  }

  ~Lock(void) { cudaFree(mutex); }

  // Spin until an atomic compare-and-swap call returns a 0, indicating that no one else has a lock.
  __device__ void lock(void) {
    while (atomicCAS(mutex, 0, 1) != 0);
    __threadfence();
  }

  __device__ void unlock(void) { 
    __threadfence();
    atomicExch(mutex, 0); 
  }

  /* 
     See section B.5 in the CUDA C++ Programming Guide for an explanation of __threadfence(). Essentially, 
     this function enforces the ordering of memory reads and writes across threads above the level of the 
     streaming multi-processor (i.e. block-level) registers, including shared memory, global memory, page-locked 
     (pinned) host memory, and memory on any other GPU. This is necessary because CUDA assumes a weakly-ordered 
     memory model at these higher levels of the memory hierarchy.
     
     The __threadfence() call guarantees:
     1) All other threads on the GPU see the memory writes prior to the call to __threadfence() as happening 
        before any memory writes after the call. This might not otherwise be the case, since updates in one
        thread can be written to higher level caches in a different order than the writes happened within the thread.
     2) All reads made by the thread before calling __threadfence() are ordered before all reads after the call.
        (Note that this second guarantee does not involve any other threads, unlike bullet #1.)

     This version of the function enforces these rules across a single GPU (i.e. in shared memory and GPU global 
     memory). On the other hand, __threadfence_block() performs the analogous function  within a single block
     (shared memory only), while __threadfence_system() works across multiple peer GPU devices.

    Essentially, __threadfence() stalls an individual thread until previous writes in the thread are cached at the 
    higher levels of the memory hierarchy. The thread can continue once that task completes without regard for the 
    state of any other thread. In contrast, __synchthreads() stalls the thread until all other threads reach the same 
    point, *and* all global and shared memory accesses made by these threads prior to the call are visible to all 
    threads in the block.

    The atomicCAS() call in the lock() function above only guarantees that the read-modify-write operation on
    mutex happens without interference from other threads. By itself, however, it does not imply synchronization
    or ordering constraints on memory operations. These latter effects are enforced by the __threadfence() call.
    (See section B.14 in the CUDA C++ Programming Guide.) In the lock() function, the __threadfence() call ensures 
    that all threads in the GPU see the new value of mutex (i.e. that it is locked) before any steps in this 
    thread subsequent to the __threadfence() call (i.e. the steps performed during the locked period). Suppose that
    thread #1 obtains a lock (L), does locked work (W), then unlocks (U). The atomicCAS() and atomicExch() call 
    prevent thread #2 from obtaining a lock until after this period. If thread #1 did not call __threadfence(), 
    however, the effects of the work done during the locked period (e.g. writes to Entries in a hash table) could be 
    seen by thread #2 as having happened before thread #1 locked or after thread #1 unlocked. (It seems like only the 
    latter case would be problematic, however.) Another possibility is that thread #2 observes the unlock event on 
    thread #1 before the lock event. In the intervening period, then, thread #2 might obtain its lock and start its 
    work. (This seems unlikely, though, since mutex is a global variable.)
  */
};

#endif
