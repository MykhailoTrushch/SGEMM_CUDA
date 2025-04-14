#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

/*
Matrix sizes:
MxK * KxN = MxN

Matmul: alpha * A x B + beta * C

int M, N, K are sizes of matrices.
float alpha is 
const float* A is the matrix A (size M*K)
const float* B is the matrix B (size K*N)
float beta is the coefficient by which the existing entry in C is multiplied
float* C is the resulting matrix C

Explanation:

This kernel makes use of shared memory. Each SM has its own shared memory, on par with the global memory and L2 cache.
Shared memory is shared between the threads of a single block. It is located on chip and has much lower latency compared to global memory.

With this kernel, let's remember that the execution in GPUs is split into blocks which then consist of threads. On each block, threads can use their shared memory to speedup the computation,
and then move to the next relevant block to continue computation with the next relevant data. That is exactly what we will do.

Here, in A, we will start with some block on the very left of the matrix, and next up the blocks on the right will be taken. In B, the first block will be on the very top, and then the blocks in the bottom will be taken.

Shared memory can be allocated with __shared__. We allocate the As and Bs, so a part of A and B that will be stored in the shared memory in a block.

Then, each thread loads one element assigned to this thread to the shared memory. The threads are synced so that all threads could load their info in the shared memory and the operations finish before proceeding.

Then, each thread computes and stores in tmp the dot product using elements from the shared memory. Then, we move the pointers to A and B to the next block.

Each GPU has a different amount of shared memory, so this kernel could be executed with different values of BLOCKSIZE based on that.
*/

template <const int BLOCKSIZE>
__global__ void sgemm_shared_mem_block(int M, int N, int K, float alpha,
                                       const float *A, const float *B,
                                       float beta, float *C) {
  // the output block that we want to compute in this threadblock
  const uint cRow = blockIdx.x;
  const uint cCol = blockIdx.y;

  // allocate buffer for current block in fast shared mem
  // shared mem is shared between all threads in a block
  __shared__ float As[BLOCKSIZE * BLOCKSIZE];
  __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

  // the inner row & col that we're accessing in this thread
  const uint threadCol = threadIdx.x % BLOCKSIZE;
  const uint threadRow = threadIdx.x / BLOCKSIZE;

  // advance pointers to the starting positions
  A += cRow * BLOCKSIZE * K;                    // row=cRow, col=0
  B += cCol * BLOCKSIZE;                        // row=0, col=cCol
  C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE; // row=cRow, col=cCol

  float tmp = 0.0;
  for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
    // Have each thread load one of the elements in A & B
    // Make the threadCol (=threadIdx.x) the consecutive index
    // to allow global memory access coalescing
    As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
    Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];

    // block threads in this block until cache is fully populated
    __syncthreads();
    A += BLOCKSIZE;
    B += BLOCKSIZE * N;

    // execute the dotproduct on the currently cached block
    for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
      tmp += As[threadRow * BLOCKSIZE + dotIdx] *
             Bs[dotIdx * BLOCKSIZE + threadCol];
    }
    // need to sync again at the end, to avoid faster threads
    // fetching the next block into the cache before slower threads are done
    __syncthreads();
  }
  C[threadRow * N + threadCol] =
      alpha * tmp + beta * C[threadRow * N + threadCol];
}

// re-implementing from memory
template <const int BLOCKSIZE>
__global__ void sgemm_shared_mem_block_reimpl(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C) {
  const uint cRow = blockIdx.x;
  const uint cCol = blockIdx.y;

  __shared__ float As[BLOCKSIZE*BLOCKSIZE];
  __shared__ float Bs[BLOCKSIZE*BLOCKSIZE];

  A += cRow * BLOCKSIZE * K;
  B += cCol * BLOCKSIZE;
  C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE;

  const uint threadCol = threadIdx.x % BLOCKSIZE;
  const uint threadRow = threadIdx.x / BLOCKSIZE;

  float tmp = 0.0;
  for (int i = 0; i < K; i += BLOCKSIZE) {
    As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
    Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];
    __syncthreads();
    A += BLOCKSIZE;
    B += BLOCKSIZE * N;

    for(int dotIdx = 0; dotIdx < BLOCKSIZE; dotIdx++) {
      tmp += As[threadRow * BLOCKSIZE + dotIdx] * Bs[dotIdx * BLOCKSIZE + threadCol];
    }
    __syncthreads();
  }
  C[threadRow * N + threadCol] = alpha * tmp + beta * C[threadRow * N + threadCol];
}
