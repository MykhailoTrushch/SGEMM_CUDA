#pragma once

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

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

This is a pretty much almost a copy of the naive kernel.

Firstly, what is needed to be understood is the concept of warps. Groups of threads of 32 are grouped into warps, which are then executed using the warp scheduler, a physical unit which executes instructions.
There are 4 warp schedulers per processor.
The warp schedulers support 32B, 64B, 128B memory accesses, but only when the seeked data is located sequentially in memory. In this case, the memory access is coalesced (combined) into one.
In our case, we the floats are 4B each, so we could access 32 * 4B floats using one memory access, in theory.
However, in the previous kernel, threadIdx.x and threadIdx.y are used to point to a specific location in a block, and the consecutive threads
are increased in x until the dimension is reached, and then y increases. The consecutive threads accessed a part of different rows of A, and the memory was not located sequentially there, so we did not make use of memory coalescing.

In this kernel, only threadIdx.x is used. With increase in threadIdx.x, the column increases by one, and row only increases when threadIdx.x is a multiple of BLOCKSIZE.
Let's look at the memory accesses (same as in previous kernel):

tmp += A[cRow * K + i] * B[i * N + cCol];

Here, consecutive threads access different entries in one row in B (when cCol increases by 1, we move to the right in the row by 1). Therefore, here memory coalescing is used and we will use memory instructions of larger size.
*/

template <const uint BLOCKSIZE>
__global__ void sgemm_global_mem_coalesce(int M, int N, int K, float alpha,
                                          const float *A, const float *B,
                                          float beta, float *C)
{
  const int cRow = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  const int cCol = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

  // if statement is necessary to make things work under tile quantization
  if (cRow < M && cCol < N)
  {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i)
    {
      tmp += A[cRow * K + i] * B[i * N + cCol];
    }
    C[cRow * N + cCol] = alpha * tmp + beta * C[cRow * N + cCol];
  }
}

// re-implementing from memory
template <const uint BLOCKSIZE>
__global__ void sgemm_global_mem_coalesce_reimpl(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C)
{
  const int x = blockDim.x * blockIdx.x + threadIdx.x / BLOCKSIZE;
  const int y = blockDim.y * blockIdx.y + threadIdx.x % BLOCKSIZE;
  float tmp = 0.0;
  if (x < K && y < N)
  {
    for (int i = 0; i < K; ++i)
    {
      tmp += A[x * K + i] * B[i * N + y];
    }
    C[x * N + y] = alpha * tmp + beta * C[x * N + y];
  }
}