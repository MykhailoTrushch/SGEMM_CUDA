#pragma once

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

This is a naive kernel implementation. A single kernal computes one element of the resulting matrix,
at location x, y in some block.

Here, threadIdx.x and threadIdx.y are used to point to a specific location in a block. The consecutive threads
are increased in x until the dimension is reached, and then y increases.
*/

__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C)
{
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  // if statement is necessary to make things work under tile quantization
  if (x < M && y < N)
  {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i)
    {
      tmp += A[x * K + i] * B[i * N + y];
    }
    // C = α*(A@B)+β*C
    C[x * N + y] = alpha * tmp + beta * C[x * N + y];
  }
}

// re-implementing from memory
__global__ void sgemm_naive_reimpl(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C)
{
  const int x = blockDim.x * blockIdx.x + threadIdx.x;
  const int y = blockDim.y * blockIdx.y + threadIdx.y;
  float tmp = 0.0;
  // basically, there is tile quantization, where there may be blocks with threads that are not used fully.
  // Some threads will not be assigned to any entry in C because all threads prior to that already cover the whole
  // matrix. Therefore, we don't need to do anything in these threads.
  if (x < K && y < N)
  {
    for (int i = 0; i < K; ++i)
    {
      tmp += A[x * K + i] * B[i * N + y];
    }
    C[x * N + y] = alpha * tmp + beta * C[x * N + y];
  }
}