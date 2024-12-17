#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1)
    sgemm2DBlocktiling(int M, int N, int K, float alpha, const float *A,
                       const float *B, float beta, float *C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  const uint totalResultsBlocktile = BM * BN;
  // A thread is responsible for calculating TM*TN elements in the blocktile
  const uint numThreadsBlocktile = totalResultsBlocktile / (TM * TN);

  // ResultsPerBlock / ResultsPerThread == ThreadsPerBlock
  assert(numThreadsBlocktile == blockDim.x);

  // BN/TN are the number of threads to span a column
  const int threadCol = threadIdx.x % (BN / TN);
  const int threadRow = threadIdx.x / (BN / TN);

  // allocate space for the current blocktile in smem
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // Move blocktile to beginning of A's row and B's column
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  // calculating the indices that this thread will load into SMEM
  const uint innerRowA = threadIdx.x / BK;
  const uint innerColA = threadIdx.x % BK;
  // calculates the number of rows of As that are being loaded in a single step
  // by a single block
  const uint strideA = numThreadsBlocktile / BK;
  const uint innerRowB = threadIdx.x / BN;
  const uint innerColB = threadIdx.x % BN;
  // for both As and Bs we want each load to span the full column-width, for
  // better GMEM coalescing (as opposed to spanning full row-width and iterating
  // across columns)
  const uint strideB = numThreadsBlocktile / BN;

  // allocate thread-local cache for results in registerfile
  float threadResults[TM * TN] = {0.0};
  // register caches for As and Bs
  float regM[TM] = {0.0};
  float regN[TN] = {0.0};

  // outer-most loop over block tiles
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // populate the SMEM caches
    for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
      As[(innerRowA + loadOffset) * BK + innerColA] =
          A[(innerRowA + loadOffset) * K + innerColA];
    }
    for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
      Bs[(innerRowB + loadOffset) * BN + innerColB] =
          B[(innerRowB + loadOffset) * N + innerColB];
    }
    __syncthreads();

    // advance blocktile
    A += BK;     // move BK columns to right
    B += BK * N; // move BK rows down

    // calculate per-thread results
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // block into registers
      for (uint i = 0; i < TM; ++i) {
        regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
      }
      for (uint i = 0; i < TN; ++i) {
        regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
      }
      for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
          threadResults[resIdxM * TN + resIdxN] +=
              regM[resIdxM] * regN[resIdxN];
        }
      }
    }
    __syncthreads();
  }

  // write out the results
  for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
    for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
      C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] =
          alpha * threadResults[resIdxM * TN + resIdxN] +
          beta * C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN];
    }
  }
}

A (4x4):
1  2  3  4
5  6  7  8
9 10 11 12
13 14 15 16

B (4x4):
17 18 19 20
21 22 23 24
25 26 27 28
29 30 31 32


// Detailed Example (threadRow = 0, threadCol = 0):

// Let's consider threadRow = 0 and threadCol = 0. This means the thread is responsible for the top-left 2x2 sub-matrix of the 4x4 output sub-matrix.

// dotIdx = 0:

// Load regM:
// regM[0] = As[(0 * 2 + 0) * 4 + 0] = As[0]
// regM[1] = As[(0 * 2 + 1) * 4 + 0] = As[4]
// Load regN:
// regN[0] = Bs[0 * 4 + 0 * 2 + 0] = Bs[0]
// regN[1] = Bs[0 * 4 + 0 * 2 + 1] = Bs[1]
// Compute partial results:
// threadResults[0 * 2 + 0] += regM[0] * regN[0] = As[0] * Bs[0]
// threadResults[0 * 2 + 1] += regM[0] * regN[1] = As[0] * Bs[1]
// threadResults[1 * 2 + 0] += regM[1] * regN[0] = As[4] * Bs[0]
// threadResults[1 * 2 + 1] += regM[1] * regN[1] = As[4] * Bs[1]
// dotIdx = 1:

// Load regM:
// regM[0] = As[(0 * 2 + 0) * 4 + 1] = As[1]
// regM[1] = As[(0 * 2 + 1) * 4 + 1] = As[5]
// Load regN:
// regN[0] = Bs[1 * 4 + 0 * 2 + 0] = Bs[4]
// regN[1] = Bs[1 * 4 + 0 * 2 + 1] = Bs[5]
// Compute partial results:
// threadResults[0] += regM[0] * regN[0] = As[1] * Bs[4]
// threadResults[1] += regM[0] * regN[1] = As[1] * Bs[5]
// threadResults[2] += regM[1] * regN[0] = As[5] * Bs[4]
// threadResults[3] += regM[1] * regN[1] = As[5] * Bs[5]
// This process continues for dotIdx = 2 and dotIdx = 3.