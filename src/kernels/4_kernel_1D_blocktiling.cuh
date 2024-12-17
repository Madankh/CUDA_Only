#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const int BM, const int BN, const int BK, const int TM>
__global__ void sgemm1DBlocktiling(int M, int N, int K, float alpha,
                                   const float *A, const float *B, float beta,
                                   float *C) {
  // If we flip x and y here we get ~30% less performance for large matrices.
  // The current, 30% faster configuration ensures that blocks with sequential
  // blockIDs access columns of B sequentially, while sharing the same row of A.
  // The slower configuration would share columns of A, but access into B would
  // be non-sequential. So the faster configuration has better spatial locality
  // and hence a greater L2 hit rate.
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // each warp will calculate 32*TM elements, with 32 being the columnar dim.
  const int threadCol = threadIdx.x % BN;
  const int threadRow = threadIdx.x / BN;

  // allocate space for the current blocktile in SMEM
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // Move blocktile to beginning of A's row and B's column
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  // todo: adjust this to each thread to load multiple entries and
  // better exploit the cache sizes
  assert(BM * BK == blockDim.x);
  assert(BN * BK == blockDim.x);
  const uint innerColA = threadIdx.x % BK; // warp-level GMEM coalescing
  const uint innerRowA = threadIdx.x / BK;
  const uint innerColB = threadIdx.x % BN; // warp-level GMEM coalescing
  const uint innerRowB = threadIdx.x / BN;

  // allocate thread-local cache for results in registerfile
  float threadResults[TM] = {0.0};

  // outer loop over block tiles
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // populate the SMEM caches
    As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
    Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
    __syncthreads();

    // advance blocktile
    A += BK;
    B += BK * N;

    // calculate per-thread results
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // we make the dotproduct loop the outside loop, which facilitates
      // reuse of the Bs entry, which we can cache in a tmp var.
      float tmpB = Bs[dotIdx * BN + threadCol];
      for (uint resIdx = 0; resIdx < TM; ++resIdx) {
        threadResults[resIdx] +=
            As[(threadRow * TM + resIdx) * BK + dotIdx] * tmpB;
      }
    }
    __syncthreads();
  }

  // write out the results
  for (uint resIdx = 0; resIdx < TM; ++resIdx) {
    C[(threadRow * TM + resIdx) * N + threadCol] =
        alpha * threadResults[resIdx] +
        beta * C[(threadRow * TM + resIdx) * N + threadCol];
  }
}





/*
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
We want to compute C = A * B.

Parameters:

BM = 4 (Block size for rows of A)
BN = 4 (Block size for columns of B)
BK = 4 (Block size for the shared dimension)
TM = 2 (Threads per row computation)
Simplified Kernel Code (Focus on Key Parts):


Outer Loop (dotIdx):

dotIdx = 0:
tmpB = Bs[0 * 4 + threadCol]. Let's say threadCol = 0. Then tmpB = Bs[0] = 17.
Inner Loop (resIdx):
resIdx = 0: As[(0 * 2 + 0) * 4 + 0] * tmpB = As[0] * 17 = 1 * 17 = 17. threadResults[0] becomes 17.
resIdx = 1: As[(0 * 2 + 1) * 4 + 0] * tmpB = As[4] * 17 = 5 * 17 = 85. threadResults[1] becomes 85.


dotIdx = 1:
tmpB = Bs[1 * 4 + threadCol] = Bs[4] = 21.
Inner Loop (resIdx):
resIdx = 0: As[(0 * 2 + 0) * 4 + 1] * tmpB = As[1] * 21 = 2 * 21 = 42. threadResults[0] becomes 17 + 42 = 59.
resIdx = 1: As[(0 * 2 + 1) * 4 + 1] * tmpB = As[5] * 21 = 6 * 21 = 126. threadResults[1] becomes 85 + 126 = 211.

dotIdx = 2:
tmpB = Bs[2 * 4 + threadCol] = Bs[8] = 25.
Inner Loop (resIdx):
resIdx = 0: As[(0 * 2 + 0) * 4 + 2] * tmpB = As[2] * 25 = 3 * 25 = 75. threadResults[0] becomes 59 + 75 = 134.
resIdx = 1: As[(0 * 2 + 1) * 4 + 2] * tmpB = As[6] * 25 = 7 * 25 = 175. threadResults[1] becomes 211 + 175 = 386.

dotIdx = 3:
tmpB = Bs[3 * 4 + threadCol] = Bs[12] = 29.
Inner Loop (resIdx):
resIdx = 0: As[(0 * 2 + 0) * 4 + 3] * tmpB = As[3] * 29 = 4 * 29 = 116. threadResults[0] becomes 134 + 116 = 250.
resIdx = 1: As[(0 * 2 + 1) * 4 + 3] * tmpB = As[7] * 29 = 8 * 29 = 232. threadResults[1] becomes 386 + 232 = 618.

Result for the First Row (threadRow=0):

threadResults[0] = 250 (This corresponds to C[0][0])
threadResults[1] = 618 (This corresponds to C[0][1])
If we continued this process for other threadCol values (1, 2, 3), we would calculate the rest of the first row. Then, by changing threadRow to 1, 2, and 3, we would calculate the remaining rows of C.

Key Observation:

Notice how tmpB is loaded once for each dotIdx and then reused by the two threads (because TM=2) in the inner loop (resIdx). This reuse is the core optimization
*/