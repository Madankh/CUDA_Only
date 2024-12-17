#include <cuda_runtime.h>
#include <iostream>

#define TILE_SIZE 16  // Size of each tile

// CUDA kernel for matrix multiplication with tiling
__global__ void matrixMulTiled(float *A, float *B, float *C, int N) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];  // Shared memory for tile of A
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];  // Shared memory for tile of B

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float value = 0;

    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load a tile of A and B into shared memory
        if (row < N && t * TILE_SIZE + threadIdx.x < N) {
            tileA[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0;
        }

        if (col < N && t * TILE_SIZE + threadIdx.y < N) {
            tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0;
        }

        __syncthreads();  // Synchronize to ensure all threads load their tiles

        // Compute partial product for this tile
        for (int i = 0; i < TILE_SIZE; ++i) {
            value += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];
        }

        __syncthreads();  // Synchronize before loading next tile
    }

    // Write the result to the output matrix
    if (row < N && col < N) {
        C[row * N + col] = value;
    }
}

int main() {
    int N = 1024;  // Size of the matrix (N x N)

    // Allocate host memory
    float *h_A = new float[N * N];
    float *h_B = new float[N * N];
    float *h_C = new float[N * N];

    // Initialize matrices A and B
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 1.0f;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * N * sizeof(float));
    cudaMalloc(&d_B, N * N * sizeof(float));
    cudaMalloc(&d_C, N * N * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * N * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    dim3 dimGrid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    // Launch kernel
    matrixMulTiled<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify result
    for (int i = 0; i < N * N; ++i) {
        if (h_C[i] != N) {
            std::cout << "Error at " << i << ": " << h_C[i] << "\n";
            break;
        }
    }

    std::cout << "Matrix multiplication completed successfully.\n";

    // Free memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
