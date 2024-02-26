#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define SIZE 2048


__global__ void multiplyMatrices(double* matrixA, double* matrixB, double* matrixMult) {
    double value;

    int initialRow = blockIdx.x, countBlocks = gridDim.x;
    int initialColumn = threadIdx.x, countThread = blockDim.x;

    for (int i = initialRow; i < SIZE; i += countBlocks)
        for (int j = initialColumn; j < SIZE; j += countThread) {
            value = 0;

            for (int k = 0; k < SIZE; ++k)
                value += matrixA[i * SIZE + k] * matrixB[k * SIZE + j];

            matrixMult[i * SIZE + j] = value;
        }
}

int main() {
    double* matrixA, * matrixB, * matrixMult;
    int sizeInt = SIZE * SIZE * sizeof(double);

    cudaEvent_t start, stop;
    float gpuTime;

    cudaMallocManaged(&matrixA, sizeInt);
    cudaMallocManaged(&matrixB, sizeInt);
    cudaMallocManaged(&matrixMult, sizeInt);

    for (int i = 0; i < SIZE * SIZE; ++i)
        matrixA[i] = matrixB[i] = 2;

    int _blocks = 32, _threads = 1024;
    dim3 threads(_threads);
    dim3 blocks(_blocks);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    multiplyMatrices << <blocks, threads >> > (matrixA, matrixB, matrixMult);
    cudaDeviceSynchronize();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);

    printf("blocks = %i, count threads = %i, time = %f", _blocks, _threads, gpuTime);

    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(matrixA); cudaFree(matrixB); cudaFree(matrixMult);
}