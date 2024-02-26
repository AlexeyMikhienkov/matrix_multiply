#include <cstdio>
#include "mpi.h"
#include <iostream>

#define SIZE 2048

using namespace std;

double* multiplyMatrices(double* matrixA, double* matrixB, int rank, int maxRank) {
    double* matrixMult;
    matrixMult = new double[SIZE * SIZE];

    for (int i = 0; i < SIZE; i++)
        for (int j = rank; j < SIZE; j += maxRank) {
            matrixMult[i * SIZE + j] = 0.0;
            for (int k = 0; k < SIZE; k++)
                matrixMult[i * SIZE + j] += matrixA[i * SIZE + k] * matrixB[k * SIZE + j];
        }

    return matrixMult;
}

int main(int *argc, char **argv)
{
    int tasks, rank;
    double* matrixA, * matrixB, * matrixMult, * matrixResult, tempTime;

    matrixA = new double[SIZE * SIZE];
    matrixB = new double[SIZE * SIZE];
    matrixResult = new double[SIZE * SIZE];

    MPI_Init(argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &tasks);

    if (rank == 0) {
        for (int i = 0; i < SIZE; i++)
            for (int j = 0; j < SIZE; j++)
                matrixA[i * SIZE + j] = matrixB[i * SIZE + j] = 2;

        tempTime = MPI_Wtime();

        for (int tempRank = 1; tempRank < tasks; tempRank++) {
            MPI_Send(matrixA, SIZE * SIZE, MPI_DOUBLE, tempRank, 1, MPI_COMM_WORLD);
            MPI_Send(matrixB, SIZE * SIZE, MPI_DOUBLE, tempRank, 2, MPI_COMM_WORLD);
        }
    }
    else {
        MPI_Recv(matrixA, SIZE * SIZE, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(matrixB, SIZE * SIZE, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    matrixMult = multiplyMatrices(matrixA, matrixB, rank, tasks);
    MPI_Reduce(matrixMult, matrixResult, SIZE * SIZE, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0)
        cout << "Count thread = " << tasks << ", time = " << MPI_Wtime() - tempTime << "s\n";

    MPI_Finalize();

    return 0;
}

