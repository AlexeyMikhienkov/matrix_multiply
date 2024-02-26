#include <iostream>
#include <omp.h>

#define SIZE 1024

using namespace std;


void multiplyMatrices(double** matrixA, double** matrixB, double** matrixMult) {
#pragma omp parallel for shared(matrixA, matrixB, matrixMult)
	for (int i = 0; i < SIZE; i++)
		for (int j = 0; j < SIZE; j++) {
			matrixMult[i][j] = 0.0;
			for (int k = 0; k < SIZE; k++)
				matrixMult[i][j] += matrixA[i][k] * matrixB[k][j];
		}
}

double** getMatrix() {
	double** matrix = new double* [SIZE];
	for (int i = 0; i < SIZE; i++)
		matrix[i] = new double[SIZE];

	return matrix;
}

double** fillWithRandomValues(double** matrix) {
	for (int i = 0; i < SIZE; i++)
		for (int j = 0; j < SIZE; j++)
			matrix[i][j] = rand() % 10;

	return matrix;
}

int main() {
	double** matrixA, ** matrixB, ** matrixMult, tempTime;
	int beforeCountThread, maxThreads;

	cout << "Input before_count_thread = ";
	cin >> beforeCountThread;

	maxThreads = omp_get_num_procs();
	cout << "max_threads = " << maxThreads << "\n";
	beforeCountThread = min(maxThreads, max(1, beforeCountThread));

	matrixA = fillWithRandomValues(getMatrix());
	matrixB = fillWithRandomValues(getMatrix());

	for (int countThread = 1; countThread <= beforeCountThread; countThread++) {
		matrixMult = getMatrix();
		omp_set_num_threads(countThread);

		tempTime = omp_get_wtime();
		multiplyMatrices(matrixA, matrixB, matrixMult);

		cout << "Count thread = " << countThread << ", time = " << omp_get_wtime() - tempTime << "s\n";
	}
}