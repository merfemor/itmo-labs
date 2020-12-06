#include <omp.h>
#include <assert.h>
#include <stdio.h>
#include "common.h"

#define DOUBLE_EQUALS_PRECISION 1e-6
#define PRINT_DEBUG_MATRICES


double **multiplySquareMatricesParallelOmp(double **const A, double **const B, const unsigned int size) {
    volatile double **const result = (volatile double **const) allocateMatrix(size, size);
    assert(SQUARE_ROWS * SQUARE_COLS == WORKERS_NUM);

#pragma omp parallel num_threads(WORKERS_NUM)
    {
        int threadNum = omp_get_thread_num();
        assert(threadNum >= 0 && threadNum < WORKERS_NUM);
        unsigned int rowFrom, rowTo, colFrom, colTo;
        countRowColRangesFromWorkerNum(threadNum, SQUARE_ROWS, SQUARE_COLS, size,
                                       &rowFrom, &rowTo, &colFrom, &colTo);

#ifdef PRINT_DEBUG_MATRICES
        printf("For %d, ranges rows [%d, %d], cols [%d, %d]\n", threadNum, rowFrom, rowTo, colFrom, colTo);
        fflush(stdout);
#endif


        for (unsigned int i = rowFrom; i < rowTo; ++i) {
            for (unsigned int j = colFrom; j < colTo; ++j) {
                double sum = 0;
                for (unsigned int k = 0; k < size; ++k) {
                    sum += A[i][k] * B[k][j];
                }
                result[i][j] = sum;
            }
        }
    }
    return (double **) result;
}

int main() {
    srand(RANDOM_SEED);

    double **const matrix1 = allocateMatrix(MATRIX_SIZE, MATRIX_SIZE);
    fillSquareMatrixWithRandomNumbers(matrix1, MATRIX_SIZE);
    double **const matrix2 = allocateMatrix(MATRIX_SIZE, MATRIX_SIZE);
    fillSquareMatrixWithRandomNumbers(matrix2, MATRIX_SIZE);
    double start, elapsedTime;

#ifdef PRINT_DEBUG_MATRICES
    puts("Matrix1");
    printMatrix(matrix1, MATRIX_SIZE, MATRIX_SIZE);

    puts("\nMatrix2");
    printMatrix(matrix2, MATRIX_SIZE, MATRIX_SIZE);
#endif

    start = omp_get_wtime();
    double **const resultSerial = multiplySquareMatricesSerial(matrix1, matrix2, MATRIX_SIZE);
    elapsedTime = omp_get_wtime() - start;

#ifdef PRINT_DEBUG_MATRICES
    puts("\nResult serial");
    printMatrix(resultSerial, MATRIX_SIZE, MATRIX_SIZE);
#endif

    printf("Elapsed serial %f s\n", elapsedTime);

    start = omp_get_wtime();
    double **const resultOmp = multiplySquareMatricesParallelOmp(matrix1, matrix2, MATRIX_SIZE);
    elapsedTime = omp_get_wtime() - start;

#ifdef PRINT_DEBUG_MATRICES
    puts("\nResult omp");
    printMatrix(resultOmp, MATRIX_SIZE, MATRIX_SIZE);
#endif

    assert(areMatricesEqual(resultSerial, resultOmp, MATRIX_SIZE, MATRIX_SIZE, DOUBLE_EQUALS_PRECISION));
    printf("Elapsed parallel with OpenMP %f s\n", elapsedTime);

    return 0;
}
