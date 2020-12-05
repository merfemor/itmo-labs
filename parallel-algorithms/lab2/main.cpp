#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <assert.h>
#include <stdbool.h>
#include <time.h>
#include <math.h>

#define MATRIX_SIZE 700
#define THREADS_NUM 9
#define SQUARE_HEIGHT_WIDTH 3
#define RANDOM_SEED 13
#define DOUBLE_EQUALS_PRECISION 1e-6
//#define PRINT_DEBUG_MATRICES

double **allocateSquareMatrix(const unsigned int size) {
    double **const matrix = (double **const) malloc(sizeof(double **) * size);
    matrix[0] = (double *) malloc(sizeof(double) * size * size);
    for (int i = 1; i < size; ++i) {
        matrix[i] = matrix[0] + i * size;
    }
    return matrix;
}

double doubleRand(double from, double to) {
    int r = rand();
    double r01 = ((double) (r)) / RAND_MAX;
    return (to - from) * r01 + from;
}

void fillSquareMatrixWithRandomNumbers(double **matrix, const unsigned int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            matrix[i][j] = doubleRand(-5., 5.);
        }
    }
}

double **multiplySquareMatricesSerial(double **const A, double **const B, const unsigned int size) {
    double **const result = allocateSquareMatrix(MATRIX_SIZE);
    for (unsigned int i = 0; i < size; ++i) {
        for (unsigned int j = 0; j < size; ++j) {
            double sum = 0;
            for (unsigned int k = 0; k < size; ++k) {
                sum += A[i][k] * B[k][j];
            }
            result[i][j] = sum;
        }
    }
    return result;
}

double **multiplySquareMatricesParallelOmp(double **const A, double **const B, const unsigned int size) {
    volatile double **const result = (volatile double **const) allocateSquareMatrix(MATRIX_SIZE);
    const int areaWidthHeight = size / 3;

#pragma omp parallel num_threads(THREADS_NUM)
    {
        // determine range of indices for counting
        int threadNum = omp_get_thread_num();
        assert(threadNum >= 0 && threadNum < THREADS_NUM);
        int areaI = threadNum % SQUARE_HEIGHT_WIDTH;
        int areaJ = threadNum / SQUARE_HEIGHT_WIDTH;
        int iFrom = areaI * areaWidthHeight;
        int iTo = iFrom + areaWidthHeight;
        if (areaI == SQUARE_HEIGHT_WIDTH - 1) {
            iTo = size;
        }

        int jFrom = areaJ * areaWidthHeight;
        int jTo = jFrom + areaWidthHeight;
        if (areaJ == SQUARE_HEIGHT_WIDTH - 1) {
            jTo = size;
        }

        for (int i = iFrom; i < iTo; ++i) {
            for (int j = jFrom; j < jTo; ++j) {
                double sum = 0;
                for (int k = 0; k < size; ++k) {
                    sum += A[i][k] * B[k][j];
                }
                result[i][j] = sum;
            }
        }
    }
    return (double **) result;
}

bool areSquareMatricesEqual(double **A, double **B, const unsigned int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            if (fabs(A[i][j] - B[i][j]) > DOUBLE_EQUALS_PRECISION) {
                return false;
            }
        }
    }
    return true;
}

void printMatrix(double **matrix, const unsigned int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            printf("%f ", matrix[i][j]);
        }
        puts("");
    }
}

int main() {
    srand(RANDOM_SEED);

    double **const matrix1 = allocateSquareMatrix(MATRIX_SIZE);
    fillSquareMatrixWithRandomNumbers(matrix1, MATRIX_SIZE);
    double **const matrix2 = allocateSquareMatrix(MATRIX_SIZE);
    fillSquareMatrixWithRandomNumbers(matrix2, MATRIX_SIZE);
    double start, elapsedTime;

#ifdef PRINT_DEBUG_MATRICES
    puts("Matrix1");
    printMatrix(matrix1, MATRIX_SIZE);

    puts("\nMatrix2");
    printMatrix(matrix2, MATRIX_SIZE);
#endif

    start = omp_get_wtime();
    double **const resultSerial = multiplySquareMatricesSerial(matrix1, matrix2, MATRIX_SIZE);
    elapsedTime = omp_get_wtime() - start;

#ifdef PRINT_DEBUG_MATRICES
    puts("\nResult serial");
    printMatrix(resultSerial, MATRIX_SIZE);
#endif

    printf("Elapsed serial %f s\n", elapsedTime);

    start = omp_get_wtime();
    double **const resultOmp = multiplySquareMatricesParallelOmp(matrix1, matrix2, MATRIX_SIZE);
    elapsedTime = omp_get_wtime() - start;

#ifdef PRINT_DEBUG_MATRICES
    puts("\nResult omp");
    printMatrix(resultOmp, MATRIX_SIZE);
#endif

    assert(areSquareMatricesEqual(resultSerial, resultOmp, MATRIX_SIZE));
    printf("Elapsed parallel with OpenMP %f s\n", elapsedTime);

    return 0;
}
