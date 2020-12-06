#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

#define MATRIX_SIZE 1000
#define RANDOM_SEED 13
#define WORKERS_NUM 4
#define SQUARE_ROWS 2
#define SQUARE_COLS 2


extern double **allocateMatrix(const unsigned int n, const unsigned int m) {
    double **const matrix = (double **const) malloc(sizeof(double **) * n);
    matrix[0] = (double *) malloc(sizeof(double) * n * m);
    for (unsigned int i = 1; i < n; ++i) {
        matrix[i] = matrix[0] + i * m;
    }
    return matrix;
}

static double doubleRand(double from, double to) {
    int r = rand();
    double r01 = ((double) (r)) / RAND_MAX;
    return (to - from) * r01 + from;
}

extern void fillSquareMatrixWithRandomNumbers(double **matrix, const unsigned int size) {
    for (unsigned int i = 0; i < size; ++i) {
        for (unsigned int j = 0; j < size; ++j) {
            matrix[i][j] = doubleRand(-5., 5.);
        }
    }
}

extern void printMatrix(double **matrix, const unsigned int n, const unsigned int m) {
    for (unsigned int i = 0; i < n; ++i) {
        for (unsigned int j = 0; j < m; ++j) {
            printf("%f ", matrix[i][j]);
        }
        puts("");
    }
}

extern bool areMatricesEqual(
        double **A, double **B,
        const unsigned int rows, const unsigned int cols, double precision
) {
    for (unsigned int i = 0; i < rows; ++i) {
        for (unsigned int j = 0; j < cols; ++j) {
            if (fabs(A[i][j] - B[i][j]) > precision) {
                return false;
            }
        }
    }
    return true;
}

extern double **multiplyMatricesSerial(
        double **const A, const unsigned int rowsA, const unsigned int colsA,
        double **const B, const unsigned int rowsB, const unsigned int colsB
) {
    if (colsA != rowsB) {
        assert(false);
        return NULL;
    }

    double **const result = allocateMatrix(rowsA, colsB);
    for (unsigned int i = 0; i < rowsA; ++i) {
        for (unsigned int j = 0; j < colsB; ++j) {
            double sum = 0;
            for (unsigned int k = 0; k < colsA; ++k) {
                sum += A[i][k] * B[k][j];
            }
            result[i][j] = sum;
        }
    }
    return result;
}


extern double **multiplySquareMatricesSerial(double **const A, double **const B, const unsigned int size) {
    return multiplyMatricesSerial(A, size, size, B, size, size);
}

extern void countRowColRangesFromWorkerNum(
        unsigned int myNum, unsigned int rows, unsigned int cols, unsigned int size,
        unsigned int *rowFrom, unsigned int *rowTo, unsigned int *colFrom,
        unsigned int *colTo
) {
    unsigned int areaRows = size / rows;
    unsigned int areaCols = size / cols;

    unsigned int areaRowIndex = myNum / cols;
    unsigned int areaColIndex = myNum % cols;

    *rowFrom = areaRowIndex * areaRows;
    if (areaRowIndex == rows - 1) {
        *rowTo = size;
    } else {
        *rowTo = (*rowFrom) + areaRows;
    }

    *colFrom = areaColIndex * areaCols;
    if (areaColIndex == cols - 1) {
        *colTo = size;
    } else {
        *colTo = (*colFrom) + areaCols;
    }

    assert((*rowFrom) <= size);
    assert((*rowTo) <= size);
    assert((*colFrom) <= size);
    assert((*colTo) <= size);
}