#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

#define MATRIX_SIZE 10
#define RANDOM_SEED 13
#define WORKERS_NUM 6
#define SQUARE_ROWS 2
#define SQUARE_COLS 3

extern double **allocateSquareMatrix(const unsigned int size) {
    double **const matrix = (double **const) malloc(sizeof(double **) * size);
    matrix[0] = (double *) malloc(sizeof(double) * size * size);
    for (unsigned int i = 1; i < size; ++i) {
        matrix[i] = matrix[0] + i * size;
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

extern void printMatrix(double **matrix, const unsigned int size) {
    for (unsigned int i = 0; i < size; ++i) {
        for (unsigned int j = 0; j < size; ++j) {
            printf("%f ", matrix[i][j]);
        }
        puts("");
    }
}

extern bool areSquareMatricesEqual(double **A, double **B, const unsigned int size, double precision) {
    for (unsigned int i = 0; i < size; ++i) {
        for (unsigned int j = 0; j < size; ++j) {
            if (fabs(A[i][j] - B[i][j]) > precision) {
                return false;
            }
        }
    }
    return true;
}


extern double **multiplySquareMatricesSerial(double **const A, double **const B, const unsigned int size) {
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

extern void countRowColRangesFromWorkerNum(
        unsigned int myNum, unsigned int workersNum,
        unsigned int rows, unsigned int cols, unsigned int size,
        unsigned int *rowFrom, unsigned int *rowTo,
        unsigned int *colFrom, unsigned int *colTo
) {
    assert(myNum >= 0 && myNum < workersNum);

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