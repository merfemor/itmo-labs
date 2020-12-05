#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <assert.h>

#define MATRIX_SIZE 500

long **allocateSquareMatrix(const unsigned int size) {
    long **const matrix = (long **const) malloc(sizeof(long **) * size);
    matrix[0] = (long *) malloc(sizeof(long) * size * size);
    for (int i = 1; i < size; ++i) {
        matrix[i] = matrix[0] + i * size;
    }
    return matrix;
}

long **multiplySquareMatricesSerial(long **const A, long **const B, const unsigned int size) {
    long **const result = allocateSquareMatrix(MATRIX_SIZE);
    for (unsigned int i = 0; i < size; ++i) {
        for (unsigned int j = 0; j < size; ++j) {
            long sum = 0;
            for (unsigned int k = 0; k < size; ++k) {
                sum += A[i][k] * B[k][j];
            }
            result[i][j] = sum;
        }
    }
    return result;
}

int main() {
    long **const matrix1 = allocateSquareMatrix(MATRIX_SIZE);
    long **const matrix2 = allocateSquareMatrix(MATRIX_SIZE);

    double start = omp_get_wtime();
    long **const result = multiplySquareMatricesSerial(matrix1, matrix2, MATRIX_SIZE);
    double elapsedTime = omp_get_wtime() - start;
    printf("Elapsed serial %f s", elapsedTime);

    return 0;
}
