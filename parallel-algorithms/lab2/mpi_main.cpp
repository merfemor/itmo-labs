#include <mpi.h>
#include <time.h>
#include <stdio.h>
#include <assert.h>
#include "common.h"

#define PRINT_DEBUG_MATRICES

int manager_proc(int commSize) {
    srand(RANDOM_SEED);
    double **matrix1 = allocateSquareMatrix(MATRIX_SIZE);
    fillSquareMatrixWithRandomNumbers(matrix1, MATRIX_SIZE);
    double **matrix2 = allocateSquareMatrix(MATRIX_SIZE);
    fillSquareMatrixWithRandomNumbers(matrix2, MATRIX_SIZE);

    double **expectedResult = multiplySquareMatricesSerial(matrix1, matrix2, MATRIX_SIZE);

#ifdef PRINT_DEBUG_MATRICES
    puts("Matrix 1");
    printMatrix(matrix1, MATRIX_SIZE);

    puts("\nMatrix 2");
    printMatrix(matrix2, MATRIX_SIZE);

    puts("\nSerial result");
    printMatrix(expectedResult, MATRIX_SIZE);
#endif

    for (int i = 0; i < commSize; ++i) {
        unsigned int rowFrom, rowTo, colFrom, colTo;
        countRowColRangesFromWorkerNum(i, commSize, SQUARE_ROWS, SQUARE_COLS, MATRIX_SIZE,
                                       &rowFrom, &rowTo, &colFrom, &colTo);

#ifdef PRINT_DEBUG_MATRICES
        printf("For %d, ranges rows [%d, %d], cols [%d, %d]\n", i, rowFrom, rowTo, colFrom, colTo);
#endif
    }

    return 0;
}

int worker_proc(int rank) {
    perror("Not implemented\n");
    return 0;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int commSize;
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);

    assert(WORKERS_NUM > 1);
    if (commSize != WORKERS_NUM) {
        fprintf(stderr, "Expected at lest %d processes, actual %d\n", WORKERS_NUM, commSize);
        return 1;
    }

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    assert(rank >= 0 && rank < commSize);
    printf("Started process #%d\n", rank);

    int res;
    if (rank == 0) {
        res = manager_proc(commSize);
    } else {
        res = worker_proc(rank);
    }
    MPI_Finalize();
    return res;
}