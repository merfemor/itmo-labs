#include <mpi.h>
#include <time.h>
#include <stdio.h>
#include <assert.h>
#include "common.h"

#define MANAGER_PROCESS_RANK 0
#define NO_TAG 0
#define DOUBLE_EQUALS_PRECISION 1e-6
#define USE_DERIVED_DTYPES

struct Config {
    bool useDerivedDtypes;
};

struct Config CFG;

void printLogPrefix(int rank) {
    printf("[%d] ", rank);
}

void managerProcessCreateMatrices(double ***pMatrix1, double ***pMatrix2) {
    double **matrix1 = allocateMatrix(MATRIX_SIZE, MATRIX_SIZE);
    fillSquareMatrixWithRandomNumbers(matrix1, MATRIX_SIZE);
    double **matrix2 = allocateMatrix(MATRIX_SIZE, MATRIX_SIZE);
    fillSquareMatrixWithRandomNumbers(matrix2, MATRIX_SIZE);

    *pMatrix1 = matrix1;
    *pMatrix2 = matrix2;
}


int managerProcessSendMatrixParts(const int commSize, double **const matrix1, double **const matrix2) {
    int err;
    double *buf = NULL;
    if (!CFG.useDerivedDtypes) {
        buf = (double *) malloc(sizeof(double) * MATRIX_SIZE);
    }
    for (int procRank = 1; procRank < commSize; ++procRank) {
        unsigned int rowFrom, rowTo, colFrom, colTo;
        unsigned int workerNum = procRank - 1; // do not count MANAGER_PROCESS_RANK
        countRowColRangesFromWorkerNum(workerNum, SQUARE_ROWS, SQUARE_COLS, MATRIX_SIZE,
                                       &rowFrom, &rowTo, &colFrom, &colTo);

        // should send A[rFrom..rTo][..], B[..][colFrom..colTo]
        err = MPI_Send(matrix1[rowFrom], MATRIX_SIZE * (rowTo - rowFrom), MPI_DOUBLE,
                       procRank, NO_TAG, MPI_COMM_WORLD);
        if (err) {
            perror("Failed to send matrix 1 parts\n");
            return err;
        }

        if (CFG.useDerivedDtypes) {
            MPI_Datatype matrix2PartType;
            err = MPI_Type_vector(MATRIX_SIZE, colTo - colFrom, MATRIX_SIZE, MPI_DOUBLE, &matrix2PartType);
            if (err) {
                perror("Failed to init matrix 2 part type\n");
                return err;
            }
            err = MPI_Type_commit(&matrix2PartType);
            if (err) {
                perror("Failed to commit matrix 2 part type\n");
                return err;
            }

            err = MPI_Send(&(matrix2[0][colFrom]), 1, matrix2PartType, procRank, NO_TAG, MPI_COMM_WORLD);
            if (err) {
                perror("Failed to send matrix 2 parts\n");
                return err;
            }
            MPI_Type_free(&matrix2PartType);
        } else {
            for (unsigned int c = colFrom; c < colTo; ++c) {
                for (unsigned int i = 0; i < MATRIX_SIZE; ++i) {
                    buf[i] = matrix2[i][c];
                }
                err = MPI_Send(buf, MATRIX_SIZE, MPI_DOUBLE, procRank, NO_TAG, MPI_COMM_WORLD);
                if (err) {
                    perror("Failed to send matrix 2 parts\n");
                    return err;
                }
            }
        }
    }
    if (buf != NULL) {
        free(buf);
    }
    return 0;
}

int managerProcessReceiveMatrixParts(const unsigned int commSize, double ***pResult) {
    int err;
    MPI_Status status;
    double **result = allocateMatrix(MATRIX_SIZE, MATRIX_SIZE);

    double *buf = NULL;
    if (!CFG.useDerivedDtypes) {
        buf = (double *) malloc(sizeof(double) * MATRIX_SIZE * MATRIX_SIZE);
    }

    unsigned int remainedProcess = commSize - 1;

    while (remainedProcess > 0) {
        err = MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        if (err) {
            perror("Failed to probe\n");
            return err;
        }
        const int procRank = status.MPI_SOURCE;

        unsigned int rowFrom, rowTo, colFrom, colTo;
        unsigned int workerNum = procRank - 1; // do not count MANAGER_PROCESS_RANK
        countRowColRangesFromWorkerNum(workerNum, SQUARE_ROWS, SQUARE_COLS, MATRIX_SIZE,
                                       &rowFrom, &rowTo, &colFrom, &colTo);

        unsigned int rows = rowTo - rowFrom, cols = colTo - colFrom;

        if (CFG.useDerivedDtypes) {
            MPI_Datatype type;
            err = MPI_Type_vector(rows, cols, MATRIX_SIZE, MPI_DOUBLE, &type);
            if (err) {
                perror("Failed to create type for result receive\n");
                return err;
            }
            err = MPI_Type_commit(&type);
            if (err) {
                perror("Failed to commit type for result receive\n");
                return err;
            }
            err = MPI_Recv(&(result[rowFrom][colFrom]), 1, type, procRank, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            if (err) {
                perror("Failed to receive result part\n");
                return err;
            }
            MPI_Type_free(&type);
        } else {
            err = MPI_Recv(buf, rows * cols, MPI_DOUBLE, procRank, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            if (err) {
                perror("Failed to receive result part\n");
                return err;
            }
            for (int r = 0; r < rows; ++r) {
                for (int c = 0; c < cols; ++c) {
                    result[rowFrom + r][colFrom + c] = buf[rows * r + c];
                }
            }
        }
        printLogPrefix(MANAGER_PROCESS_RANK);
        printf("Received part from %d\n", procRank);

        remainedProcess--;
    }
    if (buf != NULL) {
        free(buf);
    }
    *pResult = result;
    return 0;
}

int managerProcessMain(int commSize) {
    int err;
    srand(RANDOM_SEED);
    double **matrix1, **matrix2;
    managerProcessCreateMatrices(&matrix1, &matrix2);

    double **expectedResult = multiplySquareMatricesSerial(matrix1, matrix2, MATRIX_SIZE);

#ifdef PRINT_DEBUG_MATRICES
    puts("Matrix 1");
    printMatrix(matrix1, MATRIX_SIZE, MATRIX_SIZE);

    puts("\nMatrix 2");
    printMatrix(matrix2, MATRIX_SIZE, MATRIX_SIZE);

    puts("\nSerial result");
    printMatrix(expectedResult, MATRIX_SIZE, MATRIX_SIZE);
    puts("");
#endif

    err = managerProcessSendMatrixParts(commSize, matrix1, matrix2);
    if (err) {
        return err;
    }
    free(matrix1);
    free(matrix2);
    matrix1 = NULL;
    matrix2 = NULL;

    double **result;
    err = managerProcessReceiveMatrixParts(commSize, &result);
    if (err) {
        return err;
    }

#ifdef PRINT_DEBUG_MATRICES
    puts("Result");
    printMatrix(result, MATRIX_SIZE, MATRIX_SIZE);
    puts("");
#endif

    assert(areMatricesEqual(expectedResult, result, MATRIX_SIZE, MATRIX_SIZE, DOUBLE_EQUALS_PRECISION));
    return 0;
}


void workerProcessCountRanges(int rank, unsigned int *pMatrix1Rows, unsigned int *pMatrix2Cols,
                              unsigned int *pColFrom, unsigned int *pColTo) {
    unsigned int rowFrom, rowTo, colFrom, colTo;
    int workerNum = rank - 1;
    countRowColRangesFromWorkerNum(workerNum, SQUARE_ROWS, SQUARE_COLS, MATRIX_SIZE,
                                   &rowFrom, &rowTo, &colFrom, &colTo);

    *pMatrix1Rows = rowTo - rowFrom;
    *pMatrix2Cols = colTo - colFrom;
    *pColFrom = colFrom;
    *pColTo = colTo;
}


int workerProcessReceiveMatrices(const int rank, double ***pMatrix1Part, const unsigned int matrix1Rows,
                                 double ***pMatrix2Part, const unsigned int matrix2Cols,
                                 const unsigned int colFrom, const unsigned int colTo) {
    int err;
    MPI_Status status;

    double **matrix1Part = allocateMatrix(matrix1Rows, MATRIX_SIZE);
    err = MPI_Recv(*matrix1Part, MATRIX_SIZE * matrix1Rows, MPI_DOUBLE, MANAGER_PROCESS_RANK, MPI_ANY_TAG,
                   MPI_COMM_WORLD,
                   &status);
    if (err) {
        perror("Failed to receive\n");
        return err;
    }

    double **matrix2Part = allocateMatrix(MATRIX_SIZE, matrix2Cols);

    if (CFG.useDerivedDtypes) {
        MPI_Datatype matrix2PartType;
        err = MPI_Type_vector(MATRIX_SIZE, colTo - colFrom, colTo - colFrom, MPI_DOUBLE, &matrix2PartType);
        if (err) {
            perror("Failed to create matrix 2 type");
            return err;
        }
        err = MPI_Type_commit(&matrix2PartType);
        if (err) {
            perror("Failed to commit matrix 2 type");
            return err;
        }

        err = MPI_Recv(matrix2Part[0], 1, matrix2PartType, MANAGER_PROCESS_RANK,
                       MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        if (err) {
            perror("Failed to receive matrix2 part\n");
            return err;
        }

        MPI_Type_free(&matrix2PartType);
    } else {
        double *buf = NULL;
        buf = (double *) malloc(sizeof(double) * MATRIX_SIZE);

        for (int c = 0; c < matrix2Cols; ++c) {

            err = MPI_Recv(buf, MATRIX_SIZE, MPI_DOUBLE, MANAGER_PROCESS_RANK,
                           MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            if (err) {
                perror("Failed to receive matrix2 part\n");
                return err;
            }
            for (int i = 0; i < MATRIX_SIZE; ++i) {
                matrix2Part[i][c] = buf[i];
            }

        }
        if (buf != NULL) {
            free(buf);
        }
    }

    *pMatrix1Part = matrix1Part;
    *pMatrix2Part = matrix2Part;
    return 0;
}


int workerProcessSendResultPart(double **const resultPart,
                                const unsigned int rows, const unsigned int cols) {
    int err;
    err = MPI_Send(resultPart[0], rows * cols, MPI_DOUBLE, MANAGER_PROCESS_RANK, NO_TAG, MPI_COMM_WORLD);
    if (err) {
        perror("Failed to send result part\n");
        return err;
    }
    return 0;
}


int workerProcessMain(int rank) {
    int err;
    unsigned int matrix1Rows, matrix2Cols, colFrom, colTo;
    workerProcessCountRanges(rank, &matrix1Rows, &matrix2Cols, &colFrom, &colTo);

    double **matrix1Part, **matrix2Part;
    err = workerProcessReceiveMatrices(rank, &matrix1Part, matrix1Rows,
                                       &matrix2Part, matrix2Cols, colFrom, colTo);
    if (err) {
        return err;
    }

    double **resultPart = multiplyMatricesSerial(matrix1Part, matrix1Rows, MATRIX_SIZE,
                                                 matrix2Part, MATRIX_SIZE, matrix2Cols);
    free(matrix1Part);
    free(matrix2Part);
    matrix1Part = NULL;
    matrix2Part = NULL;

    err = workerProcessSendResultPart(resultPart, matrix1Rows, matrix2Cols);
    return err;
}


int initStatic() {
#ifdef USE_DERIVED_DTYPES
    CFG.useDerivedDtypes = true;
#else
    CFG.useDerivedDtypes = false;
#endif
    return 0;
}

int main(int argc, char *argv[]) {
    assert(MANAGER_PROCESS_RANK == 0);
    int err;

    MPI_Init(&argc, &argv);
    int commSize;
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);

    assert(WORKERS_NUM > 1);
    if (commSize != WORKERS_NUM + 1) {
        fprintf(stderr, "Expected at %d processes, actual %d\n", WORKERS_NUM + 1, commSize);
        MPI_Finalize();
        return 1;
    }

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    assert(rank >= 0 && rank < commSize);

    printLogPrefix(rank);
    puts("Started");
    double startTime = MPI_Wtime();

    err = initStatic();
    if (err) {
        MPI_Finalize();
        return err;
    }

    if (rank == MANAGER_PROCESS_RANK) {
        err = managerProcessMain(commSize);
    } else {
        err = workerProcessMain(rank);
    }
    if (err) {
        MPI_Finalize();
        return err;
    }
    double elapsedTime = MPI_Wtime() - startTime;
    printLogPrefix(rank);
    printf("Elapsed %f s\n", elapsedTime);

    MPI_Finalize();
    return 0;
}