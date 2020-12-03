#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <assert.h>

#define PRECISION 1e-12

#define CRITICAL 1
#define ATOMIC 2
#define LOCK 3
#define REDUCTION 4

#define TYPES_COUNT 4

const unsigned int LOCK_TYPES[TYPES_COUNT] = {CRITICAL, ATOMIC, LOCK, REDUCTION};

#define FROM_TO_PAIRS_SIZE 7

const double FROM_TO_PAIRS[FROM_TO_PAIRS_SIZE][2] = {
        {1e-5, 1e-4},
        {1e-4, 1e-3},
        {1e-3, 0.01},
        {0.01, 0.1},
        {0.1,  1},
        {1,    10},
        {10,   100},
};


double function(const double x) {
    const double sinRes = sin(1 / x);
    return 1 / (x * x) * sinRes * sinRes;
}


double getIntegralResult(double from, double to) {
    return 0.25 * (2 * (to - from) / (from * to) + sin(2 / to) - sin(2 / from));
}


double countIntegralLinear(double from, double to, unsigned int segmentsNum) {
    double segmentWidth = (to - from) / ((double) segmentsNum);
    double result = 0;
    double cur = from + segmentWidth;

    for (unsigned int i = 1; i < segmentsNum - 1; ++i) {
        result += function(cur);
        cur += segmentWidth;
    }

    result = segmentWidth * (result + (function(from) + function(to)) / 2);
    return result;
}


double countIntegralParallelCritical(
        const double from,
        const double to,
        const unsigned int segmentsNum,
        const unsigned int threadsNum
) {
    const double segmentWidth = (to - from) / ((double) segmentsNum);
    double result = 0;

    omp_set_num_threads(threadsNum);
#pragma omp parallel for
    for (unsigned int i = 1; i < segmentsNum - 1; ++i) {
        double cur = from + i * segmentWidth;
        double f = function(cur);
#pragma omp critical
        result += f;
    }

    result = segmentWidth * (result + (function(from) + function(to)) / 2);
    return result;
}


double countIntegralParallelAtomic(
        const double from,
        const double to,
        const unsigned int segmentsNum,
        const unsigned int threadsNum
) {
    const double segmentWidth = (to - from) / ((double) segmentsNum);
    double result = 0;

    omp_set_num_threads(threadsNum);
#pragma omp parallel for
    for (unsigned int i = 1; i < segmentsNum - 1; ++i) {
        double cur = from + i * segmentWidth;
        double f = function(cur);
#pragma omp atomic
        result += f;
    }

    result = segmentWidth * (result + (function(from) + function(to)) / 2);
    return result;
}


double countIntegralParallelLock(
        const double from,
        const double to,
        const unsigned int segmentsNum,
        const unsigned int threadsNum
) {
    const double segmentWidth = (to - from) / ((double) segmentsNum);
    double result = 0;
    omp_lock_t lock;
    omp_init_lock(&lock);

    omp_set_num_threads(threadsNum);
#pragma omp parallel for
    for (unsigned int i = 1; i < segmentsNum - 1; ++i) {
        double cur = from + i * segmentWidth;
        double f = function(cur);
        omp_set_lock(&lock);
        result += f;
        omp_unset_lock(&lock);
    }
    omp_destroy_lock(&lock);

    result = segmentWidth * (result + (function(from) + function(to)) / 2);
    return result;
}


double countIntegralParallelReduction(
        const double from,
        const double to,
        const unsigned int segmentsNum,
        const unsigned int threadsNum
) {
    const double segmentWidth = (to - from) / ((double) segmentsNum);
    double result = 0;

    omp_set_num_threads(threadsNum);
#pragma omp parallel for reduction(+:result)
    for (unsigned int i = 1; i < segmentsNum - 1; ++i) {
        double cur = from + i * segmentWidth;
        double f = function(cur);
        result += f;
    }

    result = segmentWidth * (result + (function(from) + function(to)) / 2);
    return result;
}


unsigned int segmentsNumSearchFunction(const unsigned int i) {
    return i * i;
}


unsigned int determineOptimalSegmentsNum(double from, double to) {
    unsigned int segmentsNumI = 1;
    unsigned int segmentsNum = segmentsNumSearchFunction(segmentsNumI);

    double prevResult = countIntegralLinear(from, to, segmentsNum);
    int maxThreads = omp_get_max_threads();
    while (1) {
        segmentsNumI++;
        segmentsNum = segmentsNumSearchFunction(segmentsNumI);
        double result = countIntegralParallelReduction(from, to, segmentsNum, maxThreads);
        if (fabs(result - prevResult) <= PRECISION * fabs(result)) {
            break;
        }
        prevResult = result;
    }
    return segmentsNum;
}


double measureTimeAndResult(
        const double from,
        const double to,
        const unsigned int type,
        const unsigned int threadsNum,
        const unsigned int segmentsNum,
        double *result
) {
    assert(result != NULL);
    double startTime = omp_get_wtime();
    switch (type) {
        case CRITICAL:
            *result = countIntegralParallelCritical(from, to, segmentsNum, threadsNum);
            break;
        case ATOMIC:
            *result = countIntegralParallelAtomic(from, to, segmentsNum, threadsNum);
            break;
        case LOCK:
            *result = countIntegralParallelLock(from, to, segmentsNum, threadsNum);
            break;
        case REDUCTION:
            *result = countIntegralParallelReduction(from, to, segmentsNum, threadsNum);
            break;
        default:
            assert(false); // Unknown type
            break;
    }
    double elapsedTimeMs = (omp_get_wtime() - startTime);
    return elapsedTimeMs;
}

const char *lockTypeToString(const unsigned int type) {
    switch (type) {
        case CRITICAL:
            return "critical";
        case ATOMIC:
            return "atomic";
        case LOCK:
            return "lock";
        case REDUCTION:
            return "reduction";
        default:
            assert(false); // Unknown type
    }
    return NULL;
}


int main() {
    printf("Precision is %g\n", PRECISION);
    for (int i = 0; i < FROM_TO_PAIRS_SIZE; ++i) {
        double from = FROM_TO_PAIRS[i][0];
        double to = FROM_TO_PAIRS[i][1];
        printf("--------------------- [%f, %f] -------------------------\n", from, to);
        double expectedResult = getIntegralResult(from, to);
        printf("Expected %f\n", expectedResult);
        double result;

        unsigned int segmentsNum;
        {
            puts("Determine optimal segments num...");
            double startTime = omp_get_wtime();
            segmentsNum = determineOptimalSegmentsNum(from, to);
            double elapsedTime = (omp_get_wtime() - startTime);
            printf("Optimal segments num is %d, founded in %f s\n", segmentsNum, elapsedTime);
        }

        printf("Threads num,Lock type,");
        for (int ti = 0; ti < TYPES_COUNT; ++ti) {
            const char *typeStr = lockTypeToString(LOCK_TYPES[ti]);
            printf("Elapsed %s (ms),", typeStr);
        }
        puts("");
        {
            double startTime = omp_get_wtime();
            result = countIntegralLinear(from, to, segmentsNum);
            double elapsedTime = (omp_get_wtime() - startTime);
            printf("-,-,%f,\n", elapsedTime * 1000);
        }
        int maxThreads = omp_get_max_threads();

        for (int threadsNum = 1; threadsNum <= maxThreads; threadsNum++) {
            printf("%d,", threadsNum);

            for (int ti = 0; ti < TYPES_COUNT; ti++) {
                unsigned int type = LOCK_TYPES[ti];
                double elapsedTime = measureTimeAndResult(from, to, type, threadsNum, segmentsNum, &result);
                printf("%f,", elapsedTime * 1000);
            }
            puts("");
        }

        puts("");
    }
    return 0;
}
