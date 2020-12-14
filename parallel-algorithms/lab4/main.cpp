#include <iostream>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <omp.h>

#define CLUSTERS_NUM 5
#define DATASET_FILE_PATH "/Users/merfemor/Downloads/birch3.txt"
#define MAX_ITERATIONS 1000000
#define RANDOM_SEED 13

using namespace std;


struct Point {
    const long x;
    const long y;
    unsigned int cluster;
};

struct Centroid {
    double x;
    double y;
};

vector<Point> *readDataSetFromFile(const char *filePath) {
    FILE *file = fopen(filePath, "r");
    if (file == NULL) {
        fprintf(stderr, "Failed to read file %s\n", DATASET_FILE_PATH);
        exit(1);
    }
    vector<Point> *vec = new vector<Point>();

    long x, y;
    while (fscanf(file, "%ld %ld", &x, &y) != EOF) {
        Point p{.x = x, .y = y, 0};
        vec->push_back(p);
    }
    fclose(file);
    return vec;
}


void randomilyAssignClusters(vector<Point> *pPoints) {
    for (Point &pPoint : *pPoints) {
        pPoint.cluster = rand() % CLUSTERS_NUM;
        assert(0 <= pPoint.cluster && pPoint.cluster < CLUSTERS_NUM);
    }
}

void countCentroids(vector<Centroid> *pCentroids, vector<Point> *pPoints) {
    vector<long> *avgX = new vector<long>(CLUSTERS_NUM, 0);
    vector<long> *avgY = new vector<long>(CLUSTERS_NUM, 0);
    vector<unsigned long> *pointsNum = new vector<unsigned long>(CLUSTERS_NUM, 0);
    for (Point &p : *pPoints) {
        (*avgX)[p.cluster] += p.x;
        (*avgY)[p.cluster] += p.y;
        (*pointsNum)[p.cluster]++;
    }
    for (int i = 0; i < CLUSTERS_NUM; ++i) {
        (*pCentroids)[i].x = ((double) (*avgX)[i]) / (*pointsNum)[i];
        (*pCentroids)[i].y = ((double) (*avgY)[i]) / (*pointsNum)[i];
    }
    delete avgX;
    delete avgY;
    delete pointsNum;
}

void countCentroidsParallel(vector<Centroid> *pCentroids, vector<Point> *pPoints, int threadsNum) {
    assert(threadsNum > 1);

    // TODO: consider parallelize this loop too?
    for (unsigned int i = 0; i < CLUSTERS_NUM; ++i) {
        Centroid &c = (*pCentroids)[i];

        long sumX = 0;
        long sumY = 0;
        unsigned long points = 0;
#pragma omp parallel for num_threads(threadsNum) reduction(+:sumX, sumY, points) if (threadsNum > 1)
        for (unsigned int j = 0; j < pPoints->size(); j++) {
            const Point &p = (*pPoints)[j];
            if (p.cluster != i) {
                continue;
            }
            sumX += p.x;
            sumY += p.y;
            points++;
        }
        c.x = ((double) sumX) / points;
        c.y = ((double) sumY) / points;
    }
}

void printClusters(vector<Point> *pPoints) {
    puts("Clusters dump:");
    for (unsigned int i = 0; i < pPoints->size(); ++i) {
        printf("%d: %ld %ld cluster %d\n", i, (*pPoints)[i].x, (*pPoints)[i].y, (*pPoints)[i].cluster);
    }
    puts("");
}

void printCentroids(vector<Centroid> *pCentroids) {
    puts("Centroids dump:");
    for (unsigned int i = 0; i < pCentroids->size(); i++) {
        printf("Centroid of %d cluster: %f %f\n", i, (*pCentroids)[i].x, (*pCentroids)[i].y);
    }
    puts("");
}

double distanceEuclideanSquared(double x1, double y1, double x2, double y2) {
    return pow(x2 - x1, 2) + pow(y2 - y1, 2);
}

unsigned int reassignClusters(vector<Point> *pPoints, vector<Centroid> *pCentroids, unsigned int threadsNum) {
    assert(threadsNum > 0);
    unsigned int changes = 0;
#pragma omp parallel for num_threads(threadsNum) if (threadsNum > 1)
    for (unsigned int i = 0; i < pPoints->size(); i++) {
        Point &p = (*pPoints)[i];
        const Centroid &currentCentroid = (*pCentroids)[p.cluster];
        double bestDistance = distanceEuclideanSquared(p.x, p.y, currentCentroid.x, currentCentroid.y);
        unsigned int bestClusterNum = p.cluster;
        for (unsigned int i = 0; i < pCentroids->size(); ++i) {
            if (i == p.cluster) {
                continue;
            }
            Centroid &c = (*pCentroids)[i];
            double curDistance = distanceEuclideanSquared(p.x, p.y, c.x, c.y);
            if (curDistance < bestDistance) {
                bestDistance = curDistance;
                bestClusterNum = i;
            }
        }
        if (p.cluster != bestClusterNum) {
            p.cluster = bestClusterNum;
            changes++;
        }
    }
    return changes;
}

unsigned int kmeansSerial(vector<Point> *pPoints) {
    vector<Centroid> *pCentroids = new vector<Centroid>(CLUSTERS_NUM, Centroid{});
    unsigned int iterations = 0;
    for (; iterations < MAX_ITERATIONS; iterations++) {
        countCentroids(pCentroids, pPoints);
        int changes = reassignClusters(pPoints, pCentroids, 1);
        if (changes == 0) {
            break;
        }
    }
    delete pCentroids;
    return iterations;
}

unsigned int kmeansParallel(vector<Point> *pPoints, unsigned int threadsNum) {
    vector<Centroid> *pCentroids = new vector<Centroid>(CLUSTERS_NUM, Centroid{});
    unsigned int iterations = 0;
    for (; iterations < MAX_ITERATIONS; iterations++) {
        countCentroidsParallel(pCentroids, pPoints, threadsNum);
        int changes = reassignClusters(pPoints, pCentroids, threadsNum);
        if (changes == 0) {
            break;
        }
    }
    delete pCentroids;
    return iterations;
}

bool areEqual(vector<Point> *expected, vector<Point> *actual) {
    if (expected->size() != actual->size()) {
        return false;
    }
    for (unsigned int i = 0; i < expected->size(); ++i) {
        Point &pe = (*expected)[i];
        Point &pa = (*actual)[i];
        if (pe.cluster != pa.cluster) {
            return false;
        }
    }
    return true;
}

int main() {
    vector<Point> *originalDataSet = readDataSetFromFile(DATASET_FILE_PATH);
    unsigned long size = originalDataSet->size();
    printf("Size of dataset is %ld\n", size);
    srand(RANDOM_SEED);
    randomilyAssignClusters(originalDataSet);

//    puts("Initial clusters:");
//    printClusters(originalDataSet);

    puts("Threads,Time (ms),Speedup,Iterations");
    vector<Point> *pSerialResult = new vector<Point>(*originalDataSet);
    double startSerial = omp_get_wtime();
    unsigned int iterations = kmeansSerial(pSerialResult);
    double serialTime = omp_get_wtime() - startSerial;

//    printClusters(dataSet);
    printf("1,%f,-,%d\n", serialTime * 1000, iterations);

    int maxThreads = omp_get_max_threads();
    for (int threadsNum = 2; threadsNum <= maxThreads; ++threadsNum) {
        vector<Point> *dataSet = new vector<Point>(*originalDataSet);
        double start = omp_get_wtime();
        unsigned int iterations = kmeansParallel(dataSet, threadsNum);
        double end = omp_get_wtime();
        double speedup = serialTime / (end - start);
        printf("%d,%f,%f,%d\n", threadsNum, (end - start) * 1000, speedup, iterations);

//        puts("Result:");
//        printClusters(dataSet);
        assert(areEqual(pSerialResult, dataSet));
        delete dataSet;
    }
    return 0;
}
