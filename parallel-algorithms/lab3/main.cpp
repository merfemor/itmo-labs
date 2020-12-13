#include <iostream>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <omp.h>

#define CLUSTERS_NUM 5
#define DATASET_FILE_PATH "/Users/merfemor/Downloads/birch3_frist10.txt"
#define MAX_ITERATIONS 1000000
#define RANDOM_SEED 13

using namespace std;


struct Point {
    double x;
    double y;
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
        Point p{};
        p.x = x;
        p.y = y;
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
    vector<double> *avgX = new vector<double>(CLUSTERS_NUM, 0);
    vector<double> *avgY = new vector<double>(CLUSTERS_NUM, 0);
    for (Point p : *pPoints) {
        (*avgX)[p.cluster] += p.x;
        (*avgY)[p.cluster] += p.y;
    }
    for (int i = 0; i < CLUSTERS_NUM; ++i) {
        (*pCentroids)[i].x = (*avgX)[i] / CLUSTERS_NUM;
        (*pCentroids)[i].y = (*avgY)[i] / CLUSTERS_NUM;
    }
    delete avgX;
    delete avgY;
}

void printClusters(vector<Point> *pPoints) {
    puts("Clusters dump:");
    for (unsigned int i = 0; i < pPoints->size(); ++i) {
        printf("%d: %f %f cluster %d\n", i, (*pPoints)[i].x, (*pPoints)[i].y, (*pPoints)[i].cluster);
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

unsigned int reassignClusters(vector<Point> *pPoints, vector<Centroid> *pCentroids) {
    unsigned int changes = 0;
    for (Point &p : *pPoints) {
        Centroid &currentCentroid = (*pCentroids)[p.cluster];
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
        int changes = reassignClusters(pPoints, pCentroids);
        if (changes == 0) {
            break;
        }
    }
    delete pCentroids;
    return iterations;
}

int main() {
    vector<Point> *dataSet = readDataSetFromFile(DATASET_FILE_PATH);
    unsigned long size = dataSet->size();
    printf("Size of dataset is %ld\n", size);
    srand(RANDOM_SEED);
    randomilyAssignClusters(dataSet);

    puts("Initial clusters:");
    printClusters(dataSet);

    double start = omp_get_wtime();
    unsigned int iterations = kmeansSerial(dataSet);
    double end = omp_get_wtime();

    printf("Result achieved, %f ms, %d iterations:\n", (end - start) * 1000, iterations);
    printClusters(dataSet);
    return 0;
}
