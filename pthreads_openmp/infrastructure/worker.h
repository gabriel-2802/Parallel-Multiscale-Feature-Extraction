#pragma once

#include <vector>
#include <pthread.h>

// data structure for convolution thread
struct ThreadData {
    std::vector<std::vector<double>>* input;  
    std::vector<std::vector<double>>* output;

    std::vector<std::vector<int>> kernel;

    int width, height, kernelSize, padding;
    double divisor;
    int startRow, endRow;

    double localMin, localMax;
};

// data structure for normalization thread
struct NormData {
    std::vector<std::vector<double>>* matrix;
    int startRow, endRow;
    double globalMin, globalMax;
};

void* threadRoutine(void* arg);
void* normalizationRoutine(void* arg);
