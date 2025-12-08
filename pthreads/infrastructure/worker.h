#pragma once

#include <vector>
#include <pthread.h>

struct ThreadData {
    std::vector<std::vector<double>>* input;  
    std::vector<std::vector<double>>* output;

    std::vector<std::vector<int>> kernel;

    int width;
    int height;
    int kernelSize;
    int padding;

    double divisor;

    int startRow;
    int endRow;
};

void* threadRoutine(void* arg);
