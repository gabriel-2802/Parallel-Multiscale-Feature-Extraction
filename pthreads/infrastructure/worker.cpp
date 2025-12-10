#include "worker.h"
#include <iostream>
#include <pthread.h>
#include <vector>
#include <algorithm>
#include <climits>
#include <cstddef>

using namespace std;

void* threadRoutine(void* arg) {
    ThreadData* data = (ThreadData*)(arg);
    auto& input = *(data->input);
    auto& output = *(data->output);
    const auto& kernel = data->kernel;
    int padding = data->padding;
    double divisor = data->divisor;

    for (int y = data->startRow; y < data->endRow; ++y) {
        for (int x = 0; x < data->width; ++x) {
            double sum = 0.0;
            for (int ky = -padding; ky <= padding; ++ky) {
                for (int kx = -padding; kx <= padding; ++kx) {
                    int iy = std::min(std::max(y + ky, 0), data->height - 1);
                    int ix = std::min(std::max(x + kx, 0), data->width - 1);
                    sum += input[iy][ix] * kernel[ky + padding][kx + padding];
                }
            }
            output[y][x] = sum / divisor;
            if (output[y][x] < data->localMin) data->localMin = output[y][x];
            if (output[y][x] > data->localMax) data->localMax = output[y][x];
        }
    }
    
    return nullptr;
}

void* normalizationRoutine(void* arg)
{
    NormData* data = (NormData*)(arg);
    auto& matrix = *(data->matrix);

    double range = (data->globalMax - data->globalMin == 0.0) ? 1.0 : (data->globalMax - data->globalMin);

    for (int i = data->startRow; i < data->endRow; ++i)
    {
        auto &row = matrix[i];
        for (size_t j = 0; j < row.size(); ++j)
            row[j] = 255.0 * (row[j] - data->globalMin) / range;
    }

    return nullptr;
}