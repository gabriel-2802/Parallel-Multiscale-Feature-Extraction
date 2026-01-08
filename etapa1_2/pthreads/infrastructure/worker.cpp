#include "worker.h"
#include <pthread.h>
#include <algorithm>
#include <cstddef>

using namespace std;

// convolution thread function with OpenMP
void* threadRoutine(void* arg) {
    ThreadData* data = (ThreadData*)(arg);
    auto& input = *(data->input);
    auto& output = *(data->output);
    const auto& kernel = data->kernel;
    int padding = data->padding;
    double divisor = data->divisor;

    // loop over the rows assigned to this thread
    for (int y = data->startRow; y < data->endRow; ++y) {
        for (int x = 0; x < data->width; ++x) {
            double sum = 0.0;
            // apply kernel
            for (int ky = -padding; ky <= padding; ++ky) {
                for (int kx = -padding; kx <= padding; ++kx) {
                    // handle borders by clamping
                    int iy = std::min(std::max(y + ky, 0), data->height - 1);
                    int ix = std::min(std::max(x + kx, 0), data->width - 1);
                    sum += input[iy][ix] * kernel[ky + padding][kx + padding];
                }
            }
    
            output[y][x] = sum / divisor;

            // update local min/max
            // each thread keeps track of its own local min/max
            if (output[y][x] < data->localMin) data->localMin = output[y][x];
            if (output[y][x] > data->localMax) data->localMax = output[y][x];
        }
    }
    
    return nullptr;
}

// normalization thread function with OpenMP
void* normalizationRoutine(void* arg)
{
    NormData* data = (NormData*)(arg);
    auto& matrix = *(data->matrix);
    
    double range = (data->globalMax - data->globalMin == 0.0) ? 1.0 : (data->globalMax - data->globalMin);

    // loop over the rows assigned to this thread
    for (int i = data->startRow; i < data->endRow; ++i)
    {
        auto &row = matrix[i];
        // normalize each pixel in the row to [0, 255]
        for (size_t j = 0; j < row.size(); ++j)
            row[j] = 255.0 * (row[j] - data->globalMin) / range;
    }

    return nullptr;
}