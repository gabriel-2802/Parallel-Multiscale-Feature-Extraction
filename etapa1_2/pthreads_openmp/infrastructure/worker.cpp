#include "worker.h"
#include <pthread.h>
#include <omp.h>
#include <algorithm>
#include <cstddef>
#include <cfloat>

using namespace std;

// convolution thread function with OpenMP
void* threadRoutine(void* arg) {
    ThreadData* data = (ThreadData*)(arg);
    auto& input = *(data->input);
    auto& output = *(data->output);
    const auto& kernel = data->kernel;
    int padding = data->padding;
    double divisor = data->divisor;

    // these will store the min/max values
    // computed by this thread only
    double localMin = DBL_MAX;
    double localMax = -DBL_MAX;

    // this creates multiple OpenMP threads inside one pthread
    // the for loop is automatically divided among the OpenMP threads
    // each OpenMP gets its copy of localMin and localMax
    // at the end of the loop, OpenMP combines the results (all min values, all max values)
    // the final results are stored back in localMin and localMax
    #pragma omp parallel for reduction(min:localMin) reduction(max:localMax)
    for (int y = data->startRow; y < data->endRow; ++y) {
        // each omp thread processes multiple rows
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

            // update local min/max
            if (output[y][x] < localMin) localMin = output[y][x];
            if (output[y][x] > localMax) localMax = output[y][x];
        }
    }

    // store local min and max back to ThreadData
    data->localMin = localMin;
    data->localMax = localMax;
    
    return nullptr;
}

// normalization thread function with OpenMP
void* normalizationRoutine(void* arg)
{
    NormData* data = (NormData*)(arg);
    auto& matrix = *(data->matrix);

    double range = (data->globalMax - data->globalMin == 0.0) ? 1.0 : (data->globalMax - data->globalMin);

    // each OpenMP thread processes different rows
    // no two threads write to the same pixel
    // all rows are normalized in parallel
    #pragma omp parallel for
    for (int i = data->startRow; i < data->endRow; ++i)
    {
        auto &row = matrix[i];
        for (size_t j = 0; j < row.size(); ++j)
            row[j] = 255.0 * (row[j] - data->globalMin) / range;
    }

    return nullptr;
}
