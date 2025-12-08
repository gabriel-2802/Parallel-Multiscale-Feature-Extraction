#include "worker.h"
#include <iostream>
#include <pthread.h>
#include <vector>
#include <algorithm>

using namespace std;

void* threadRoutine(void* arg) {
    cout << "Thread started." << endl;
    ThreadData* data = (ThreadData*)(arg);

    auto& input = *(data->input);
    auto& output = *(data->output);
    const auto& kernel = data->kernel;
    int width = data->width;
    int height = data->height;
    int padding = data->padding;
    double divisor = data->divisor;
    int startRow = data->startRow;
    int endRow = data->endRow;

    for (int y = startRow; y < endRow; ++y) {
        for (int x = 0; x < width; ++x) {
            double sum = 0.0;
            for (int ky = -padding; ky <= padding; ++ky) {
                for (int kx = -padding; kx <= padding; ++kx) {
                    int iy = std::min(std::max(y + ky, 0), height - 1);
                    int ix = std::min(std::max(x + kx, 0), width - 1);
                    sum += input[iy][ix] * kernel[ky + padding][kx + padding];
                }
            }
            output[y][x] = sum / divisor;
        }
    }
    
    return nullptr;
}