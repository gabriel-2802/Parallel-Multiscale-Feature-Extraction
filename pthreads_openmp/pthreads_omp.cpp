#include "../helpers/image.h"
#include "../helpers/kernels.h"

#include "infrastructure/utils.h"
#include "infrastructure/thread_manager.h"

#include <iostream>
#include <vector>
#include <limits>
#include <thread>
#include <pthread.h>

using namespace std;

int main() {

    GreyScaleImage img("../images/image.png");

    auto input = img.getMatrix();
    auto output = allocateMatrix(img.getHeight(), img.getWidth());

    unsigned int numThreads = std::thread::hardware_concurrency();
    double globalMin, globalMax;

    for (int l = 0; l < NUM_LAYERS; ++l) {
        LAYER layer = static_cast<LAYER>(l);
        auto threadData = runConvolutionThreads(input, output, layer, numThreads);

        computeGlobalMinMax(threadData, globalMin, globalMax, numThreads);

        runNormalizationThreads(output, globalMin, globalMax, numThreads);

        std::swap(input, output);
    }

    img.setMatrix(input);
    img.save("../images/output_pthreads_omp.png");

    return 0;
}