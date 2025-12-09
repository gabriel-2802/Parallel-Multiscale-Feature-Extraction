#include "../helpers/image.h"
#include "../helpers/kernels.h"

#include "infrastructure/thread_manager.h"

#include <iostream>
#include <vector>
#include <limits>
#include <pthread.h>

using namespace std;

std::vector<std::vector<double>> allocateMatrix(int height, int width) {
    return std::vector<std::vector<double>>(height, std::vector<double>(width, 0.0));
}

int main() {

    GreyScaleImage img("../images/image.png");

    auto input = img.getMatrix();
    auto output = allocateMatrix(img.getHeight(), img.getWidth());

    int numThreads = 4;   // just for now

    auto threadData = runConvolutionThreads(input, output, LAYER_1_KERNEL, LAYER_1_PADDING, LAYER_1_DIV, numThreads);

    double globalMin, globalMax;
    computeGlobalMinMax(threadData, globalMin, globalMax, numThreads);

    runNormalizationThreads(output, globalMin, globalMax, numThreads);

    img.setMatrix(output);
    img.save("../images/pthreads_output.png");

    return 0;
}