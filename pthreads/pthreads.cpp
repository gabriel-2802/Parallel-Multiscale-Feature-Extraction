#include <iostream>
#include <vector>
#include <pthread.h>

#include "../helpers/image.h"
#include "../helpers/kernels.h"

#include "infrastructure/convolution.h"
#include "infrastructure/worker.h"

using namespace std;

int main() {

    GreyScaleImage img("../images/image.png");

    auto input = img.getMatrix();
    int height = img.getHeight();
    int width = img.getWidth();

    auto output = allocateMatrix(height, width);

    int numThreads = 4;   // just for now
    vector<pthread_t> threads(numThreads);
    vector<ThreadData> threadData(numThreads);

    int rowsPerThread = height / numThreads;

    for (int i = 0; i < numThreads; ++i) {
        threadData[i].input = &input;
        threadData[i].output = &output;
        threadData[i].kernel = LAYER_1_KERNEL;
        threadData[i].width = width;
        threadData[i].height = height;
        threadData[i].kernelSize = LAYER_1_KERNEL.size();
        threadData[i].padding = LAYER_1_PADDING;
        threadData[i].divisor = LAYER_1_DIV;
        threadData[i].startRow = i * rowsPerThread;
        threadData[i].endRow = (i == numThreads - 1) ? height : (i + 1) * rowsPerThread;

        pthread_create(&threads[i], nullptr, threadRoutine, &threadData[i]);
    }

    for (int i = 0; i < numThreads; ++i) {
        pthread_join(threads[i], nullptr);
    }

    normalizeMatrix(output);

    img.setMatrix(output);
    img.save("../images/pthreads_output.png");

    return 0;
}