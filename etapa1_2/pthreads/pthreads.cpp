#include "../helpers/image.h"
#include "../helpers/kernels.h"
#include "infrastructure/utils.h"
#include "infrastructure/thread_manager.h"
#include <thread>

using namespace std;

int main() {

    GreyScaleImage img("../images/upscaled_upscaled_image.png");

    auto input = img.getMatrix();
    auto output = allocateMatrix(img.getHeight(), img.getWidth());

    // number of threads = number of hardware cores
    unsigned int numThreads = std::thread::hardware_concurrency();
    double globalMin, globalMax;

    // apply each layer sequentially
    for (int l = 0; l < NUM_LAYERS; ++l) {
        // convert int to LAYER enum
        LAYER layer = static_cast<LAYER>(l);

        // convolution
        auto threadData = runConvolutionThreads(input, output, layer, numThreads);

        // compute global min and max from all threads
        computeGlobalMinMax(threadData, globalMin, globalMax, numThreads);

        // normalization
        runNormalizationThreads(output, globalMin, globalMax, numThreads);

        // output becomes input for next layer
        std::swap(input, output);
    }

    img.setMatrix(input);
    img.save("../images/output_pthreads.png");

    return 0;
}