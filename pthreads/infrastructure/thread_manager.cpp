#include "../helpers/kernels.h"
#include "thread_manager.h"
#include <iostream>
#include <pthread.h>
#include <limits>

using namespace std;

// create pthreads for convolution
std::vector<ThreadData> runConvolutionThreads(std::vector<std::vector<double>>& input,
                           std::vector<std::vector<double>>& output,
                           LAYER layer, int numThreads) 
{
    
    std::vector<pthread_t> threads(numThreads);
    std::vector<ThreadData> threadData(numThreads);
    int height = input.size();
    int width = input[0].size();
    int rowsPerThread = height / numThreads;

    for (int i = 0; i < numThreads; ++i) {
        // select kernel parameters based on layer
        threadData[i].kernel = (layer == LAYER::ONE) ? LAYER_1_KERNEL :
                               (layer == LAYER::TWO) ? LAYER_2_KERNEL :
                               LAYER_3_KERNEL;
        threadData[i].padding = (layer == LAYER::ONE) ? LAYER_1_PADDING :
                                (layer == LAYER::TWO) ? LAYER_2_PADDING :
                                LAYER_3_PADDING;
        threadData[i].divisor = (layer == LAYER::ONE) ? LAYER_1_DIV :
                                (layer == LAYER::TWO) ? LAYER_2_DIV :
                                LAYER_3_DIV;
        // assign data to thread
        threadData[i].input = &input;
        threadData[i].output = &output;
        threadData[i].width = width;
        threadData[i].height = height;
        threadData[i].kernelSize = threadData[i].kernel.size();
        threadData[i].startRow = i * rowsPerThread;
        threadData[i].endRow = (i == numThreads - 1) ? height : (i + 1) * rowsPerThread;
        threadData[i].localMin = std::numeric_limits<double>::max();
        threadData[i].localMax = std::numeric_limits<double>::lowest();

        // launch thread
        int rc = pthread_create(&threads[i], nullptr, threadRoutine, &threadData[i]);
        if (rc != 0) {
            cerr << "pthread_create failed" << rc << endl;
        }
    }

    // wait for all threads to finish
    for (int i = 0; i < numThreads; ++i) {
        pthread_join(threads[i], nullptr);
    }

    return threadData;
}

// compute global min/max from all thread results
// All threads have a local min/max value
// We want to find the global min/max across all threads
void computeGlobalMinMax(const std::vector<ThreadData>& threadData,
                         double& globalMin, double& globalMax, int numThreads) 
{
    globalMin = std::numeric_limits<double>::max();
    globalMax = std::numeric_limits<double>::lowest();

    for (int i = 0; i < numThreads; ++i) {
        if (threadData[i].localMin < globalMin) globalMin = threadData[i].localMin;
        if (threadData[i].localMax > globalMax) globalMax = threadData[i].localMax;
    }
}

// create pthreads for normalization
void runNormalizationThreads(std::vector<std::vector<double>>& matrix,
                             double globalMin, double globalMax,
                             int numThreads) 
{
    std::vector<pthread_t> threads(numThreads);
    std::vector<NormData> normData(numThreads);
    int height = matrix.size();
    int rowsPerThread = height / numThreads;

    for (int i = 0; i < numThreads; ++i) {
        normData[i].matrix = &matrix;
        normData[i].startRow = i * rowsPerThread;
        normData[i].endRow = (i == numThreads - 1) ? height : (i + 1) * rowsPerThread;
        normData[i].globalMin = globalMin;
        normData[i].globalMax = globalMax;

        int rc = pthread_create(&threads[i], nullptr, normalizationRoutine, &normData[i]);
        if (rc != 0) {
            cerr << "pthread_create failed" << rc << endl;
        }
    }

    for (int i = 0; i < numThreads; ++i) {
        pthread_join(threads[i], nullptr);
    }
}
