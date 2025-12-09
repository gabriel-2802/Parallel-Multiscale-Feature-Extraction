#include "thread_manager.h"
#include <pthread.h>
#include <limits>

std::vector<ThreadData> runConvolutionThreads(std::vector<std::vector<double>>& input,
                           std::vector<std::vector<double>>& output,
                           const std::vector<std::vector<int>>& kernel,
                           int padding, double divisor,
                           int numThreads) 
{
    std::vector<pthread_t> threads(numThreads);
    std::vector<ThreadData> threadData(numThreads);
    int height = input.size();
    int width = input[0].size();
    int rowsPerThread = height / numThreads;

    for (int i = 0; i < numThreads; ++i) {
        threadData[i].input = &input;
        threadData[i].output = &output;
        threadData[i].kernel = kernel;
        threadData[i].width = width;
        threadData[i].height = height;
        threadData[i].kernelSize = kernel.size();
        threadData[i].padding = padding;
        threadData[i].divisor = divisor;
        threadData[i].startRow = i * rowsPerThread;
        threadData[i].endRow = (i == numThreads - 1) ? height : (i + 1) * rowsPerThread;
        threadData[i].localMin = std::numeric_limits<double>::max();
        threadData[i].localMax = std::numeric_limits<double>::lowest();

        pthread_create(&threads[i], nullptr, threadRoutine, &threadData[i]);
    }

    for (int i = 0; i < numThreads; ++i) {
        pthread_join(threads[i], nullptr);
    }

    return threadData;
}

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

        pthread_create(&threads[i], nullptr, normalizationRoutine, &normData[i]);
    }

    for (int i = 0; i < numThreads; ++i) {
        pthread_join(threads[i], nullptr);
    }
}
