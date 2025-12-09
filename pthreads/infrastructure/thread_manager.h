#pragma once
#include "worker.h"
#include "convolution.h"
#include <vector>

std::vector<ThreadData> runConvolutionThreads(std::vector<std::vector<double>>& input,
                           std::vector<std::vector<double>>& output,
                           const std::vector<std::vector<int>>& kernel,
                           int padding, double divisor,
                           int numThreads);

void computeGlobalMinMax(const std::vector<ThreadData>& threadData,
                         double& globalMin, double& globalMax, int numThreads);

void runNormalizationThreads(std::vector<std::vector<double>>& matrix,
                             double globalMin, double globalMax,
                             int numThreads);
