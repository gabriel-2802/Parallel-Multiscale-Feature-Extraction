#pragma once
#include "worker.h"
#include "utils.h"
#include <vector>

std::vector<ThreadData> runConvolutionThreads(std::vector<std::vector<double>>& input,
                           std::vector<std::vector<double>>& output,
                           LAYER layer, int numThreads);

void computeGlobalMinMax(const std::vector<ThreadData>& threadData,
                         double& globalMin, double& globalMax, int numThreads);

void runNormalizationThreads(std::vector<std::vector<double>>& matrix,
                             double globalMin, double globalMax,
                             int numThreads);
