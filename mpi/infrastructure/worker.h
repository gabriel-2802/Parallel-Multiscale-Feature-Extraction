#pragma once

#include <climits>

#include "entity.h"
#include "../helpers/image.h"
#include "../helpers/kernels.h"

class Worker : public Entity {
public:
    Worker(int numtasks, int rank);
    ~Worker() override;
    void run() override;

private:
    std::vector<std::vector<double>> imageSegment;
    double localMin = INT_MAX;
    double localMax = INT_MIN;

    void receive();
    void process();
    void computeMinMax();
    void send();
};