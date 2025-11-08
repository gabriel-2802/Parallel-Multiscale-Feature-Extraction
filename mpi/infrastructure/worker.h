#pragma once

#include <mpi.h>
#include <cfloat>

#include "../helpers/kernels.h"
#include "aux.h"
#include "entity.h"
#include "../helpers/image.h"
#include "../helpers/kernels.h"

class Worker : public Entity {
public:
    Worker(int numtasks, int rank);
    ~Worker() override;
    void run() override;

private:
    std::vector<std::vector<double>> pixels;
    ProcessDims dims{0,0,0,0,0};
    MinMaxVals minMax{DBL_MAX, -DBL_MAX};

    void receive();
    void process(LAYER layer);
    void computeMinMax();
    void normalize();
    void send();
};