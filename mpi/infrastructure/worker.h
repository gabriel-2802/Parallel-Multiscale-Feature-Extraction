#pragma once

#include <mpi.h>
#include <cfloat>
#include <vector>

#include "../helpers/kernels.h"
#include "aux.h"
#include "entity.h"
#include "../helpers/image.h"
#include "../helpers/kernels.h"

class Crew : public Entity {
public:
    Crew(int numtasks, int rank);
    ~Crew() override;
    void run() override;

private:
    std::vector<double> pixels; // flat array
    ProcessDims dims{0,0,0,0,0};
    MinMaxVals minMax{DBL_MAX, -DBL_MAX};

    void receive();
    void send();

};