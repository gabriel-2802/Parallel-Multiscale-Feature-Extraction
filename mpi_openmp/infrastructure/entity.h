#pragma once


#include <mpi.h>
#include <cfloat>
#include <vector>
#include <algorithm>
#include <omp.h>

#include "../helpers/kernels.h"
#include "aux.h"
#include "entity.h"
#include "../helpers/image.h"
#include "../helpers/kernels.h"

// abstract class
class Entity {
public:
    Entity(int numtasks, int rank) : numtasks(numtasks), rank(rank) {};
    virtual ~Entity() {};
    virtual void run() {};

protected:
    const int numtasks;
    const int rank;

    std::vector<double> pixels; // flat array
    ProcessDims dims{0,0,0,0,0};
    MinMaxVals minMax{DBL_MAX, -DBL_MAX};

    void process(LAYER layer);
    void computeMinMax();
    void normalize();

    // helper to access pixel at (row, col) in flat array
    inline double& at(int row, int col) { return pixels[row * dims.width + col]; }
    inline const double& at(int row, int col) const { return pixels[row * dims.width + col]; }
};