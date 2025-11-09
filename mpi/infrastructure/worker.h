#pragma once

#include <mpi.h>

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
    void receive();
    void process(LAYER layer);
    void computeMinMax();
    void normalize();
    void send();
    
    // Helper to access pixel at (row, col) in flat array
    inline double& at(int row, int col) { return pixels[row * dims.width + col]; }
    inline const double& at(int row, int col) const { return pixels[row * dims.width + col]; }
};