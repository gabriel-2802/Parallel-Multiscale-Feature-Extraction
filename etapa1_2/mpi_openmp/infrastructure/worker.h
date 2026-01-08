#pragma once

#include <mpi.h>
#include <cfloat>
#include <vector>

#include "../helpers/kernels.h"
#include "auxs.h"
#include "entity.h"
#include "../helpers/image.h"
#include "../helpers/kernels.h"

class Crew : public Entity {
public:
    Crew(int numtasks, int rank);
    ~Crew() override;
    void run() override;

private:
    void receive();
    void send();
};