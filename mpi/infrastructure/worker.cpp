#include "worker.h"
#include <algorithm>

using namespace std;

Crew::Crew(int numtasks, int rank) : Entity(numtasks, rank) {}

Crew::~Crew() {
}

void Crew::run() {
    for (int layer = LAYER::ONE; layer <= LAYER::THREE; ++layer) {
        receive();
        process(static_cast<LAYER>(layer));
        computeMinMax();
        normalize();
        send();
    }
}

void Crew::receive() {
    ProcessDims dims(0,0,0,0,0);
    MPI_Recv(&dims, sizeof(ProcessDims), MPI_BYTE, MASTER_RANK, COMM_TAGS::DIMENSIONS, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    this->dims = dims;

    // receive directly into flat array
    pixels.resize(dims.totalRows * dims.width);
    MPI_Recv(pixels.data(), dims.totalRows * dims.width, MPI_DOUBLE, MASTER_RANK, COMM_TAGS::IMAGE_DATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

void Crew::send() {
    int startIdx = dims.offset * dims.width;
    MPI_Send(&pixels[startIdx], dims.rowsForWorker * dims.width, MPI_DOUBLE, MASTER_RANK, COMM_TAGS::RESULT_DATA, MPI_COMM_WORLD);
}
