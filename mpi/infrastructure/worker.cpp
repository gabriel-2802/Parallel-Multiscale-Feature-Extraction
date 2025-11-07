#include "worker.h"


using namespace std;

Worker::Worker(int numtasks, int rank) : Entity(numtasks, rank) {}

Worker::~Worker() {
}

void Worker::run() {
    for (int layer = LAYER::ONE; layer <= LAYER::THREE; ++layer) {
        receive();
        process();
        computeMinMax();
        normalize();
        send();
    }
}

void Worker::receive() {
    ProcessDims dims(0,0,0,0,0);
    MPI_Recv(&dims, sizeof(ProcessDims), MPI_BYTE, MASTER_RANK, COMM_TAGS::DIMENSIONS, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    this->dims = dims;

    vector<double> flatData(dims.totalRows * dims.width);
    MPI_Recv(flatData.data(), dims.totalRows * dims.width, MPI_DOUBLE, MASTER_RANK, COMM_TAGS::IMAGE_DATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    pixels.assign(dims.totalRows, vector<double>(dims.width));
    for (int i = 0; i < dims.totalRows; ++i) {
        copy(flatData.begin() + i * dims.width, flatData.begin() + (i + 1) * dims.width, pixels[i].begin());
    }
}

void Worker::process() {
}

void Worker::computeMinMax() {
    for (const auto& row : pixels) {
        for (double val : row) {
            if (val < minMax.localMin) {
                minMax.localMin = val;
            }
            if (val > minMax.localMax) {
                minMax.localMax = val;
            }
        }
    }
    // send local min/max to master and receive global min/max
    MPI_Send(&minMax, sizeof(MinMaxVals), MPI_BYTE, MASTER_RANK, COMM_TAGS::MIN_MAX_DATA, MPI_COMM_WORLD);
    MPI_Recv(&minMax, sizeof(MinMaxVals), MPI_BYTE, MASTER_RANK, COMM_TAGS::MIN_MAX_DATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

void Worker::normalize() {
    double range = minMax.localMax - minMax.localMin;
    for (auto& row : pixels) {
        for (auto& val : row) {
            val = 255.0 * (val - minMax.localMin) / range;
        }
    }
}

void Worker::send() {
    vector<double> flatData(dims.rowsForWorker * dims.width);
    for (int i = 0; i < dims.rowsForWorker; ++i) {
        copy(pixels[i + dims.offset].begin(), pixels[i + dims.offset].end(), flatData.begin() + i * dims.width);
    }
    MPI_Send(flatData.data(), dims.rowsForWorker * dims.width, MPI_DOUBLE, MASTER_RANK, COMM_TAGS::RESULT_DATA, MPI_COMM_WORLD);
}
