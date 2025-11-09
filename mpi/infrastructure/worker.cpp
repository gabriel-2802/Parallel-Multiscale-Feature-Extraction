#include "worker.h"
#include <algorithm>

using namespace std;

Worker::Worker(int numtasks, int rank) : Entity(numtasks, rank) {}

Worker::~Worker() {
}

void Worker::run() {
    for (int layer = LAYER::ONE; layer <= LAYER::THREE; ++layer) {
        receive();
        process(static_cast<LAYER>(layer));
        computeMinMax();
        normalize();
        send();
    }
}

void Worker::receive() {
    ProcessDims dims(0,0,0,0,0);
    MPI_Recv(&dims, sizeof(ProcessDims), MPI_BYTE, MASTER_RANK, COMM_TAGS::DIMENSIONS, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    this->dims = dims;

    // receive directly into flat array
    pixels.resize(dims.totalRows * dims.width);
    MPI_Recv(pixels.data(), dims.totalRows * dims.width, MPI_DOUBLE, MASTER_RANK, COMM_TAGS::IMAGE_DATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

void Worker::computeMinMax() {
    MinMaxVals localMinMax{DBL_MAX, -DBL_MAX};
    
    // compute local min/max for worker's rows using flat array
    for (int i = 0; i < dims.rowsForWorker; ++i) {
        for (int j = 0; j < dims.width; ++j) {
            double val = at(dims.offset + i, j);
            if (val < localMinMax.min) {
                localMinMax.min = val;
            }
            if (val > localMinMax.max) {
                localMinMax.max = val;
            }
        }
    }
    
    MPI_Allreduce(&localMinMax.min, &minMax.min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&localMinMax.max, &minMax.max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
}

void Worker::process(LAYER layer) {
    const auto& kernel = (layer == LAYER::ONE) ? LAYER_1_KERNEL :
                         (layer == LAYER::TWO) ? LAYER_2_KERNEL :
                         LAYER_3_KERNEL;

    const double divisor = (layer == LAYER::ONE) ? LAYER_1_DIV :
                           (layer == LAYER::TWO) ? LAYER_2_DIV :
                           LAYER_3_DIV;
    
    int kernelSize = kernel.size();
    int kernelRadius = kernelSize / 2;
    
    // create a flat result array for the processed rows (without padding)
    vector<double> result(dims.rowsForWorker * dims.width);
    
    // apply convolution only to the working rows
    for (int i = 0; i < dims.rowsForWorker; ++i) {
        for (int j = 0; j < dims.width; ++j) {
            double sum = 0.0;
            
            for (int ki = 0; ki < kernelSize; ++ki) {
                for (int kj = 0; kj < kernelSize; ++kj) {
                    int pixelRow = dims.offset + i + (ki - kernelRadius);
                    int pixelCol = j + (kj - kernelRadius);
                    
                    pixelRow = max(0, min(pixelRow, dims.totalRows - 1));
                    pixelCol = max(0, min(pixelCol, dims.width - 1));
                    
                    sum += at(pixelRow, pixelCol) * kernel[ki][kj];
                }
            }
            
            result[i * dims.width + j] = sum / divisor;
        }
    }
    
    // copy result back to working rows in pixels
    for (int i = 0; i < dims.rowsForWorker; ++i) {
        for (int j = 0; j < dims.width; ++j) {
            at(dims.offset + i, j) = result[i * dims.width + j];
        }
    }
}

void Worker::normalize() {
    double range = (minMax.max - minMax.min == 0) ? 1.0 : (minMax.max - minMax.min);

    for (int i = 0; i < dims.rowsForWorker; ++i) {
        for (int j = 0; j < dims.width; ++j) {
            at(dims.offset + i, j) = 255.0 * (at(dims.offset + i, j) - minMax.min) / range;
        }
    }
}

void Worker::send() {
    int startIdx = dims.offset * dims.width;
    MPI_Send(&pixels[startIdx], dims.rowsForWorker * dims.width, MPI_DOUBLE, MASTER_RANK, COMM_TAGS::RESULT_DATA, MPI_COMM_WORLD);
}
