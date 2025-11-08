#include "worker.h"


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

    vector<double> flatData(dims.totalRows * dims.width);
    MPI_Recv(flatData.data(), dims.totalRows * dims.width, MPI_DOUBLE, MASTER_RANK, COMM_TAGS::IMAGE_DATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    pixels.assign(dims.totalRows, vector<double>(dims.width));
    for (int i = 0; i < dims.totalRows; ++i) {
        copy(flatData.begin() + i * dims.width, flatData.begin() + (i + 1) * dims.width, pixels[i].begin());
    }
}

void Worker::computeMinMax() {
    minMax.min = DBL_MAX;
    minMax.max = -DBL_MAX;
    
    for (int i = 0; i < dims.rowsForWorker; ++i) {
        for (int j = 0; j < dims.width; ++j) {
            double val = pixels[dims.offset + i][j];
            if (val < minMax.min) {
                minMax.min = val;
            }
            if (val > minMax.max) {
                minMax.max = val;
            }
        }
    }
    
    // send local min/max to master and receive global min/max
    MPI_Send(&minMax, sizeof(MinMaxVals), MPI_BYTE, MASTER_RANK, COMM_TAGS::MIN_MAX_DATA, MPI_COMM_WORLD);
    MPI_Recv(&minMax, sizeof(MinMaxVals), MPI_BYTE, MASTER_RANK, COMM_TAGS::MIN_MAX_DATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
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
    
    // create a result matrix for the processed rows (without padding)
    vector<vector<double>> result(dims.rowsForWorker, vector<double>(dims.width));
    
    // apply convolution only to the working rows
    for (int i = 0; i < dims.rowsForWorker; ++i) {
        for (int j = 0; j < dims.width; ++j) {
            double sum = 0.0;
            
            for (int ki = 0; ki < kernelSize; ++ki) {
                for (int kj = 0; kj < kernelSize; ++kj) {
                    int pixelRow = dims.offset + i + (ki - kernelRadius);
                    int pixelCol = j + (kj - kernelRadius);
                    
                    // CLAMP instead of boundary check
                    pixelRow = std::max(0, std::min(pixelRow, dims.totalRows - 1));
                    pixelCol = std::max(0, std::min(pixelCol, dims.width - 1));
                    
                    sum += pixels[pixelRow][pixelCol] * kernel[ki][kj];
                }
            }
            
            result[i][j] = sum / divisor;
        }
    }
    
    for (int i = 0; i < dims.rowsForWorker; ++i)
        pixels[dims.offset + i] = std::move(result[i]);
}

void Worker::normalize() {
    double range = (minMax.max - minMax.min == 0) ? 1.0 : (minMax.max - minMax.min);

    for (int i = 0; i < dims.rowsForWorker; ++i)
        for (int j = 0; j < dims.width; ++j)
            pixels[dims.offset + i][j] = 255.0 * (pixels[dims.offset + i][j] - minMax.min) / range;
}

void Worker::send() {
    vector<double> flatData(dims.rowsForWorker * dims.width);
    for (int i = 0; i < dims.rowsForWorker; ++i) {
        copy(pixels[i + dims.offset].begin(), pixels[i + dims.offset].end(), flatData.begin() + i * dims.width);
    }
    MPI_Send(flatData.data(), dims.rowsForWorker * dims.width, MPI_DOUBLE, MASTER_RANK, COMM_TAGS::RESULT_DATA, MPI_COMM_WORLD);
}
