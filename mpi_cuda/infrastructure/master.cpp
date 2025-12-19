#include "master.h"
#include <mpi.h>
#include <cfloat>
#include "../helpers/kernels.h"
#include <algorithm>

using namespace std;

Master::Master(int numtasks, int rank, string inputImagePath, string outputImagePath) : Entity(numtasks, rank) {
    image = make_unique<GreyScaleImage>(inputImagePath);
    outImagePath = outputImagePath;
}

Master::~Master() {
}

void Master::run() {
    for (int layer = LAYER::ONE; layer <= LAYER::THREE; ++layer) {
        scatter(static_cast<LAYER>(layer));
        
        // Initialize CUDA on first layer
        if (layer == LAYER::ONE) {
            initCUDA();
        }
        
        process(static_cast<LAYER>(layer));
        computeMinMax();
        normalize();
        gatherAndSaveLayer();
        
        // Cleanup CUDA after last layer
        if (layer == LAYER::THREE) {
            cleanupCUDA();
        }
    }

    saveImage();
}

void Master::scatter(LAYER layer) {

    // image data
    const auto& flattenMatrix = image->getFlattenedMatrix();
    int height = image->getHeight();
    int width = image->getWidth();
    int padding = getPaddingForLayer(layer);
    
    // number of workers (excluding master)
    int numWorkers = numtasks;

    // base rows per worker and remainder
    int baseRows = height / numWorkers;
    int remainder = height % numWorkers;

    vector<MPI_Request> requests((numtasks - 1) * 2);
    int reqIdx = 0;

    int startRow = 0;    
    for (int worker = 0; worker < numtasks; ++worker) {
        // rows for this worker (distribute remainder)
        int rowsForWorker = baseRows + (worker < remainder ? 1 : 0);
        
        // calculate actual start and end with padding
        int actualStart = max(0, startRow - padding);
        int actualEnd = min(height, startRow + rowsForWorker + padding);
        int totalRows = actualEnd - actualStart;
        
        // prep dimensions
        ProcessDims dims(totalRows, width, rowsForWorker, padding, startRow - actualStart);

        // prep work for self
        if (worker == MASTER_RANK) {
            this->dims = dims;
            pixels.resize(dims.totalRows * dims.width);
            copy(flattenMatrix.begin() + actualStart * width, flattenMatrix.begin() + actualStart * width + totalRows * width, pixels.begin());
            startRow += rowsForWorker;
            continue;
        }
        
        // non-blocking send to worker for overlapping communication
        MPI_Isend(&dims, sizeof(ProcessDims), MPI_BYTE, worker, COMM_TAGS::DIMENSIONS, MPI_COMM_WORLD, &requests[reqIdx++]);
        MPI_Isend(flattenMatrix.data() + actualStart * width, totalRows * width, MPI_DOUBLE, worker, COMM_TAGS::IMAGE_DATA, MPI_COMM_WORLD, &requests[reqIdx++]);
        
        startRow += rowsForWorker;
    }
    
    // wait for all sends to complete
    MPI_Waitall(reqIdx, requests.data(), MPI_STATUSES_IGNORE);
}

int Master::getPaddingForLayer(LAYER layer) {
    switch (layer) {
        case LAYER::ONE:
            return LAYER_1_PADDING;
        case LAYER::TWO:
            return LAYER_2_PADDING;
        case LAYER::THREE:
            return LAYER_3_PADDING;
        default:
            return 0;
    }
}

void Master::gatherAndSaveLayer() {
    int height = image->getHeight();
    int width = image->getWidth();

    vector<double> pixels(height * width);

    // number of workers 
    int numWorkers = numtasks;

    // base rows per worker and remainder
    int baseRows = height / numWorkers;
    int remainder = height % numWorkers;

    // gather from self
    int rowsForMaster = baseRows + (0 < remainder ? 1 : 0);
    copy(this->pixels.begin() + dims.offset * dims.width, this->pixels.begin() + (dims.offset + dims.rowsForWorker) * dims.width, pixels.begin());

    // post all receives concurrently
    vector<MPI_Request> requests(numtasks - 1);
    int startRow = rowsForMaster;    
    for (int worker = 1; worker < numtasks; ++worker) {
        // rows for this worker (distribute remainder)
        int rowsForWorker = baseRows + (worker < remainder ? 1 : 0);
        
        MPI_Irecv(&pixels[startRow * width], rowsForWorker * width, MPI_DOUBLE, worker, COMM_TAGS::RESULT_DATA, MPI_COMM_WORLD, &requests[worker - 1]);
        
        startRow += rowsForWorker;
    }
    
    // wait for all receives to complete
    MPI_Waitall(numtasks - 1, requests.data(), MPI_STATUSES_IGNORE);

    image->setFlattenedMatrix(pixels);
}

void Master::saveImage() {
    image->save(outImagePath);
}
