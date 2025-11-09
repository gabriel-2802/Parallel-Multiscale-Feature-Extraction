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
        process(static_cast<LAYER>(layer));
        computeMinMax();
        normalize();
        gatherAndSaveLayer();
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

    int startRow = 0;    
    for (int worker = 0; worker < numtasks; ++worker) {
        // rows for this worker (distribute remainder)
        int rowsForWorker = baseRows + (worker - 1 < remainder ? 1 : 0);
        
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
        
        // send to worker
        MPI_Send(&dims, sizeof(ProcessDims), MPI_BYTE, worker, COMM_TAGS::DIMENSIONS, MPI_COMM_WORLD);
        MPI_Send(flattenMatrix.data() + actualStart * width, totalRows * width, MPI_DOUBLE, worker, COMM_TAGS::IMAGE_DATA, MPI_COMM_WORLD);
        
        startRow += rowsForWorker;
    }
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
    copy(this->pixels.begin() + dims.offset * dims.width, this->pixels.begin() + (dims.offset + dims.rowsForWorker) * dims.width, pixels.begin());

    int startRow = 0;    
    for (int worker = 1; worker < numtasks; ++worker) {
        // rows for this worker (distribute remainder)
        int rowsForWorker = baseRows + (worker - 1 < remainder ? 1 : 0);
        
        MPI_Recv(&pixels[startRow * width], rowsForWorker * width, MPI_DOUBLE, worker, COMM_TAGS::RESULT_DATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        startRow += rowsForWorker;
    }

    image->setFlattenedMatrix(pixels);
}

void Master::saveImage() {
    image->save(outImagePath);
}
