#include "master.h"
#include <mpi.h>
#include <cfloat>
#include "../helpers/kernels.h"
#include <algorithm>

using namespace std;

Master::Master(int numtasks, int rank, std::string inputImagePath, std::string outputImagePath) : Entity(numtasks, rank) {
    image = std::make_unique<GreyScaleImage>(inputImagePath);
    outImagePath = outputImagePath;
}

Master::~Master() {
}

void Master::run() {
    for (int layer = LAYER::ONE; layer <= LAYER::THREE; ++layer) {
        scatter(static_cast<LAYER>(layer));
        findGlobalMinMax();
        gatherAndSaveLayer();
        break;
    }

    saveImage();
}


void Master::scatter(LAYER layer) {

    // image data
    const auto& matrix = image->getMatrix();
    int height = image->getHeight();
    int width = image->getWidth();
    int padding = getPaddingForLayer(layer);
    
    // number of workers (excluding master)
    int numWorkers = numtasks - 1;

    // base rows per worker and remainder
    int baseRows = height / numWorkers;
    int remainder = height % numWorkers;

    int startRow = 0;    
    for (int worker = 1; worker < numtasks; ++worker) {
        // rows for this worker (distribute remainder)
        int rowsForWorker = baseRows + (worker <= remainder ? 1 : 0);
        
        // calculate actual start and end with padding
        int actualStart = max(0, startRow - padding);
        int actualEnd = min(height, startRow + rowsForWorker + padding);
        int totalRows = actualEnd - actualStart;
        
        // send dimensions
        ProcessDims dims(totalRows, width, rowsForWorker, padding, startRow - actualStart);
        MPI_Send(&dims, sizeof(ProcessDims), MPI_BYTE, worker, COMM_TAGS::DIMENSIONS, MPI_COMM_WORLD);

        vector<double> buffer(totalRows * width);
        for (int i = 0; i < totalRows; ++i) {
            copy(matrix[actualStart + i].begin(), matrix[actualStart + i].end(), buffer.begin() + i * width);
        }
        MPI_Send(buffer.data(), totalRows * width, MPI_DOUBLE, worker, COMM_TAGS::IMAGE_DATA, MPI_COMM_WORLD);
        
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

void Master::findGlobalMinMax() {
    MinMaxVals minMax{DBL_MAX, -DBL_MAX};

    for (int worker = 1; worker < numtasks; ++worker) {
        MinMaxVals localMinMax{0.0, 0.0};
        MPI_Recv(&localMinMax, sizeof(MinMaxVals), MPI_BYTE, worker, COMM_TAGS::MIN_MAX_DATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        minMax.localMin = min(minMax.localMin, localMinMax.localMin);
        minMax.localMax = max(minMax.localMax, localMinMax.localMax);
    }

    for (int worker = 1; worker < numtasks; ++worker) {
        MPI_Send(&minMax, sizeof(MinMaxVals), MPI_BYTE, worker, COMM_TAGS::MIN_MAX_DATA, MPI_COMM_WORLD);
    }
}

void Master::gatherAndSaveLayer() {
    int height = image->getHeight();
    int width = image->getWidth();

    vector<vector<double>> pixels(height, vector<double>(width));

    // number of workers (excluding master)
    int numWorkers = numtasks - 1;

    // base rows per worker and remainder
    int baseRows = height / numWorkers;
    int remainder = height % numWorkers;

    int startRow = 0;    
    for (int worker = 1; worker < numtasks; ++worker) {
        // rows for this worker (distribute remainder)
        int rowsForWorker = baseRows + (worker <= remainder ? 1 : 0);

        vector<double> buffer(rowsForWorker * width);
        MPI_Recv(buffer.data(), rowsForWorker * width, MPI_DOUBLE, worker, COMM_TAGS::RESULT_DATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for (int i = 0; i < rowsForWorker; ++i) {
            copy(buffer.begin() + i * width, buffer.begin() + (i + 1) * width, pixels[startRow + i].begin());
        }

        startRow += rowsForWorker;
    }

    image->setMatrix(pixels);
}

void Master::saveImage() {
    image->save(outImagePath);
}
