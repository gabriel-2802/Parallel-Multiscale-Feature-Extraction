#include "entity.h"
#include <iostream>

using namespace std;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << endl; \
            exit(1); \
        } \
    } while(0)

#define BLOCK_SIZE 16
#define MAX_KERNEL_SIZE 7

// CUDA kernel for convolution - uses global memory for kernel instead of constant memory
__global__ void convolutionKernel(const double* input, double* output, const int* kernel,
                                   int width, int totalRows, int rowsForWorker, int offset,
                                   int kernelSize, int padding, double divisor) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= rowsForWorker) return;
    
    double sum = 0.0;
    
    for (int ky = -padding; ky <= padding; ++ky) {
        for (int kx = -padding; kx <= padding; ++kx) {
            int iy = offset + y + ky;
            int ix = x + kx;
            
            // Clamp to boundaries
            iy = min(max(iy, 0), totalRows - 1);
            ix = min(max(ix, 0), width - 1);
            
            int kidx = (ky + padding) * kernelSize + (kx + padding);
            sum += input[iy * width + ix] * kernel[kidx];
        }
    }
    
    // Write to output using relative indexing (no offset needed)
    output[y * width + x] = sum / divisor;
}

// CUDA kernel to find min and max values (reduction)
__global__ void findMinMaxKernel(const double* data, double* minVals, double* maxVals, 
                                  int width, int rowsForWorker) {
    extern __shared__ double sdata[];
    double* smin = sdata;
    double* smax = sdata + blockDim.x;
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    int size = rowsForWorker * width;
    
    // Initialize with extreme values
    double localMin = DBL_MAX;
    double localMax = -DBL_MAX;
    
    // Load and do first comparison - data is now compact without padding
    if (i < size) {
        double val = data[i];
        localMin = val;
        localMax = val;
    }
    if (i + blockDim.x < size) {
        double val = data[i + blockDim.x];
        localMin = val < localMin ? val : localMin;
        localMax = val > localMax ? val : localMax;
    }
    
    smin[tid] = localMin;
    smax[tid] = localMax;
    __syncthreads();
    
    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smin[tid] = smin[tid] < smin[tid + s] ? smin[tid] : smin[tid + s];
            smax[tid] = smax[tid] > smax[tid + s] ? smax[tid] : smax[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block
    if (tid == 0) {
        minVals[blockIdx.x] = smin[0];
        maxVals[blockIdx.x] = smax[0];
    }
}

// CUDA kernel to normalize values
__global__ void normalizeKernel(double* data, int width, int rowsForWorker, 
                                 double minVal, double range) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= rowsForWorker) return;
    
    int idx = y * width + x;
    data[idx] = 255.0 * (data[idx] - minVal) / range;
}

Entity::Entity(int numtasks, int rank) 
    : numtasks(numtasks), rank(rank), d_input(nullptr), d_output(nullptr), cudaInitialized(false) {
}

Entity::~Entity() {
    cleanupCUDA();
}

void Entity::initCUDA() {
    // Get the number of available CUDA devices (only once)
    static bool deviceSet = false;
    if (!deviceSet) {
        int deviceCount = 0;
        cudaError_t err = cudaGetDeviceCount(&deviceCount);
        if (err != cudaSuccess || deviceCount == 0) {
            cerr << "Rank " << rank << ": No CUDA devices available or error: " 
                 << cudaGetErrorString(err) << endl;
            exit(1);
        }
        
        // Assign each MPI process to a GPU (round-robin if more processes than GPUs)
        int deviceId = rank % deviceCount;
        CUDA_CHECK(cudaSetDevice(deviceId));
        deviceSet = true;
    }
    
    int inputSize = dims.totalRows * dims.width;
    int outputSize = dims.rowsForWorker * dims.width;
    
    if (inputSize <= 0 || outputSize <= 0) {
        cerr << "Rank " << rank << ": Invalid dimensions - totalRows=" << dims.totalRows 
             << " width=" << dims.width << " rowsForWorker=" << dims.rowsForWorker << endl;
        exit(1);
    }
    
    // Free existing buffers if dimensions changed
    if (cudaInitialized) {
        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_output));
    }
    
    // Allocate with current dimensions
    CUDA_CHECK(cudaMalloc(&d_input, inputSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_output, outputSize * sizeof(double)));
    
    cudaInitialized = true;
}

void Entity::cleanupCUDA() {
    if (cudaInitialized) {
        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_output));
        cudaInitialized = false;
    }
}

void Entity::computeMinMax() {
    int size = dims.rowsForWorker * dims.width;
    int blockSize = 256;
    int numBlocks = (size + blockSize * 2 - 1) / (blockSize * 2);
    
    double *d_minVals, *d_maxVals;
    CUDA_CHECK(cudaMalloc(&d_minVals, numBlocks * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_maxVals, numBlocks * sizeof(double)));
    
    // Find min/max with reduction on output buffer (compact data)
    findMinMaxKernel<<<numBlocks, blockSize, 2 * blockSize * sizeof(double)>>>(
        d_output, d_minVals, d_maxVals, dims.width, dims.rowsForWorker);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy partial results to host for final reduction
    vector<double> h_minVals(numBlocks), h_maxVals(numBlocks);
    CUDA_CHECK(cudaMemcpy(h_minVals.data(), d_minVals, numBlocks * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_maxVals.data(), d_maxVals, numBlocks * sizeof(double), cudaMemcpyDeviceToHost));
    
    MinMaxVals localMinMax{DBL_MAX, -DBL_MAX};
    for (int i = 0; i < numBlocks; ++i) {
        localMinMax.min = min(localMinMax.min, h_minVals[i]);
        localMinMax.max = max(localMinMax.max, h_maxVals[i]);
    }
    
    CUDA_CHECK(cudaFree(d_minVals));
    CUDA_CHECK(cudaFree(d_maxVals));
    
    // Global reduction across all MPI processes
    MPI_Allreduce(&localMinMax.min, &minMax.min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&localMinMax.max, &minMax.max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
}

void Entity::process(LAYER layer) {
    const auto& kernel = (layer == LAYER::ONE) ? LAYER_1_KERNEL :
                         (layer == LAYER::TWO) ? LAYER_2_KERNEL :
                         LAYER_3_KERNEL;

    const double divisor = (layer == LAYER::ONE) ? LAYER_1_DIV :
                           (layer == LAYER::TWO) ? LAYER_2_DIV :
                           LAYER_3_DIV;
    
    int kernelSize = kernel.size();
    int padding = kernelSize / 2;
    
    // Flatten kernel and allocate on device (using global memory instead of constant)
    vector<int> flatKernel(kernelSize * kernelSize);
    int* flatPtr = flatKernel.data();
    for (int i = 0; i < kernelSize; ++i) {
        const int* rowPtr = kernel[i].data();
        for (int j = 0; j < kernelSize; ++j) {
            *flatPtr++ = *rowPtr++;
        }
    }
    
    int* d_kernel;
    CUDA_CHECK(cudaMalloc(&d_kernel, kernelSize * kernelSize * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_kernel, flatKernel.data(), 
                          kernelSize * kernelSize * sizeof(int), cudaMemcpyHostToDevice));
    
    // Upload pixels to GPU
    int size = dims.totalRows * dims.width;
    CUDA_CHECK(cudaMemcpy(d_input, pixels.data(), size * sizeof(double), cudaMemcpyHostToDevice));
    
    // Launch convolution kernel
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((dims.width + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                 (dims.rowsForWorker + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    convolutionKernel<<<gridDim, blockDim>>>(d_input, d_output, d_kernel,
                                              dims.width, dims.totalRows, 
                                              dims.rowsForWorker, dims.offset,
                                              kernelSize, padding, divisor);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Free the kernel memory
    CUDA_CHECK(cudaFree(d_kernel));
    
    // Copy compact output back to padded input for next layer
    // This allows next layer's convolution to access full padded buffer
    CUDA_CHECK(cudaMemcpy(d_input + dims.offset * dims.width, d_output, 
                         dims.rowsForWorker * dims.width * sizeof(double), 
                         cudaMemcpyDeviceToDevice));
}

void Entity::normalize() {
    double range = (minMax.max - minMax.min == 0) ? 1.0 : (minMax.max - minMax.min);
    
    // Launch normalization kernel
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((dims.width + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                 (dims.rowsForWorker + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    normalizeKernel<<<gridDim, blockDim>>>(d_output, dims.width, dims.rowsForWorker, 
                                            minMax.min, range);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Download normalized data back to host (only worker's rows)
    CUDA_CHECK(cudaMemcpy(pixels.data() + dims.offset * dims.width, d_output, 
                         dims.rowsForWorker * dims.width * sizeof(double), 
                         cudaMemcpyDeviceToHost));
}
