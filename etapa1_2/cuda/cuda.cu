#include <iostream>
#include <chrono>
#include <vector>
#include <climits>
#include <cfloat>
#include <cuda_runtime.h>
#include "../helpers/image.h"
#include "../helpers/kernels.h"

using namespace std;
using namespace std::chrono;

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

// stored the convolution kernel in constant memory for faster access
__constant__ int d_kernel[MAX_KERNEL_SIZE * MAX_KERNEL_SIZE];

// convolution kernel implementation
__global__ void convolutionKernel(const double* input, double* output, 
                                   int width, int height, 
                                   int kernelSize, int padding, double divisor) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    double sum = 0.0;
    
    for (int ky = -padding; ky <= padding; ++ky) {
        for (int kx = -padding; kx <= padding; ++kx) {
            int iy = min(max(y + ky, 0), height - 1);
            int ix = min(max(x + kx, 0), width - 1);
            int kidx = (ky + padding) * kernelSize + (kx + padding);
            sum += input[iy * width + ix] * d_kernel[kidx];
        }
    }
    
    output[y * width + x] = sum / divisor;
}

// reduction kernel to find the min and max values
__global__ void findMinMaxKernel(const double* data, double* minVals, double* maxVals, int size) {
    extern __shared__ double sdata[];
    double* smin = sdata;
    double* smax = sdata + blockDim.x;
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    
    // initializing with extreme values before comparison
    double localMin = DBL_MAX;
    double localMax = -DBL_MAX;
    
    // loading data and performing the first comparison step
    if (i < size) {
        localMin = data[i];
        localMax = data[i];
    }
    if (i + blockDim.x < size) {
        double val = data[i + blockDim.x];
        localMin = val < localMin ? val : localMin;
        localMax = val > localMax ? val : localMax;
    }
    
    smin[tid] = localMin;
    smax[tid] = localMax;
    __syncthreads();
    
    // performing the reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smin[tid] = smin[tid] < smin[tid + s] ? smin[tid] : smin[tid + s];
            smax[tid] = smax[tid] > smax[tid + s] ? smax[tid] : smax[tid + s];
        }
        __syncthreads();
    }
    
    // writing the block's result back to global memory
    if (tid == 0) {
        minVals[blockIdx.x] = smin[0];
        maxVals[blockIdx.x] = smax[0];
    }
}

// kernel for normalizing the pixel values
__global__ void normalizeKernel(double* data, int size, double minVal, double range) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = 255.0 * (data[idx] - minVal) / range;
    }
}

void normalizeMatrixGPU(double* d_data, int size) {
    int blockSize = 256;
    int numBlocks = (size + blockSize * 2 - 1) / (blockSize * 2);
    
    double *d_minVals, *d_maxVals;
    CUDA_CHECK(cudaMalloc(&d_minVals, numBlocks * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_maxVals, numBlocks * sizeof(double)));
    
    // min and max values are found using the reduction kernel
    findMinMaxKernel<<<numBlocks, blockSize, 2 * blockSize * sizeof(double)>>>(
        d_data, d_minVals, d_maxVals, size);
    CUDA_CHECK(cudaGetLastError());
    
    // then, partial results are copied back to the host for final reduction
    vector<double> h_minVals(numBlocks), h_maxVals(numBlocks);
    CUDA_CHECK(cudaMemcpy(h_minVals.data(), d_minVals, numBlocks * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_maxVals.data(), d_maxVals, numBlocks * sizeof(double), cudaMemcpyDeviceToHost));
    
    double minVal = h_minVals[0], maxVal = h_maxVals[0];
    for (int i = 1; i < numBlocks; ++i) {
        minVal = minVal < h_minVals[i] ? minVal : h_minVals[i];
        maxVal = maxVal > h_maxVals[i] ? maxVal : h_maxVals[i];
    }
    
    double range = (maxVal - minVal == 0.0) ? 1.0 : (maxVal - minVal);
    
    // applying the normalization kernel
    int normalizeBlocks = (size + blockSize - 1) / blockSize;
    normalizeKernel<<<normalizeBlocks, blockSize>>>(d_data, size, minVal, range);
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaFree(d_minVals));
    CUDA_CHECK(cudaFree(d_maxVals));
}

// handles the kernel application on the GPU
void applyKernelGPU(double* d_input, double* d_output, int width, int height,
                    const vector<vector<int>>& kernel, double divisor, int padding) {
    int kernelSize = kernel.size();
    
    // flatten the 2D kernel and copy it to the constant memory symbol
    vector<int> flatKernel(kernelSize * kernelSize);
    for (int i = 0; i < kernelSize; ++i) {
        for (int j = 0; j < kernelSize; ++j) {
            flatKernel[i * kernelSize + j] = kernel[i][j];
        }
    }
    CUDA_CHECK(cudaMemcpyToSymbol(d_kernel, flatKernel.data(), 
                                   kernelSize * kernelSize * sizeof(int)));
    
    // launching the convolution kernel
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((width + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                 (height + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    convolutionKernel<<<gridDim, blockDim>>>(d_input, d_output, width, height, 
                                              kernelSize, padding, divisor);
    CUDA_CHECK(cudaGetLastError());
    
    // normalizing the result matrix
    normalizeMatrixGPU(d_output, width * height);
    
    CUDA_CHECK(cudaDeviceSynchronize());
}

int main() {
    auto start = high_resolution_clock::now();

    GreyScaleImage img("../images/upscaled_upscaled_image.png");
    
    int width = img.getWidth();
    int height = img.getHeight();
    int size = width * height;
    
    // converting the image to a flattened matrix for CUDA
    vector<double> h_data = img.getFlattenedMatrix();
    
    double *d_buffer1, *d_buffer2;
    CUDA_CHECK(cudaMalloc(&d_buffer1, size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_buffer2, size * sizeof(double)));
    
    // copying the image data to the GPU
    CUDA_CHECK(cudaMemcpy(d_buffer1, h_data.data(), size * sizeof(double), cudaMemcpyHostToDevice));

    auto kernelStart = high_resolution_clock::now();

    // applying the first layer kernel
    applyKernelGPU(d_buffer1, d_buffer2, width, height, LAYER_1_KERNEL, LAYER_1_DIV, LAYER_1_PADDING);
    cout << "Layer 1 complete" << endl;

    // applying  the second layer, using the output of the first as input
    applyKernelGPU(d_buffer2, d_buffer1, width, height, LAYER_2_KERNEL, LAYER_2_DIV, LAYER_2_PADDING);
    cout << "Layer 2 complete" << endl;

    // applying the last (third) layer
    applyKernelGPU(d_buffer1, d_buffer2, width, height, LAYER_3_KERNEL, LAYER_3_DIV, LAYER_3_PADDING);
    cout << "Layer 3 complete" << endl;

    auto kernelStop = high_resolution_clock::now();
    auto kernelDuration = duration_cast<milliseconds>(kernelStop - kernelStart);
    cout << "Kernel processing time: " << kernelDuration.count() << " ms" << endl;

    // copying the processed data back to the host
    CUDA_CHECK(cudaMemcpy(h_data.data(), d_buffer2, size * sizeof(double), cudaMemcpyDeviceToHost));
    
    // updating the image object and saving the result
    img.setFlattenedMatrix(h_data);
    img.save("../images/output_cuda.png");

    CUDA_CHECK(cudaFree(d_buffer1));
    CUDA_CHECK(cudaFree(d_buffer2));

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    cout << "Total processing time: " << duration.count() << " ms" << endl;
    
    return 0;
}
