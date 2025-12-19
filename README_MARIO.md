# Parallel Multiscale Feature Extraction - OpenMP & CUDA

This project implements a parallel image processing pipeline using OpenMP for CPU parallelism and CUDA for GPU acceleration. The pipeline performs multiscale convolution filtering across three layers with different kernel sizes, computing normalized feature maps for each layer.

## Common Pipeline Overview

All implementations follow the same processing pipeline for each of the three layers:

1. **Convolution**: Apply layer-specific kernels to the image
   - Layer 1: 3×3 kernel
   - Layer 2: 5×5 kernel  
   - Layer 3: 7×7 kernel
2. **Min/Max Computation**: Find minimum and maximum pixel values
3. **Normalization**: Scale pixel values to [0, 255] range
4. **Output**: Each layer processes the output of the previous layer

Normalization formula: `normalized = 255 × (value - min) / (max - min)`

---

## 2. OpenMP Implementation

Implements parallel image processing using OpenMP for shared memory parallelism on multi-core CPUs.

### Architecture
- **Shared Memory Model**: Utilizes multiple threads sharing the same memory space.
- **Loop Parallelism**: Distributes loop iterations across available threads.
- **Dynamic Scheduling**: Threads process independent chunks of the image.

### Pipeline Stages

**Convolution**
- Parallelizes the outer loop (rows) using `#pragma omp parallel for`.
- Each thread processes a distinct set of rows independently.
- Inner loops iterate over pixels and kernel windows.
- Race conditions are avoided as each pixel output is independent.

**Min/Max Reduction**
- Uses OpenMP's built-in reduction mechanism: `#pragma omp parallel for reduction(min:minVal) reduction(max:maxVal)`.
- Each thread computes local min/max values.
- Local results are combined into global min/max values at the end of the parallel region.

**Normalization**
- Parallelizes the scaling process using `#pragma omp parallel for`.
- Each pixel calculation is independent and can be processed in parallel.

### File Structure
```
openmp/
├── openmp.cpp                   # Main entry point
└── Makefile                     # Build configuration
```

---

## 6. CUDA Implementation

Implements massively parallel image processing using NVIDIA CUDA for execution GPUs.

### Architecture
- **Host-Device Model**: CPU (Host) manages memory and orchestrates execution; GPU (Device) performs heavy computation.
- **Massive Parallelism**: Launches thousands of lightweight threads organized into blocks and grids.
- **Memory Hierarchy**: Utilizes Global, Constant, and Shared memory for optimization.

### Pipeline Stages

**Data Transfer**
- Image is flattened into a 1D array.
- Data is copied from Host to Device memory (`cudaMemcpy`).

**Convolution**
- **Kernel**: `convolutionKernel` launched with a 2D grid/block configuration.
- **Constant Memory**: Convolution kernels are stored in Constant Memory (`__constant__`) for high-speed cached access by all threads.
- **Mapping**: Each thread computes one output pixel.
- **Boundary Handling**: Threads check boundaries to handle padding logic.

**Min/Max Reduction**
- **Kernel**: `findMinMaxKernel` implements a parallel reduction.
- **Shared Memory**: Uses shared memory (`__shared__`) for fast intra-block reduction.
- **Two-Stage Reduction**: 
  1. GPU reduces data to partial results (one per block).
  2. CPU finishes the reduction on the partial results (or a second kernel launch).

**Normalization**
- **Kernel**: `normalizeKernel` scales pixels in parallel.
- Uses the global min/max values computed in the previous step.

### File Structure
```
cuda/
├── cuda.cu                      # Main CUDA implementation
└── Makefile                     # Build configuration
```
