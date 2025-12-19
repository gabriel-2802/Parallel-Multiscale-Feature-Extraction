# Parallel Multiscale Feature Extraction

This project implements a parallel image processing pipeline using multiple parallelization approaches. The pipeline performs multiscale convolution filtering across three layers with different kernel sizes, computing normalized feature maps for each layer.

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

## 1. Pthreads Implementation

Implements parallel image processing using POSIX threads (pthreads) for shared memory parallelism.

### Architecture
- Image divided into horizontal strips by rows
- Each pthread processes a contiguous block of rows
- Thread pool created for each pipeline stage

### Pipeline Stages

**Convolution**
- Image split by rows across pthreads
- Each thread processes its assigned rows
- Threads compute:
  - Convolution output
  - Local minimum
  - Local maximum

**Min/Max Reduction**
- After all threads finish, local min/max values are combined serially
- Low overhead due to small number of threads
- Produces global minimum and maximum

**Normalization**
- Second set of pthreads normalizes the output image
- Each thread processes a different range of rows
- Uses global min/max values from reduction phase

### Key Features
- Row-based work distribution
- Barrier synchronization between pipeline stages
- Thread reuse across layers
- Minimal thread creation overhead

### File Structure
```
pthreads/
├── pthreads.cpp                 # Main entry point
├── Makefile                     # Build configuration
└── infrastructure/
    ├── thread_manager.h/cpp     # Thread pool management
    ├── worker.h/cpp             # Worker thread implementation
    └── utils.h                  # Helper utilities
```

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

## 3. Pthreads + OpenMP Hybrid

Implements a hybrid parallel approach combining pthreads for coarse-grained parallelism with OpenMP for fine-grained parallelism within each thread.

### Architecture
- **Outer parallelism (pthreads)**: Image divided into row blocks
- **Inner parallelism (OpenMP)**: Pixel-level operations within each block
- Two-level parallelization strategy

### Pipeline Stages

**Convolution**
- Image split by rows across pthreads
- Each pthread processes a contiguous block of rows
- Inside each pthread, OpenMP parallelizes the inner pixel loops
- Each thread computes:
  - Convolution output
  - Local minimum
  - Local maximum

**Min/Max Reduction**
- Local min/max values combined using OpenMP reduction
- Low overhead due to efficient parallel reduction
- Produces global minimum and maximum

**Normalization**
- Pthreads perform image normalization on row blocks
- OpenMP parallelizes pixel processing inside each thread
- Fine-grained parallelism improves cache utilization

### Key Features
- Two-level parallelism (pthreads + OpenMP)
- OpenMP reductions for efficient aggregation
- Better cache locality through nested parallelism
- Flexible thread/task configuration

### File Structure
```
pthreads_openmp/
├── pthreads_omp.cpp             # Main entry point
├── Makefile                     # Build configuration
└── infrastructure/
    ├── thread_manager.h/cpp     # Thread pool management with OpenMP
    ├── worker.h/cpp             # Worker with OpenMP directives
    └── utils.h                  # Helper utilities
```

---

## 4. MPI Implementation

Implements parallel image processing using MPI (Message Passing Interface) for distributed memory parallelism across multiple processes.

### Architecture
- **Master-worker pattern**:
  - **Master (Rank 0)**: Coordinates distribution, loads/saves images, participates in processing
  - **Workers (Rank 1-N)**: Receive image chunks, process, and return results

### Pipeline Stages

**Scatter Phase**
- Master divides image by rows among all processes (including itself)
- Each process receives:
  - A contiguous block of rows
  - Padding rows for boundary convolution
  - Process dimensions metadata
- Uses non-blocking `MPI_Isend` to overlap communication

**Convolution**
- Each process applies layer-specific kernel to assigned rows
- Padding ensures boundary pixels are processed correctly
- Each layer processes output of previous layer

**Min/Max Reduction**
- Each process computes local minimum and maximum
- `MPI_Allreduce` combines values to obtain global min/max
- Operations: `MPI_MIN` and `MPI_MAX`
- All processes receive global values

**Normalization**
- Each process normalizes assigned rows using global min/max
- Scales pixel values to [0, 255] range

**Gather Phase**
- Workers send processed rows to master using `MPI_Send`
- Master collects results using `MPI_Recv`
- Reconstructs complete processed image

### Key Features

**Optimized Communication**
- Non-blocking sends for overlapping communication
- Structured communication with custom tags
- Efficient flattened array layout

**Load Balancing**
- Even row distribution across processes
- Remainder rows assigned to initial processes

**Smart Padding**
- Only necessary padding rows sent
- Edge replication at image boundaries

**Collective Operations**
- `MPI_Allreduce` for efficient min/max computation
- All processes participate, avoiding master bottleneck

### File Structure
```
mpi/
├── mpi.cpp                      # Main entry point
├── Makefile                     # Build configuration
└── infrastructure/
    ├── entity.h/cpp             # Base class for processing
    ├── master.h/cpp             # Master process implementation
    ├── worker.h/cpp             # Worker process implementation
    └── auxs.h                   # Helper structures and constants
```

---

## 5. MPI + OpenMP Hybrid

Implements a hybrid parallel approach combining MPI for distributed memory parallelism with OpenMP for shared memory parallelism, achieving two-level parallelization.

### Architecture
- **MPI Level**: Distributes work across multiple processes (distributed memory)
  - **Master (Rank 0)**: Coordinates distribution, loads/saves images, participates in processing
  - **Workers (Rank 1-N)**: Receive image chunks, process, and return results
- **OpenMP Level**: Parallelizes computation within each MPI process (shared memory)
  - Multiple threads per process parallelize loops and reductions

### Pipeline Stages

**Scatter Phase (MPI)**
- Master divides image by rows among all MPI processes
- Each process receives row block with padding
- Non-blocking `MPI_Isend` to overlap communication

**Convolution (MPI + OpenMP)**
- **MPI**: Each process handles assigned rows
- **OpenMP**: Threads parallelize row processing within each process
  - `#pragma omp parallel for schedule(static)` distributes rows across threads
  - Static scheduling for predictable load distribution

**Min/Max Reduction (OpenMP + MPI)**
- **OpenMP First**: Each process uses OpenMP reduction for local min/max
  - `#pragma omp parallel for reduction(min:localMin) reduction(max:localMax) collapse(2)`
  - Collapses nested loops for better parallelization
- **MPI Second**: `MPI_Allreduce` combines local values across processes

**Normalization (MPI + OpenMP)**
- **MPI**: Each process normalizes assigned rows
- **OpenMP**: `#pragma omp parallel for` parallelizes pixel normalization

**Result Copying (OpenMP)**
- `#pragma omp parallel for collapse(2)` parallelizes data copying
- Efficient 2D loop parallelization

**Gather Phase (MPI)**
- Workers send processed rows to master
- Master reconstructs complete image

### Key Features

**Two-Level Parallelism**
- **Coarse-grained (MPI)**: Distributes chunks across processes
- **Fine-grained (OpenMP)**: Parallelizes within chunks
- Maximizes hardware utilization on multi-node, multi-core systems

**Optimized OpenMP Directives**
- Static scheduling for predictable work distribution
- Collapse clause for better thread utilization
- Reduction clause prevents race conditions

**Load Balancing**
- MPI level: Even row distribution across processes
- OpenMP level: Automatic load balancing within processes

**Communication Efficiency**
- Non-blocking sends for communication overlap
- Collective operations for efficient reductions
- Minimized data transfer through smart padding

### File Structure
```
mpi_openmp/
├── mpi_omp.cpp                  # Main entry point
├── Makefile                     # Build configuration
└── infrastructure/
    ├── entity.h/cpp             # Base class with OpenMP operations
    ├── master.h/cpp             # Master process implementation
    ├── worker.h/cpp             # Worker process implementation
    └── auxs.h                   # Helper structures and constants
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
---