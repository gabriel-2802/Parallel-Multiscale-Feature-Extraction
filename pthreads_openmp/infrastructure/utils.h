# pragma once
#include <vector>
#include "../helpers/kernels.h"

// enumeration for available layers
enum LAYER {
    ONE,
    TWO,
    THREE,
    NUM_LAYERS
};

// utility function to allocate a 2D matrix
inline std::vector<std::vector<double>> allocateMatrix(int height, int width) {
    return std::vector<std::vector<double>>(height, std::vector<double>(width, 0.0));
}
