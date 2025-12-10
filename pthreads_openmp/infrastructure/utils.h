# pragma once
#include <vector>
#include "../helpers/kernels.h"

enum LAYER {
    ONE,
    TWO,
    THREE,
    NUM_LAYERS
};

inline std::vector<std::vector<double>> allocateMatrix(int height, int width) {
    return std::vector<std::vector<double>>(height, std::vector<double>(width, 0.0));
}
