#include "convolution.h"
#include <vector>
#include <algorithm>

std::vector<std::vector<double>> allocateMatrix(int height, int width) {
    return std::vector<std::vector<double>>(height, std::vector<double>(width, 0.0));
}

void normalizeMatrix(std::vector<std::vector<double>>& matrix) {
    // TODO
}
