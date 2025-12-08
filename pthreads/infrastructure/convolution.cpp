#include "convolution.h"
#include <vector>
#include <climits>

std::vector<std::vector<double>> allocateMatrix(int height, int width) {
    return std::vector<std::vector<double>>(height, std::vector<double>(width, 0.0));
}

void normalizeMatrix(std::vector<std::vector<double>> &matrix)
{
    double minVal = INT_MAX;
    double maxVal = INT_MIN;

    for (const auto &row : matrix)
        for (double v : row) {
            if (v < minVal) minVal = v;
            if (v > maxVal) maxVal = v;
        }

    double range = (maxVal - minVal == 0.0) ? 1.0 : (maxVal - minVal);

    for (auto &row : matrix)
        for (auto &val : row)
            val = 255.0 * (val - minVal) / range;
}