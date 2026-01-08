#include <iostream>
#include <chrono>
#include <vector>
#include <climits>
#include <omp.h>
#include "../helpers/image.h"
#include "../helpers/kernels.h"

using namespace std;
using namespace chrono;

vector<vector<double>> applyKernel(const vector<vector<double>> &input,
    const vector<vector<int>> &kernel,
    double divisor, int padding);

void normalizeMatrix(vector<vector<double>> &matrix);

int main() {
    auto start = high_resolution_clock::now();

    GreyScaleImage img("../images/upscaled_upscaled_image.png");
    const auto &inputMat = img.getMatrix();
    vector<vector<double>> layer1, layer2, layer3;

    {
        layer1 = applyKernel(inputMat, LAYER_1_KERNEL, LAYER_1_DIV, LAYER_1_PADDING);
    }

    {
        layer2 = applyKernel(layer1, LAYER_2_KERNEL, LAYER_2_DIV, LAYER_2_PADDING);
    }

    {
        layer3 = applyKernel(layer2, LAYER_3_KERNEL, LAYER_3_DIV, LAYER_3_PADDING);
    }

    img.setMatrix(layer3);
    img.save("../images/output_parallel.png");

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    cout << "Processing time: " << duration.count() << " ms" << endl;
    return 0;
}

vector<vector<double>> applyKernel(
    const vector<vector<double>> &input,
    const vector<vector<int>> &kernel,
    double divisor, int padding)
{
    int height = input.size();
    int width = input[0].size();

    vector<vector<double>> outMat(height, vector<double>(width, 0.0));

    // the outer loop over 'y' rows is splitted among threads
    // each thread works on its own specific rows, writing to different parts of 'outMat'.
    #pragma omp parallel for
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            double sum = 0.0;
            for (int ky = -padding; ky <= padding; ++ky) {
                for (int kx = -padding; kx <= padding; ++kx) {
                    int iy = (y + ky < 0) ? 0 : (y + ky >= height ? height - 1 : y + ky);
                    int ix = (x + kx < 0) ? 0 : (x + kx >= width ? width - 1 : x + kx);
                    sum += input[iy][ix] * kernel[ky + padding][kx + padding];
                }
            }
            outMat[y][x] = sum / divisor;
        }
    }

    normalizeMatrix(outMat);
    return outMat;
}

void normalizeMatrix(vector<vector<double>> &matrix)
{
    double minVal = INT_MAX;
    double maxVal = INT_MIN;
    int height = matrix.size();
    int width = matrix[0].size();

    // reduction to find min and max values
    #pragma omp parallel for reduction(min:minVal) reduction(max:maxVal)
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            double v = matrix[i][j];
            if (v < minVal) minVal = v;
            if (v > maxVal) maxVal = v;
        }
    }

    double range = (maxVal - minVal == 0.0) ? 1.0 : (maxVal - minVal);

    // normalization step where each pixel is processed in parallel
    #pragma omp parallel for
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            matrix[i][j] = 255.0 * (matrix[i][j] - minVal) / range;
        }
    }
}