#include <iostream>
#include <chrono>
#include <vector>
#include <climits>
#include "../helpers/image.h"
#include "../helpers/kernels.h"

using namespace std;
using namespace std::chrono;

std::vector<std::vector<double>> applyKernel(
    const std::vector<std::vector<double>> &input,
    const std::vector<std::vector<int>> &kernel,
    double divisor, int padding)
{
    int height = input.size();
    int width = input[0].size();

    std::vector<std::vector<double>> outMat(height, std::vector<double>(width, 0.0));

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            double sum = 0.0;
            for (int ky = -padding; ky <= padding; ++ky) {
                for (int kx = -padding; kx <= padding; ++kx) {
                    int iy = std::min(std::max(y + ky, 0), height - 1);
                    int ix = std::min(std::max(x + kx, 0), width - 1);
                    sum += input[iy][ix] * kernel[ky + padding][kx + padding];
                }
            }
            outMat[y][x] = sum / divisor;
        }
    }

    normalizeMatrix(outMat);
    return outMat;
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

int main() {
    //load input image
    GreyScaleImage img("../images/image.png");
    cout << "Loaded image: " << img.getWidth() << "x" << img.getHeight() << endl;

    // convert loaded image to a double matrix
    const auto &inputMat = img.getMatrix();
    vector<vector<double>> layer1, layer2, layer3;

    // layer 1
    {
        auto start = high_resolution_clock::now();
        layer1 = applyKernel(inputMat, LAYER_1_KERNEL, LAYER_1_DIV, LAYER_1_PADDING);
        auto stop = high_resolution_clock::now();
        cout << "Layer 1: " << duration_cast<milliseconds>(stop - start).count() << " ms" << endl;
    }

    // layer 2
    {
        auto start = high_resolution_clock::now();
        layer2 = applyKernel(layer1, LAYER_2_KERNEL, LAYER_2_DIV, LAYER_2_PADDING);
        auto stop = high_resolution_clock::now();
        cout << "Layer 2: " << duration_cast<milliseconds>(stop - start).count() << " ms" << endl;
    }

    // layer 3
    {
        auto start = high_resolution_clock::now();
        layer3 = applyKernel(layer2, LAYER_3_KERNEL, LAYER_3_DIV, LAYER_3_PADDING);
        auto stop = high_resolution_clock::now();
        cout << "Layer 3: " << duration_cast<milliseconds>(stop - start).count() << " ms" << endl;
    }

    GreyScaleImage out1("../images/image.png"), out2("../images/image.png"), out3("../images/image.png");
    out1.setMatrix(layer1);
    out2.setMatrix(layer2);
    out3.setMatrix(layer3);
    

    // save results
    out1.save("images_results/1.1.png");
    out2.save("images_results/1.2.png");
    out3.save("images_results/1.3.png");
    return 0;
}
