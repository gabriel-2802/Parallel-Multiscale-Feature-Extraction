#include <iostream>
#include <vector>
#include <pthread.h>

#include "../helpers/image.h"
#include "../helpers/kernels.h"

#include "infrastructure/convolution.h"
#include "infrastructure/worker.h"

using namespace std;

int main() {

    GreyScaleImage img("../images/image.png");

    auto input = img.getMatrix();
    int height = img.getHeight();
    int width = img.getWidth();

    auto output = allocateMatrix(height, width);

    //TODO

    return 0;
}