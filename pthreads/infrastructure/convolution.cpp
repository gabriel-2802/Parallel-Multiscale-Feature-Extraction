// #include "convolution.h"
// #include <pthread.h>
// #include <vector>
// #include <climits>
// #include <cstddef>


// std::vector<std::vector<double>> allocateMatrix(int height, int width) {
//     return std::vector<std::vector<double>>(height, std::vector<double>(width, 0.0));
// }

// void* normalizationRoutine(void* arg)
// {
//     NormData* data = (NormData*)(arg);
//     auto& matrix = *(data->matrix);

//     double range = (data->globalMax - data->globalMin == 0.0) ? 1.0 : (data->globalMax - data->globalMin);

//     for (int i = data->startRow; i < data->endRow; ++i)
//     {
//         auto &row = matrix[i];
//         for (size_t j = 0; j < row.size(); ++j)
//             row[j] = 255.0 * (row[j] - data->globalMin) / range;
//     }

//     return nullptr;
// }