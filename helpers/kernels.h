#pragma once
#include <vector>


/* layer 1: Extreme Laplacian (Edge Isolation) */
#define LAYER_1_KERNEL std::vector<std::vector<int>>{ \
    { -1, -1, -1, -1, -1 }, \
    { -1,  2,  2,  2, -1 }, \
    { -1,  2, 16,  2, -1 }, \
    { -1,  2,  2,  2, -1 }, \
    { -1, -1, -1, -1, -1 }  \
}
#define LAYER_1_DIV 1.0
#define LAYER_1_PADDING 2

/* layer 2: Deep Difference of Gaussians */
#define LAYER_2_KERNEL std::vector<std::vector<int>>{ \
    { -2, -6, -8, -10, -8, -6, -2 }, \
    { -6, -12, -18, -24, -18, -12, -6 }, \
    { -8, -18,   0,  24,   0, -18, -8 }, \
    { -10, -24,  24, 128,  24, -24, -10 }, \
    { -8, -18,   0,  24,   0, -18, -8 }, \
    { -6, -12, -18, -24, -18, -12, -6 }, \
    { -2, -6, -8, -10, -8, -6, -2 }  \
}
#define LAYER_2_DIV 1.0
#define LAYER_2_PADDING 3

/* layer 3: Structural Reinforcement (High-Pass Sharpen)  */
#define LAYER_3_KERNEL std::vector<std::vector<int>>{ \
    {  0, -3,  0 }, \
    { -3, 16, -3 }, \
    {  0, -3,  0 }  \
}
#define LAYER_3_DIV 1.0
#define LAYER_3_PADDING 1
