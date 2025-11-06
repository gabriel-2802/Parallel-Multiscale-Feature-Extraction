#pragma once

// layer 1 — 3×3 Gaussian kernel (divide by 16)
#define GAUSS_KERNEL_3X3 { \
    {1, 2, 1}, \
    {2, 4, 2}, \
    {1, 2, 1}  \
}
#define GAUSS_DIV_3X3 16.0

// layer 2 — 5×5 Gaussian kernel (divide by 256)
#define GAUSS_KERNEL_5X5 { \
    {1,  4,  6,  4,  1}, \
    {4, 16, 24, 16,  4}, \
    {6, 24, 36, 24,  6}, \
    {4, 16, 24, 16,  4}, \
    {1,  4,  6,  4,  1}  \
}
#define GAUSS_DIV_5X5 256.0

// layer 3 — 7×7 Gaussian kernel (divide by 4096)
#define GAUSS_KERNEL_7X7 { \
    { 1,  6,  15,  20,  15,  6,  1}, \
    { 6, 36,  90, 120,  90, 36,  6}, \
    {15, 90, 225, 300, 225, 90, 15}, \
    {20,120, 300, 400, 300,120, 20}, \
    {15, 90, 225, 300, 225, 90, 15}, \
    { 6, 36,  90, 120,  90, 36,  6}, \
    { 1,  6,  15,  20,  15,  6,  1}  \
}
#define GAUSS_DIV_7X7 4096.0
