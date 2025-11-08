#pragma once

#define MASTER_RANK 0

enum LAYER {
    ONE,
    TWO,
    THREE
};

enum COMM_TAGS {
    DIMENSIONS,
    IMAGE_DATA,
    RESULT_DATA,
    MIN_MAX_DATA
};

struct __attribute__((packed)) ProcessDims {
    int totalRows;
    int width;
    int rowsForWorker;
    int padding;
    int offset;

    ProcessDims(int tRows, int w, int rfw, int pad, int off)
        : totalRows(tRows), width(w), rowsForWorker(rfw), padding(pad), offset(off) {}
};

struct __attribute__((packed)) MinMaxVals {
    double min;
    double max;

    MinMaxVals(double minVal, double maxVal)
        : min(minVal), max(maxVal) {}
};

