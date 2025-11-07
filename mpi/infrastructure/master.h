#pragma once
#include "entity.h"
#include "../helpers/image.h"

#define MASTER_RANK 0

class Master : public Entity {
public:
    Master(int numtasks, int rank, std::string inputImagePath, std::string outputImagePath);
    ~Master() override;
    void run() override;

private:
    GreyScaleImage* image;
    std::string outImagePath;

    void loadImage();
    void scatter();
    void normalize();
    void gather();
    void saveImage();
};