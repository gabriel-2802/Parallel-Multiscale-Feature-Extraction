#pragma once
#include "entity.h"
#include "../helpers/image.h"
#include "aux.h"
#include <memory>

class Master : public Entity {
public:
    Master(int numtasks, int rank, std::string inputImagePath, std::string outputImagePath);
    ~Master() override;
    void run() override;

private:
    
    std::string outImagePath;
    std::unique_ptr<GreyScaleImage> image;

    void scatter(LAYER layer);
    int getPaddingForLayer(LAYER layer);
    void gatherAndSaveLayer();
    void saveImage();
};