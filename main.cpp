#include "helpers/image.h"


int main() {
    GreyScaleImage *img = new GreyScaleImage("images/image.png");
    img->save("images/image_copy.png");
    delete img;
    return 0;
}