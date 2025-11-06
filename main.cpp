#include "helpers/Image.h"


int main() {
    GreyScaleImage *img = new GreyScaleImage("images/2.png");
    img->save("images/2_copy.png");
    delete img;
    return 0;
}