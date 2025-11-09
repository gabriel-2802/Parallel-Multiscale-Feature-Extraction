#pragma once
#include <string>
#include <vector>
#include <iostream>
#include <stdexcept>

#define WIDTH 4434
#define HEIGHT 3547
#define CHANNELS 1

class GreyScaleImage {
public:
    /*
        default constructor
    */
    GreyScaleImage();

    /*
        constructor that loads an image from a file using stb_image library
        @param filename: path to the image file
    */
    GreyScaleImage(const std::string& filename);

    /*
        destructor to free image data
    */
    ~GreyScaleImage();

    /*
        loads an image from a file
        @param filename: path to the image file
        @return true if loading is successful, false otherwise
    */
    bool load(const std::string& filename);


    void setMatrix(const std::vector<std::vector<double>>& matrix);

    const std::vector<std::vector<double>>& getMatrix() const;

    void setFlattenedMatrix(const std::vector<double>& flatMatrix);

    const std::vector<double> getFlattenedMatrix() const;

    void save(const std::string& filename) const;

    int getWidth() const;

    int getHeight() const;

private:
    unsigned char* data = nullptr;
    int width = 0;
    int height = 0;
    int channels = 0;
    std::vector<std::vector<double>> pixels;
};