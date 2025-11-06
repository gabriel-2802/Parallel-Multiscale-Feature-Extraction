#include "Image.h"
#include <iostream>
#include <stdexcept>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image.h"
#include "stb/stb_image_write.h"

GreyScaleImage::GreyScaleImage() = default;

GreyScaleImage::GreyScaleImage(const std::string& filename) {
    load(filename);
}

GreyScaleImage::~GreyScaleImage() {
    if (data)
        stbi_image_free(data);
}

bool GreyScaleImage::load(const std::string& filename) {
    int w, h, c;
    data = stbi_load(filename.c_str(), &w, &h, &c, CHANNELS);
    if (!data) {
        std::cerr << "Failed to load image: " << filename << std::endl;
        return false;
    }

    width = w;
    height = h;
    channels = CHANNELS;

    pixels.assign(height, std::vector<double>(width, 0));
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            pixels[y][x] = static_cast<double>(data[y * width + x]);
        }
    }

    return true;
}

void GreyScaleImage::setMatrix(const std::vector<std::vector<double>>& matrix) {
    pixels = matrix;
    height = static_cast<int>(matrix.size());
    width = height > 0 ? static_cast<int>(matrix[0].size()) : 0;

    if (data) {
        stbi_image_free(data);
    }

    data = new unsigned char[width * height * channels];
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            data[y * width + x] = static_cast<unsigned char>(pixels[y][x]);
        }
    }
}

void GreyScaleImage::save(const std::string& filename) const {
    if (!data) {
        throw std::runtime_error("No image data to save.");
    }

    if (!stbi_write_png(filename.c_str(), width, height, channels, data, width * channels)) {
        throw std::runtime_error("Failed to save image: " + filename);
    }
}

const std::vector<std::vector<double>>& GreyScaleImage::getMatrix() const {
    return pixels;
}

int GreyScaleImage::getWidth() const {
    return width;
}

int GreyScaleImage::getHeight() const {
    return height;
}
