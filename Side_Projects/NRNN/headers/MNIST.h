#pragma once

#include <fstream>
#include <string>
#include <vector>
#include <cstdint>
#include <stdexcept>

struct MNIST_Image {
    std::vector<double> pixels;
    uint8_t label;
};

/**
 * @brief Handle reading from the MNIST dataset
 */
class MNIST {
    private:
        std::vector<MNIST_Image> data;

    public:
        /**
         * @brief Load each image with their corresponding label
         * @param image_file Path to the image file
         * @param label_file Path to the label file
         */
        void load(const std::string& image_file, const std::string& label_file);
        
        /**
         * @return The MNIST dataset
         */
        std::vector<MNIST_Image>& get_data();

        /**
         * @brief Save the image to a small file to display later
         * @param image The image to save
         * @param filename The path to the saving file
         */
        void save_image(const MNIST_Image& image, const std::string& filename);

    private:
        /**
         * @param file The file to read
         * @return The first 4 bytes in Littel Indian order
         */
        uint32_t read_uint32(std::ifstream& file);
};