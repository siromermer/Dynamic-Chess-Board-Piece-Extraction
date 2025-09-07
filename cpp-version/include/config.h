#ifndef CONFIG_H
#define CONFIG_H

#include <string>
#include <opencv2/opencv.hpp>

// Paths
extern std::string image_path;
extern std::string model_path;
extern std::string classes_path;

// Base image
extern cv::Mat base_image;

// Chessboard properties
extern int rows;
extern int cols;

// Perspective transform size
extern int width;
extern int height;

// Square size
extern int square_width;
extern int square_height;

#endif // CONFIG_H
