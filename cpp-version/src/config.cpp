#include "config.h"

// Initialize variables
std::string image_path = "../test-images/test-1.jpeg";
std::string model_path = "../models/chess-yolov5m.onnx";
std::string classes_path = "../models/classes.txt";

cv::Mat base_image = cv::imread(image_path);

// number of rows and columns on the chessboard
int rows = 8;
int cols = 8;

// size for perspective transform
int width = 1200;
int height = 1200;

// square size for perspective transform
int square_width = width / cols;
int square_height = height / rows;
