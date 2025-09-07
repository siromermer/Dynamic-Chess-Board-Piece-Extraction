#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <utility>  // std::pair
#include <string>
#include <fstream>
#include <iostream>

void save_squares_to_csv(const cv::Mat &perspective_matrix, const std::string &filename);