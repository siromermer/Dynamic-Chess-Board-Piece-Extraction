#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <utility>  // std::pair

cv::Mat find_valid_squares(const std::vector<std::vector<cv::Point>> &contours);