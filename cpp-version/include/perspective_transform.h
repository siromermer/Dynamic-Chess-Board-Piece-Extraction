
#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <utility>  // std::pair

std::pair<cv::Mat,cv::Mat> perspective_transform(const std::vector<cv::Point> &extreme_points);