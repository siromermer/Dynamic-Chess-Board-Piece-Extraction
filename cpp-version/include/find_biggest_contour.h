
#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <utility>  // std::pair

std::pair<cv::Mat,std::vector<cv::Point>> find_biggest_contour(cv::Mat &image);