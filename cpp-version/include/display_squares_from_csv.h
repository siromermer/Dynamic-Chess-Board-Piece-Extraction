#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <utility>  
#include <string>
#include <fstream>
#include <iostream>

cv::Mat display_squares_from_csv(const std::string &csv_path);