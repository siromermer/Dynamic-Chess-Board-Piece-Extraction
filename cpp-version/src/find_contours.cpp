#include "find_contours.h"

std::pair<cv::Mat, std::vector<std::vector<cv::Point>>> find_contours(cv::Mat &image)
{
    // Find contours
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    cv::findContours(image, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    // Create an empty image to draw contours on
    cv::Mat contour_image = cv::Mat::zeros(image.size(), CV_8UC1);


    // Draw each contour
    for (size_t i = 0; i < contours.size(); i++)
    {
        cv::Scalar color = cv::Scalar(255, 255, 255);
        cv::drawContours(contour_image, contours, static_cast<int>(i), color, 2, cv::LINE_8, hierarchy, 0);
    }

    // return contour image and contour list
    return std::make_pair(contour_image, contours);
}

