#include "image_processing.h"

cv::Mat process_image(cv::Mat &image)
{
    // Convert to grayscale first (ensure single channel)
    cv::Mat gray;
    // convert to grayscale    
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
 
    // Binary image using Otsu's method
    cv::Mat otsu_binary;
    cv::threshold(gray, otsu_binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    // Canny edge detection
    cv::Mat edges;
    cv::Canny(otsu_binary, edges, 20, 200);

    // Apply dilation to the edges with 7x7 kernel
    cv::Mat dilated_edges;
    cv::dilate(edges, dilated_edges, cv::Mat::ones(7, 7, CV_8UC1));

    // Find lines with Hough Transform
    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(dilated_edges, lines, 1, CV_PI / 180, 500, 150, 100);

    // Draw lines on a single-channel image
    cv::Mat black_image = cv::Mat::zeros(dilated_edges.size(), CV_8UC1);

    // loop over lines and draw them
    for (size_t i = 0; i < lines.size(); i++) {
        cv::line(black_image, cv::Point(lines[i][0], lines[i][1]),
                 cv::Point(lines[i][2], lines[i][3]),
                 cv::Scalar(255), 2, cv::LINE_AA);  
    }

    // Dilate the lines to make them more visible
    cv::Mat dilated_black_image;
    cv::dilate(black_image, dilated_black_image, cv::Mat::ones(3, 3, CV_8UC1));

    return dilated_black_image;  // single-channel, safe for findContours
}