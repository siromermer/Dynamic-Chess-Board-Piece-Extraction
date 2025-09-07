#include "perspective_transform.h"
#include "config.h"

std::pair<cv::Mat,cv::Mat> perspective_transform(const std::vector<cv::Point> &extreme_points)
{
    // threshold for shifting the points
    int threshold = 0;

    // Convert extreme_points to Point2f
    std::vector<cv::Point2f> src_points;
    for (const auto &pt : extreme_points)
        src_points.push_back(cv::Point2f(pt));

    // Define destination points (shifted by threshold)
    std::vector<cv::Point2f> dst_points = {
        cv::Point2f(threshold, threshold),                // Top-left
        cv::Point2f(width + threshold, threshold),       // Top-right
        cv::Point2f(threshold, height + threshold),      // Bottom-left
        cv::Point2f(width + threshold, height + threshold) // Bottom-right
    };

    // Compute the perspective transform matrix
    cv::Mat perspective_matrix = cv::getPerspectiveTransform(src_points, dst_points);

    // Apply the perspective transform to the original image
    cv::Mat warped_image;
    cv::warpPerspective(base_image, warped_image, perspective_matrix, cv::Size(width + 2 * threshold, height + 2 * threshold));

    // draw grid on the warped image
    for(int i=0;i<rows;i++)
    {
        for(int j=0;j<cols;j++)
        {
            cv::rectangle(warped_image,cv::Point(j*square_width,i*square_height),cv::Point((j+1)*square_width,(i+1)*square_height),cv::Scalar(0,255,0),3);
        }
    }

    return std::make_pair(warped_image, perspective_matrix);
}
