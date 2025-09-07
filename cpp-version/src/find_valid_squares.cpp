#include "find_valid_squares.h"
#include "config.h"

// Helper function to sort points like Python version
void sort_square_points(std::vector<cv::Point2f> &pts)
{
    // Sort points by X descending
    std::sort(pts.begin(), pts.end(), [](const cv::Point2f &a, const cv::Point2f &b) {
        return a.x > b.x;
    });

    // Swap to match Y ordering like Python
    if (pts[0].y < pts[1].y)
        std::swap(pts[0], pts[1]);
    if (pts[2].y > pts[3].y)
        std::swap(pts[2], pts[3]);
}

bool is_valid_square(const std::vector<cv::Point2f>& square)
{
    if (square.size() != 4)
        return false;

    // bottomright(1), topright(2), topleft(3), bottomleft(4)
    // Calculate the lengths of the 4 sides
    double l1 = cv::norm(square[0] - square[1]);
    double l2 = cv::norm(square[1] - square[2]);
    double l3 = cv::norm(square[2] - square[3]);
    double l4 = cv::norm(square[3] - square[0]);

    // Store lengths in vector
    std::vector<double> lengths = {l1, l2, l3, l4};

    // Find max and min lengths
    double max_length = *std::max_element(lengths.begin(), lengths.end());
    double min_length = *std::min_element(lengths.begin(), lengths.end());

    // You can define a tolerance, e.g., 30% difference allowed
    double tolerance = 0.3 * max_length;

    // Valid square/rectangle if sides are roughly equal
    return (max_length - min_length) < tolerance;
}


// Function to find valid squares
cv::Mat find_valid_squares(const std::vector<std::vector<cv::Point>> &contours)
{
    std::vector<std::vector<cv::Point2f>> squares;
    cv::Mat valid_squares_image = cv::Mat::zeros(base_image.size(), CV_8UC1);

    for (size_t i = 0; i < contours.size(); i++)
    {
        double area = cv::contourArea(contours[i]);
        if (area > 2000 && area < 20000)
        {
            std::vector<cv::Point> approx;
            cv::approxPolyDP(contours[i], approx, 0.02 * cv::arcLength(contours[i], true), true);

            if (approx.size() == 4)
            {
                // Convert to Point2f for convenience
                std::vector<cv::Point2f> square_points;
                for (const auto &pt : approx)
                    square_points.push_back(cv::Point2f(pt));

                // Sort points to match pattern: bottomright(1), topright(2), topleft(3), bottomleft(4)
                sort_square_points(square_points);

                bool valid = is_valid_square(square_points);

                // Draw lines on images
                cv::Scalar color(255, 255, 0); // cyan

                if (valid)
                {
                    cv::line(valid_squares_image, square_points[0], square_points[1], color, 7);
                    cv::line(valid_squares_image, square_points[1], square_points[2], color, 7);
                    cv::line(valid_squares_image, square_points[2], square_points[3], color, 7);
                    cv::line(valid_squares_image, square_points[3], square_points[0], color, 7);

                    // Save sorted square
                    squares.push_back(square_points);
                }
            }
        }
    }

    return valid_squares_image;
}
