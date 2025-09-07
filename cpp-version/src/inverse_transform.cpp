#include "inverse_transform.h"
#include "config.h"


cv::Mat inverse_transform(const cv::Mat &perspective_matrix)
{

    // Compute the inverse perspective transformation matrix
    cv::Mat M_inv = perspective_matrix.inv();

    std::vector<std::array<cv::Point2f, 5>> squares_data_original;

    for (int i = rows - 1; i >= 0; --i) // bottom row first
    {
        for (int j = 0; j < cols; ++j) // left to right
        {
            // Define the 4 corners of each square
            cv::Point2f top_left(j * square_width, i * square_height);
            cv::Point2f top_right((j + 1) * square_width, i * square_height);
            cv::Point2f bottom_left(j * square_width, (i + 1) * square_height);
            cv::Point2f bottom_right((j + 1) * square_width, (i + 1) * square_height);

            // Calculate the center of the square
            cv::Point2f center((top_left.x + bottom_right.x) / 2.0f,
                               (top_left.y + bottom_right.y) / 2.0f);

            // Prepare array in the same order as Python: center, bottom-right, top-right, top-left, bottom-left
            std::array<cv::Point2f, 5> square_warped = {center, bottom_right, top_right, top_left, bottom_left};

            // Convert to cv::Mat to apply perspective transform
            cv::Mat square_mat(square_warped.size(), 1, CV_32FC2);
            for (size_t k = 0; k < square_warped.size(); ++k)
                square_mat.at<cv::Point2f>(k, 0) = square_warped[k];

            cv::Mat square_original_mat;
            cv::perspectiveTransform(square_mat, square_original_mat, M_inv);

            // Convert back to std::array
            std::array<cv::Point2f, 5> square_original;
            for (size_t k = 0; k < 5; ++k)
                square_original[k] = square_original_mat.at<cv::Point2f>(k, 0);

            squares_data_original.push_back(square_original);
        }
    }

        // Copy base image to draw
    cv::Mat image_drawn = base_image.clone();

    // Draw all squares
    for (size_t idx = 0; idx < squares_data_original.size(); ++idx)
    {
        auto &square = squares_data_original[idx];

        cv::Point top_left(static_cast<int>(square[3].x), static_cast<int>(square[3].y));
        cv::Point top_right(static_cast<int>(square[2].x), static_cast<int>(square[2].y));
        cv::Point bottom_right(static_cast<int>(square[1].x), static_cast<int>(square[1].y));
        cv::Point bottom_left(static_cast<int>(square[4].x), static_cast<int>(square[4].y));

        int row = idx / cols;
        int col = idx % cols;

        // Draw necessary lines to form grid
        cv::line(image_drawn, top_left, top_right, cv::Scalar(0, 255, 0), 6);    // Top
        cv::line(image_drawn, top_left, bottom_left, cv::Scalar(0, 255, 0), 6);  // Left

        // Draw right line for last column
        if (col == cols - 1)
            cv::line(image_drawn, top_right, bottom_right, cv::Scalar(0, 255, 0), 8);

        // Draw bottom line for bottom row
        if (row == 0)
            cv::line(image_drawn, bottom_left, bottom_right, cv::Scalar(0, 255, 0), 8);
    }

    // Return the drawn image
    return image_drawn;

}