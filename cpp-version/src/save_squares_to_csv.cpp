#include "save_squares_to_csv.h"
#include "config.h"  


void save_squares_to_csv(const cv::Mat &perspective_matrix, const std::string &filename)
{

    // Compute inverse perspective to map points back to original image
    cv::Mat inverse_matrix = perspective_matrix.inv();

    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open CSV file for writing: " << filename << std::endl;
        return;
    }

    // Write CSV header
    file << "x1,y1,x2,y2,x3,y3,x4,y4\n";

    for (int i = rows - 1; i >= 0; i--) {  // bottom row first
        for (int j = 0; j < cols; j++) {   // left to right
            // 4 corners in warped image
            std::vector<cv::Point2f> warped_corners = {
                cv::Point2f(j * square_width, i * square_height),                 // top-left
                cv::Point2f((j + 1) * square_width, i * square_height),           // top-right
                cv::Point2f(j * square_width, (i + 1) * square_height),           // bottom-left
                cv::Point2f((j + 1) * square_width, (i + 1) * square_height)      // bottom-right
            };

            // Map back to original image using inverse perspective
            std::vector<cv::Point2f> original_corners;
            cv::perspectiveTransform(warped_corners, original_corners, inverse_matrix);

            // Write coordinates to CSV in same order as Python example
            file << static_cast<double>(original_corners[3].x) << "," 
                 << static_cast<double>(original_corners[3].y) << ","   // bottom-right
                 << static_cast<double>(original_corners[1].x) << "," 
                 << static_cast<double>(original_corners[1].y) << ","   // top-right
                 << static_cast<double>(original_corners[0].x) << "," 
                 << static_cast<double>(original_corners[0].y) << ","   // top-left
                 << static_cast<double>(original_corners[2].x) << "," 
                 << static_cast<double>(original_corners[2].y) << "\n"; // bottom-left
        }
    }

    file.close();
    std::cout << "CSV file saved: " << filename << std::endl;
}
