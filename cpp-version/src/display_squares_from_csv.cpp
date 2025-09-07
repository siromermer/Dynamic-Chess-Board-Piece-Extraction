#include "display_squares_from_csv.h"
#include "config.h"  

cv::Mat display_squares_from_csv(const std::string &csv_path)
{
    cv::Mat csv_image = base_image.clone();
    int square_number =1;

    std::ifstream file(csv_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open CSV file: " << csv_path << std::endl;
        return cv::Mat();
    }

    std::string line;
    bool first_line = true; // skip header
    while (std::getline(file, line)) {
        if (first_line) {
            first_line = false;
            continue;
        }
        
        std::stringstream ss(line);
        std::string value;
        std::vector<int> coords;

        while (std::getline(ss, value, ',')) {
            coords.push_back(std::stoi(value)); // convert CSV string to int
        }

        if (coords.size() != 8) {
            std::cerr << "Invalid number of coordinates in line: " << line << std::endl;
            continue;
        }

        // Prepare points for polygon
        std::vector<cv::Point> pts = {
            cv::Point(coords[0], coords[1]), // bottom-right
            cv::Point(coords[2], coords[3]), // top-right
            cv::Point(coords[4], coords[5]), // top-left
            cv::Point(coords[6], coords[7])  // bottom-left
        };

        // Draw polygon
        const cv::Point* pts_ptr = pts.data();
        int npts = static_cast<int>(pts.size());
        cv::polylines(csv_image, &pts_ptr, &npts, 1, true, cv::Scalar(255,255,255), 8);

        // draw cirlcles on the corners
        for (const auto &pt : pts) {
            cv::circle(csv_image, pt, 10, cv::Scalar(0, 0, 255), -1); // red circles
        }

        // write square number at the center
        int center_x = (coords[0] + coords[4]) / 2;
        int center_y = (coords[1] + coords[5]) / 2;
        cv::putText(csv_image, std::to_string(square_number), cv::Point(center_x, center_y), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
        square_number++;
    }

    file.close();

    return csv_image;
}
