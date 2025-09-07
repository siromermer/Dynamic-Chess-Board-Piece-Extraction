#include "find_biggest_contour.h"
#include "config.h"

std::pair<cv::Mat,std::vector<cv::Point>> find_biggest_contour(cv::Mat &image)
    {
        // apply dilation to the image with 7x7 kernel
        cv::Mat dilated_image;
        cv::dilate(image, dilated_image, cv::Mat::ones(7, 7, CV_8UC1));

        // array for saving 4 extreme points of the biggest contour
        std::vector<cv::Point> extreme_points(4);
        
        // biggest contour image
        cv::Mat biggest_contour_image = cv::Mat::zeros(base_image.size(), CV_8UC1);


        // find  contours
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(dilated_image, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        
        // find biggest contour and draw it
        double max_area = 0;
        int max_index = -1;
        for (size_t i = 0; i < contours.size(); i++)
        {
            double area = cv::contourArea(contours[i]);
            if (area > max_area)
            {
                max_area = area;
                max_index = static_cast<int>(i);
            }
        }

        if (max_index != -1)
        {   

            // find 4 extreme points of the biggest contour --> topleft, topright, bottomright, bottomleft
            std::vector<cv::Point> contour = contours[max_index];

            // initialize extreme points
            cv::Point topleft, topright, bottomleft, bottomright;
            bool initialized = false;

            for (const auto &pt : contour) {
                int x = pt.x;
                int y = pt.y;

                if (!initialized || (x + y < topleft.x + topleft.y))
                    topleft = pt;

                if (!initialized || (x - y > topright.x - topright.y))
                    topright = pt;

                if (!initialized || (x - y < bottomleft.x - bottomleft.y))
                    bottomleft = pt;

                if (!initialized || (x + y > bottomright.x + bottomright.y))
                    bottomright = pt;

                initialized = true;
             }


            // save extreme points on sequence --> top_left, top_right, bottom_left, bottom_right
            extreme_points[0] = topleft;
            extreme_points[1] = topright;
            extreme_points[2] = bottomleft;
            extreme_points[3] = bottomright;

            std::cout << "Extreme Points:" << std::endl;
            std::cout << "Top Left: (" << topleft.x << ", " << topleft.y << ")" << std::endl;
            std::cout << "Top Right: (" << topright.x << ", " << topright.y << ")" << std::endl;
            std::cout << "Bottom Right: (" << bottomright.x << ", " << bottomright.y << ")" << std::endl;
            std::cout << "Bottom Left: (" << bottomleft.x << ", " << bottomleft.y << ")" << std::endl;

            // draw the contour and draw circles on the extreme points
            cv::circle(biggest_contour_image, topleft, 15, cv::Scalar(255, 255, 255), -1);
            cv::circle(biggest_contour_image, topright, 15, cv::Scalar(255, 255, 255), -1);
            cv::circle(biggest_contour_image, bottomright, 15, cv::Scalar(255, 255, 255), -1);
            cv::circle(biggest_contour_image, bottomleft, 15, cv::Scalar(255, 255, 255), -1);

            cv::drawContours(biggest_contour_image, contours, max_index, cv::Scalar(255, 0, 0), 2);

            cv::Mat test_image= base_image.clone();
            // draw the contour and draw circles on the extreme points
            cv::circle(test_image, topleft, 15, cv::Scalar(255, 255, 255), -1);
            cv::circle(test_image, topright, 15, cv::Scalar(255, 255, 255), -1);
            cv::circle(test_image, bottomright, 15, cv::Scalar(255, 255, 255), -1);
            cv::circle(test_image, bottomleft, 15, cv::Scalar(255, 255, 255), -1);
        }

        return std::make_pair(biggest_contour_image, extreme_points);
    }