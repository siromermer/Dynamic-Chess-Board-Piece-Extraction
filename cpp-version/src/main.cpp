
#include <opencv2/opencv.hpp>
#include <algorithm> 
#include <vector>
#include <sstream>
#include <filesystem> 
#include <fstream>
#include <iostream>

#include "config.h"
#include "image_processing.h"
#include "find_contours.h"
#include "find_valid_squares.h"
#include "find_biggest_contour.h"
#include "perspective_transform.h"
#include "inverse_transform.h"
#include "save_squares_to_csv.h"
#include "display_squares_from_csv.h"
#include "chessboard_analyzer.h"
#include "fen_board_generator.h"


int main()
{
    std::cout << "OpenCV version: " << CV_VERSION << std::endl;
    
    // STEP 1: Read the image
    std::cout << "\n STEP 1: Loading Image " << std::endl;
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "ERROR: Could not open or find the image!" << std::endl;
        return -1;
    }
    std::cout << "Image loaded successfully: " << image_path << std::endl;

    // save original image to the extracted-data/extracted-images folder
    cv::imwrite("../extracted-data/extracted-images/1_original_image.jpg", image);


    /*
        STEP 2: Image Processing
        
        Image processing steps:
        1. Convert to grayscale
        2. Apply OTSU thresholding
        3. Perform Canny edge detection
        4. Apply dilation
        5. Detect Hough lines
        6. Apply dilation
    */
    std::cout << "\n STEP 2: Processing Image " << std::endl;
    cv::Mat processed_image = process_image(image);
    // save processed image to the extracted-data/extracted-images folder
    cv::imwrite("../extracted-data/extracted-images/2_processed_image.jpg", processed_image);

    
    /*
        STEP 3: Contour Detection
        
        Detect contours in the processed binary image
        Return the image with contours drawn and the list of contours
    */
    std::cout << "\n STEP 3: Finding Contours " << std::endl;
    auto result_contours = find_contours(processed_image);
    cv::Mat contour_image = result_contours.first;
    auto contours = result_contours.second;

    // save contour image to the extracted-data/extracted-images folder
    cv::imwrite("../extracted-data/extracted-images/3_contour_image.jpg", contour_image);

   
    /*
        STEP 4: Find Valid Squares

        Loop trough contours and find potential valid squares
        Basic geometric checks: Difference between line lengths
        Return binary image with valid squares drawn
    */
    std::cout << "\n STEP 4: Finding Valid Squares " << std::endl;
    cv::Mat valid_square_image = find_valid_squares(contours);

    // save valid square image to the extracted-data/extracted-images folder
    cv::imwrite("../extracted-data/extracted-images/4_valid_square_image.jpg", valid_square_image);

    
    /*
        STEP 5: Chessboard Detection 

        Find the largest contour and its extreme points
        4 extreme points should be found (corners of the chessboard)
        Return the binary image with the largest contour drawn and extreme points
    */
    std::cout << "\n STEP 5: Finding Biggest Contour " << std::endl;
    auto result_biggest_contour = find_biggest_contour(valid_square_image);
    cv::Mat biggest_contour_image = result_biggest_contour.first;
    std::vector<cv::Point> extreme_points = result_biggest_contour.second;

    // save biggest contour image to the extracted-data/extracted-images folder
    cv::imwrite("../extracted-data/extracted-images/5_biggest_contour_image.jpg", biggest_contour_image);

    
    /*
        STEP 6: Perspective Transform

        Apply perspective transform to the extracted chessboard region
        Return the warped (top-down) image and the perspective matrix
    */
    std::cout << "\n STEP 6: Perspective Transform " << std::endl;
    auto result_perspective = perspective_transform(extreme_points);
    cv::Mat warped_image = result_perspective.first;
    cv::Mat perspective_matrix = result_perspective.second;

    // save warped image to the extracted-data/extracted-images folder
    cv::imwrite("../extracted-data/extracted-images/6_warped_image.jpg", warped_image);
    

    /*
        STEP 7: Inverse Transform

        Apply inverse perspective transform to the warped image to see everything is correct on real image
        Return the original image with the chessboard region restored
    */
    std::cout << "\n STEP 7: Inverse Transform " << std::endl;
    cv::Mat inverse_image = inverse_transform(perspective_matrix);

    // save inverse transformed image to the extracted-data/extracted-images folder
    cv::imwrite("../extracted-data/extracted-images/7_inverse_transformed_image.jpg", inverse_image);
    

    /*
        STEP 8: Save Square Coordinates to CSV

        Divide the warped chessboard into an 8x8 grid
        Calculate the corner coordinates of each square
        Save the coordinates to a CSV file
    */
    std::cout << "\n STEP 8: Saving Square Data " << std::endl;
    std::string csv_path = "../extracted-data/board-square-positions-demo.csv";
    save_squares_to_csv(perspective_matrix, csv_path);
    

    // STEP 9: Detect Chess Pieces with trained YOLO model
    std::cout << "\n STEP 9: Chess Piece Detection " << std::endl;
    cv::Mat fen_image; // this will be displayed at the end
    
    try {
        /*
        ChessboardAnalyzer is a class that encapsulates all chess analysis functionalities
        It handles:
        - Loading square coordinates from CSV --> from extracted-data/board-square-positions-demo.csv
        - Performing YOLO inference to detect chess pieces
        - Matching detections to squares on the chessboard
        - Printing the chessboard with detected pieces --> terminal output
        - Saving the chessboard analysis to a text file --> extracted-data/chessboard_analysis.txt
        */
        ChessboardAnalyzer analyzer;
        
        if (analyzer.loadSquareCoordinates(csv_path)) {
            auto class_names = analyzer.loadClassList(classes_path);
            
            if (!class_names.empty()) {
                auto detections = analyzer.performYOLOInference(image, model_path, class_names);
                
                // Save YOLO detection visualization image
                analyzer.saveDetectionImage(image, detections, 
                    "../extracted-data/extracted-images/8_yolo_detections.jpg");
                
                analyzer.matchDetectionsToSquares(detections);
                analyzer.printChessboard();
                
                std::string output_path = "../extracted-data/chessboard_analysis.txt";
                analyzer.saveChessboardToFile(output_path);
                
                std::cout << "Chess analysis complete! Results saved to: " << output_path << std::endl;
                
                // Generate FEN chess board image
                std::cout << "\n STEP 9.1: Generating FEN Chess Board Image " << std::endl;
                FENBoardGenerator fen_generator;

                std::string pieces_folder = "../chess-pieces";
                std::string fen_output_path = "../extracted-data/fen_chessboard.png";

                bool fen_success = fen_generator.generateFENBoard(output_path, pieces_folder, fen_output_path);
                if (fen_success) {
                    std::cout << "FEN chess board image generated successfully" << std::endl;
                    // Save FEN board image to extracted-images folder as well
                    fen_image = cv::imread(fen_output_path);
                    if (!fen_image.empty()) {
                        cv::imwrite("../extracted-data/extracted-images/9_fen_chessboard.jpg", fen_image);
                    }
                } else {
                    std::cout << "WARNING: Could not generate FEN chess board image" << std::endl;
                }
            } else {
                std::cout << "WARNING: Could not load class names" << std::endl;
            }
        } else {
            std::cout << "WARNING: Could not load square coordinates" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cout << "ERROR in chess analysis: " << e.what() << std::endl;
    }
    
    // STEP 10: DISPLAY RESULTS 
    std::cout << "\n STEP 10: Displaying Results " << std::endl;
    cv::Mat csv_image = display_squares_from_csv(csv_path);
    
    // Save final chessboard analysis image
    cv::imwrite("../extracted-data/extracted-images/10_final_chessboard_analysis.jpg", csv_image);

    // Display the final FEN chess board image
    cv::imshow("FEN Chess Board", fen_image);
    cv::waitKey(0);

    return 0;
}