#ifndef CHESSBOARD_ANALYZER_H
#define CHESSBOARD_ANALYZER_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

// Structure to hold detection information from YOLO
struct ChessPieceDetection {
    int class_id; 
    float confidence;
    cv::Rect box;
    std::string class_name;
    cv::Point2f center;  // Center point of the detection
};

// Structure to hold square information from CSV
struct ChessSquare {
    int square_id;       // 0-63 representing the square position
    cv::Point2f corners[4];  // Four corner points of the square
    cv::Point2f center;      // Center point of the square
    std::string piece;       // Name of the piece in this square ("empty" if no piece)
    float confidence;        // Confidence of the piece detection
};

class ChessboardAnalyzer {
public:
    ChessboardAnalyzer();  // constructor
    ~ChessboardAnalyzer(); // destructor

    // Method for loading square coordinates from CSV file
    bool loadSquareCoordinates(const std::string& csv_file_path);
    
    // Method for loading class names from file
    std::vector<std::string> loadClassList(const std::string& classes_file_path);

    // Method for performing YOLO inference on image
    std::vector<ChessPieceDetection> performYOLOInference(const cv::Mat& image,
                                                         const std::string& model_path,
                                                         const std::vector<std::string>& class_names);
    
    // Save YOLO detection visualization image
    void saveDetectionImage(const cv::Mat& image, const std::vector<ChessPieceDetection>& detections, 
                           const std::string& output_path);
    
    // Method for matching detections to squares using point-in-polygon test
    void matchDetectionsToSquares(const std::vector<ChessPieceDetection>& detections);
    
    // Method for generating 8x8 chessboard representation as text
    std::string generateChessboardRepresentation();

    // Method for generating chessboard representation as 2D array
    std::vector<std::vector<std::string>> generateChessboard2D();

    // Method for printing chessboard to console
    void printChessboard();

    // Method for saving chessboard representation to the text file
    void saveChessboardToFile(const std::string& output_file_path);
    

// store data and helper functions that class uses internally
private:
    std::vector<ChessSquare> squares_;  // 64 squares of the chessboard
    cv::dnn::Net net_;                  // YOLO network
    
    // Helper methods
    bool isPointInSquare(const cv::Point2f& point, const ChessSquare& square);
    cv::Point2f calculateSquareCenter(const ChessSquare& square);
    void initializeSquares();
    cv::Mat formatYolov5(const cv::Mat& source);
    void loadYOLONetwork(const std::string& model_path);
};

#endif // CHESSBOARD_ANALYZER_H
