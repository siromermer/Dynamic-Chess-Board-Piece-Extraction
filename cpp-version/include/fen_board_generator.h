#ifndef FEN_BOARD_GENERATOR_H
#define FEN_BOARD_GENERATOR_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

class FENBoardGenerator {
public:
    FENBoardGenerator();
    ~FENBoardGenerator();
    
    // Parse chessboard analysis file and return 8x8 board representation
    std::vector<std::vector<std::string>> parseChessboardAnalysis(const std::string& file_path);
    
    // Create visual chessboard image using piece images
    cv::Mat createChessboardImage(const std::vector<std::vector<std::string>>& board, 
                                 const std::string& pieces_folder, 
                                 int square_size = 80);
    
    // Save the chess board image
    bool saveChessboardImage(const cv::Mat& board_image, const std::string& output_path);
    
    // Main function to generate FEN board from analysis file
    bool generateFENBoard(const std::string& analysis_file, 
                         const std::string& pieces_folder, 
                         const std::string& output_path);

private:
    // Load and resize piece image with transparency support
    cv::Mat loadPieceImage(const std::string& piece_path, int target_size);
    
    // Blend piece image with background (handles transparency)
    void blendPieceOnBoard(cv::Mat& board, const cv::Mat& piece, int x, int y);
    
    // Get piece filename from piece name
    std::string getPieceFilename(const std::string& piece_name);
    
    // Add coordinate labels to the board
    cv::Mat addCoordinateLabels(const cv::Mat& board, int square_size);
};

#endif // FEN_BOARD_GENERATOR_H
