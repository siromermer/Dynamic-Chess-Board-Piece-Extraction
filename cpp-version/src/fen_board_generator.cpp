#include "fen_board_generator.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <map>

FENBoardGenerator::FENBoardGenerator() {}

FENBoardGenerator::~FENBoardGenerator() {}

std::vector<std::vector<std::string>> FENBoardGenerator::parseChessboardAnalysis(const std::string& file_path) {
    std::vector<std::vector<std::string>> board;
    std::ifstream file(file_path);
    
    if (!file.is_open()) {
        std::cerr << "Error: Could not open analysis file: " << file_path << std::endl;
        return board;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        
        std::vector<std::string> row;
        std::stringstream ss(line);
        std::string piece;
        
        while (ss >> piece) {
            row.push_back(piece);
        }
        
        if (!row.empty()) {
            board.push_back(row);
        }
    }
    
    file.close();
    
    std::cout << "Parsed " << board.size() << " rows from analysis file" << std::endl;
    return board;
}

std::string FENBoardGenerator::getPieceFilename(const std::string& piece_name) {
    std::map<std::string, std::string> piece_mapping = {
        {"white-pawn", "w-pawn.png"},
        {"white-rook", "w-rook.png"},
        {"white-knight", "w-knight.png"},
        {"white-bishop", "w-bishop.png"},
        {"white-queen", "w-queen.png"},
        {"white-king", "w-king.png"},
        {"black-pawn", "b-pawn.png"},
        {"black-rook", "b-rook.png"},
        {"black-knight", "b-knight.png"},
        {"black-bishop", "b-bishop.png"},
        {"black-queen", "b-queen.png"},
        {"black-king", "b-king.png"}
    };
    
    auto it = piece_mapping.find(piece_name);
    return (it != piece_mapping.end()) ? it->second : "";
}

cv::Mat FENBoardGenerator::loadPieceImage(const std::string& piece_path, int target_size) {
    cv::Mat piece_img = cv::imread(piece_path, cv::IMREAD_UNCHANGED);
    
    if (piece_img.empty()) {
        std::cerr << "Warning: Could not load piece image: " << piece_path << std::endl;
        return cv::Mat();
    }
    
    // Resize to target size
    cv::Mat resized_piece;
    cv::resize(piece_img, resized_piece, cv::Size(target_size, target_size));
    
    return resized_piece;
}

void FENBoardGenerator::blendPieceOnBoard(cv::Mat& board, const cv::Mat& piece, int x, int y) {
    if (piece.empty()) return;
    
    int piece_height = piece.rows;
    int piece_width = piece.cols;
    
    // Make sure the piece fits on the board
    if (x + piece_width > board.cols || y + piece_height > board.rows) {
        return;
    }
    
    // Handle transparency if piece has alpha channel
    if (piece.channels() == 4) {
        for (int py = 0; py < piece_height; py++) {
            for (int px = 0; px < piece_width; px++) {
                cv::Vec4b piece_pixel = piece.at<cv::Vec4b>(py, px);
                float alpha = piece_pixel[3] / 255.0f;
                
                if (alpha > 0) {
                    cv::Vec3b& board_pixel = board.at<cv::Vec3b>(y + py, x + px);
                    
                    for (int c = 0; c < 3; c++) {
                        board_pixel[c] = (uchar)(board_pixel[c] * (1 - alpha) + piece_pixel[c] * alpha);
                    }
                }
            }
        }
    } else {
        // No transparency, just copy the piece
        cv::Mat roi = board(cv::Rect(x, y, piece_width, piece_height));
        
        // Convert piece to BGR if it's not already
        if (piece.channels() == 3) {
            piece.copyTo(roi);
        } else if (piece.channels() == 1) {
            cv::cvtColor(piece, roi, cv::COLOR_GRAY2BGR);
        }
    }
}

cv::Mat FENBoardGenerator::createChessboardImage(const std::vector<std::vector<std::string>>& board, 
                                               const std::string& pieces_folder, 
                                               int square_size) {
    // Create empty board (8x8 squares)
    cv::Mat board_img(8 * square_size, 8 * square_size, CV_8UC3);
    
    // Define colors for light and dark squares
    cv::Scalar light_color(181, 217, 240); // Light brown (BGR format)
    cv::Scalar dark_color(99, 136, 181);   // Dark brown (BGR format)
    
    // Create alternating light/dark squares
    for (int row = 0; row < 8; row++) {
        for (int col = 0; col < 8; col++) {
            // Determine square color (alternating pattern)
            cv::Scalar color = ((row + col) % 2 == 0) ? light_color : dark_color;
            
            // Fill the square
            cv::Rect square_rect(col * square_size, row * square_size, square_size, square_size);
            cv::rectangle(board_img, square_rect, color, -1);
        }
    }
    
    // Place pieces on the board
    int piece_size = (int)(square_size * 0.8);
    int margin = (square_size - piece_size) / 2;
    
    for (int row = 0; row < 8 && row < (int)board.size(); row++) {
        for (int col = 0; col < 8 && col < (int)board[row].size(); col++) {
            const std::string& piece_name = board[row][col];
            
            if (piece_name != "empty") {
                std::string piece_filename = getPieceFilename(piece_name);
                
                if (!piece_filename.empty()) {
                    std::string piece_path = pieces_folder + "\\" + piece_filename;
                    cv::Mat piece_img = loadPieceImage(piece_path, piece_size);
                    
                    if (!piece_img.empty()) {
                        int x_pos = col * square_size + margin;
                        int y_pos = row * square_size + margin;
                        
                        blendPieceOnBoard(board_img, piece_img, x_pos, y_pos);
                    }
                }
            }
        }
    }
    
    return board_img;
}

cv::Mat FENBoardGenerator::addCoordinateLabels(const cv::Mat& board, int square_size) {
    int border_size = 30;
    cv::Mat board_with_coords(board.rows + 2 * border_size, board.cols + 2 * border_size, CV_8UC3, cv::Scalar(255, 255, 255));
    
    // Copy the original board to the center
    cv::Rect board_roi(border_size, border_size, board.cols, board.rows);
    board.copyTo(board_with_coords(board_roi));
    
    // Add file labels (a-h) at bottom
    std::vector<std::string> files = {"a", "b", "c", "d", "e", "f", "g", "h"};
    for (int i = 0; i < 8; i++) {
        int x_pos = border_size + i * square_size + square_size / 2 - 10;
        int y_pos = border_size + board.rows + 25;
        
        cv::putText(board_with_coords, files[i], cv::Point(x_pos, y_pos),
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);
    }
    
    // Add rank labels (8-1) at left
    std::vector<std::string> ranks = {"8", "7", "6", "5", "4", "3", "2", "1"};
    for (int i = 0; i < 8; i++) {
        int x_pos = 15;
        int y_pos = border_size + i * square_size + square_size / 2 + 5;
        
        cv::putText(board_with_coords, ranks[i], cv::Point(x_pos, y_pos),
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);
    }
    
    return board_with_coords;
}

bool FENBoardGenerator::saveChessboardImage(const cv::Mat& board_image, const std::string& output_path) {
    if (board_image.empty()) {
        std::cerr << "Error: Empty board image" << std::endl;
        return false;
    }
    
    bool success = cv::imwrite(output_path, board_image);
    if (success) {
        std::cout << "Chess board image saved to: " << output_path << std::endl;
    } else {
        std::cerr << "Error: Could not save image to: " << output_path << std::endl;
    }
    
    return success;
}

bool FENBoardGenerator::generateFENBoard(const std::string& analysis_file, 
                                       const std::string& pieces_folder, 
                                       const std::string& output_path) {
    std::cout << "\n GENERATING FEN CHESS BOARD IMAGE " << std::endl;
    
    // Parse the chessboard analysis
    std::cout << "Reading chessboard analysis from: " << analysis_file << std::endl;
    auto board = parseChessboardAnalysis(analysis_file);
    
    if (board.empty()) {
        std::cerr << "Error: Could not parse chessboard analysis" << std::endl;
        return false;
    }
    
    // Print board representation
    std::cout << "\nBoard representation:" << std::endl;
    for (int i = 0; i < (int)board.size(); i++) {
        std::cout << "Rank " << (8-i) << ": ";
        for (const auto& piece : board[i]) {
            std::cout << piece << " ";
        }
        std::cout << std::endl;
    }
    
    // Create the visual chessboard
    std::cout << "\nCreating visual chessboard..." << std::endl;
    cv::Mat board_image = createChessboardImage(board, pieces_folder, 80);
    
    if (board_image.empty()) {
        std::cerr << "Error: Could not create board image" << std::endl;
        return false;
    }
    
    // Add coordinate labels
    cv::Mat final_board = addCoordinateLabels(board_image, 80);
    
    // Save the image
    bool success = saveChessboardImage(final_board, output_path);
    
    if (success) {
        std::cout << "FEN chess board image created successfully!" << std::endl;
    }
    
    return success;
}
