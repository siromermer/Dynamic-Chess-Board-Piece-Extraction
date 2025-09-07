#include "chessboard_analyzer.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <iomanip>

// YOLO configuration constants
const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;
const float SCORE_THRESHOLD = 0.5;
const float NMS_THRESHOLD = 0.5;
const float CONFIDENCE_THRESHOLD = 0.5;

// constructor, it calls initializeSquares to setup 64 empty squares
// you can create instance of class like this: ChessboardAnalyzer analyzer; --> check main.cpp line 110
ChessboardAnalyzer::ChessboardAnalyzer() {
    initializeSquares();
}

// destructor, it automatically cleans up resources when object goes out of scope
ChessboardAnalyzer::~ChessboardAnalyzer() {}

// helper method to initialize 64 empty squares, it doesnt return anything
void ChessboardAnalyzer::initializeSquares() {
    squares_.resize(64);
    for (int i = 0; i < 64; i++) {
        squares_[i].square_id = i;
        squares_[i].piece = "empty";
        squares_[i].confidence = 0.0f;
    }
}

// method to load square coordinates from CSV file, it returns true if successful, false otherwise
bool ChessboardAnalyzer::loadSquareCoordinates(const std::string& csv_file_path) {
    std::ifstream file(csv_file_path);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open CSV file: " << csv_file_path << std::endl;
        return false;
    }
    
    std::string line;
    // Skip header line
    std::getline(file, line);
    
    int square_idx = 0;
    while (std::getline(file, line) && square_idx < 64) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<float> coordinates;
        
        // Parse CSV line to get 8 coordinates (x1,y1,x2,y2,x3,y3,x4,y4)
        while (std::getline(ss, cell, ',')) {
            try {
                coordinates.push_back(std::stof(cell));
            } catch (const std::exception& e) {
                std::cerr << "Error parsing coordinate: " << cell << std::endl;
                return false;
            }
        }
        
        if (coordinates.size() != 8) {
            std::cerr << "Error: Expected 8 coordinates per square, got " << coordinates.size() << std::endl;
            return false;
        }
        
        // Store the four corner points
        squares_[square_idx].corners[0] = cv::Point2f(coordinates[0], coordinates[1]);
        squares_[square_idx].corners[1] = cv::Point2f(coordinates[2], coordinates[3]);
        squares_[square_idx].corners[2] = cv::Point2f(coordinates[4], coordinates[5]);
        squares_[square_idx].corners[3] = cv::Point2f(coordinates[6], coordinates[7]);
        
        // Calculate center point
        squares_[square_idx].center = calculateSquareCenter(squares_[square_idx]);
        
        square_idx++;
    }
    
    file.close();
    
    if (square_idx != 64) {
        std::cerr << "Error: Expected 64 squares, loaded " << square_idx << std::endl;
        return false;
    }
    
    std::cout << "Successfully loaded " << square_idx << " square coordinates from CSV" << std::endl;
    return true;
}

// helper method to calculate center point of a square given its corners
cv::Point2f ChessboardAnalyzer::calculateSquareCenter(const ChessSquare& square) {
    cv::Point2f center(0, 0);
    for (int i = 0; i < 4; i++) {
        center.x += square.corners[i].x;
        center.y += square.corners[i].y;
    }
    center.x /= 4.0f;
    center.y /= 4.0f;
    return center;
}

// method to load class names from file, it returns a vector of class names
std::vector<std::string> ChessboardAnalyzer::loadClassList(const std::string& classes_file_path) {
    std::vector<std::string> class_list;
    std::ifstream file(classes_file_path);
    
    if (!file.is_open()) {
        std::cerr << "Error: Could not open classes file: " << classes_file_path << std::endl;
        return class_list;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            class_list.push_back(line);
        }
    }
    
    file.close();
    std::cout << "Loaded " << class_list.size() << " classes from file" << std::endl;
    return class_list;
}

// helper method to load YOLOv5 model
void ChessboardAnalyzer::loadYOLONetwork(const std::string& model_path) {
    try {
        net_ = cv::dnn::readNet(model_path);
        std::cout << "Running YOLO on CPU" << std::endl;
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    } catch (const cv::Exception& e) {
        std::cerr << "Error loading YOLO model: " << e.what() << std::endl;
        throw;
    }
}

// helper method to format image for YOLOv5 input
cv::Mat ChessboardAnalyzer::formatYolov5(const cv::Mat& source) {
    int col = source.cols;
    int row = source.rows;
    int _max = std::max(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}

// method to perform YOLO inference on image, it returns ChessPieceDetection objects
std::vector<ChessPieceDetection> ChessboardAnalyzer::performYOLOInference(const cv::Mat& image, 
                                                                         const std::string& model_path,
                                                                         const std::vector<std::string>& class_names) {
    std::vector<ChessPieceDetection> detections;
    
    try {
        // Load network if not already loaded
        if (net_.empty()) {
            loadYOLONetwork(model_path);
        }
        
        cv::Mat blob;
        auto input_image = formatYolov5(image);
        
        // Convert the image into a blob
        cv::dnn::blobFromImage(input_image, blob, 1./255., cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);
        net_.setInput(blob);
        
        std::vector<cv::Mat> outputs;
        net_.forward(outputs, net_.getUnconnectedOutLayersNames());
        
        // Scaling factors
        float x_factor = input_image.cols / INPUT_WIDTH;
        float y_factor = input_image.rows / INPUT_HEIGHT;
        
        if (outputs.empty()) {
            std::cerr << "No outputs from YOLO network!" << std::endl;
            return detections;
        }
        
        float *data = (float *)outputs[0].data;
        int total_elements = outputs[0].total();
        int actual_rows = outputs[0].size[1];
        int actual_dimensions = outputs[0].dims >= 3 ? outputs[0].size[2] : 85;
        
        std::vector<int> class_ids;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;
        
        // Process detections
        for (int i = 0; i < actual_rows; ++i) {
            if (i * actual_dimensions + 4 >= total_elements) {
                break;
            }
            
            float confidence = data[4];
            
            if (confidence >= CONFIDENCE_THRESHOLD) {
                float* classes_scores = data + 5;
                cv::Mat scores(1, class_names.size(), CV_32FC1, classes_scores);
                cv::Point class_id;
                double max_class_score;
                cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
                
                if (max_class_score > SCORE_THRESHOLD) {
                    confidences.push_back(confidence);
                    class_ids.push_back(class_id.x);
                    
                    float x = data[0];
                    float y = data[1];
                    float w = data[2];
                    float h = data[3];
                    int left = int((x - 0.5 * w) * x_factor);
                    int top = int((y - 0.5 * h) * y_factor);
                    int width = int(w * x_factor);
                    int height = int(h * y_factor);
                    boxes.push_back(cv::Rect(left, top, width, height));
                }
            }
            data += actual_dimensions;
        }
        
        // Apply Non-Maximum Suppression
        std::vector<int> nms_result;
        cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);
        
        // Create detection objects
        for (int i = 0; i < nms_result.size(); i++) {
            int idx = nms_result[i];
            
            ChessPieceDetection detection;
            detection.class_id = class_ids[idx];
            detection.confidence = confidences[idx];
            detection.box = boxes[idx];
            detection.class_name = class_names[class_ids[idx]];
            
            // Calculate center point of detection
            detection.center.x = detection.box.x + detection.box.width / 2.0f;
            detection.center.y = detection.box.y + detection.box.height / 2.0f;
            
            detections.push_back(detection);
        }
        
        std::cout << "YOLO inference complete. Found " << detections.size() << " chess pieces" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception in YOLO inference: " << e.what() << std::endl;
    }
    
    return detections;
}

void ChessboardAnalyzer::saveDetectionImage(const cv::Mat& image, const std::vector<ChessPieceDetection>& detections, 
                                           const std::string& output_path) {
    cv::Mat detection_image = image.clone();
    
    // Draw bounding boxes and labels for each detection
    for (const auto& detection : detections) {
        // Draw bounding box
        cv::rectangle(detection_image, detection.box, cv::Scalar(0, 255, 0), 2);
        
        // Prepare label text with class name and confidence
        std::string label = detection.class_name + " (" + 
                           std::to_string((int)(detection.confidence * 100)) + "%)";
        
        // Get text size for background rectangle
        int baseline = 0;
        cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);
        
        // Draw background rectangle for text
        cv::rectangle(detection_image, 
                     cv::Point(detection.box.x, detection.box.y - textSize.height - 5),
                     cv::Point(detection.box.x + textSize.width, detection.box.y),
                     cv::Scalar(0, 255, 0), cv::FILLED);
        
        // Draw label text
        cv::putText(detection_image, label, 
                   cv::Point(detection.box.x, detection.box.y - 5),
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);
                   
        // Draw center point
        cv::circle(detection_image, detection.center, 3, cv::Scalar(255, 0, 0), -1);
    }
    
    // Save the detection image
    cv::imwrite(output_path, detection_image);
}

// helper method to check if a point is inside a square using point-in-polygon test
bool ChessboardAnalyzer::isPointInSquare(const cv::Point2f& point, const ChessSquare& square) {
    // Use ray casting algorithm to check if point is inside the quadrilateral
    std::vector<cv::Point2f> polygon;
    for (int i = 0; i < 4; i++) {
        polygon.push_back(square.corners[i]);
    }
    
    double result = cv::pointPolygonTest(polygon, point, false);
    return result >= 0;  // >= 0 means inside or on the boundary
}

// method to match detections to squares using point-in-polygon test
void ChessboardAnalyzer::matchDetectionsToSquares(const std::vector<ChessPieceDetection>& detections) {
    // First, reset all squares to empty
    for (auto& square : squares_) {
        square.piece = "empty";
        square.confidence = 0.0f;
    }
    
    // Match each detection to the best square
    for (const auto& detection : detections) {
        int best_square = -1;
        float min_distance = std::numeric_limits<float>::max();
        
        // Check all squares to find the best match
        for (int i = 0; i < 64; i++) {
            if (isPointInSquare(detection.center, squares_[i])) {
                // Calculate distance from detection center to square center
                float dx = detection.center.x - squares_[i].center.x;
                float dy = detection.center.y - squares_[i].center.y;
                float distance = std::sqrt(dx*dx + dy*dy);
                
                // Choose the closest square if multiple squares contain the point
                if (distance < min_distance) {
                    min_distance = distance;
                    best_square = i;
                }
            }
        }
        
        // Assign the piece to the best matching square
        if (best_square >= 0) {
            // If square already has a piece, keep the one with higher confidence
            if (squares_[best_square].piece == "empty" || 
                detection.confidence > squares_[best_square].confidence) {
                squares_[best_square].piece = detection.class_name;
                squares_[best_square].confidence = detection.confidence;
                
                std::cout << "Assigned " << detection.class_name 
                          << " (confidence: " << std::fixed << std::setprecision(2) << detection.confidence 
                          << ") to square " << best_square << std::endl;
            }
        } else {
            std::cout << "Warning: Could not assign " << detection.class_name 
                      << " at (" << detection.center.x << ", " << detection.center.y 
                      << ") to any square" << std::endl;
        }
    }
}

// method to generate 8x8 chessboard representation as text
std::string ChessboardAnalyzer::generateChessboardRepresentation() {
    std::ostringstream oss;
    
    // Generate 8x8 representation (row by row, from rank 8 to rank 1)
    for (int rank = 7; rank >= 0; rank--) {
        for (int file = 0; file < 8; file++) {
            int square_idx = rank * 8 + file;
            
            if (squares_[square_idx].piece == "empty") {
                oss << "empty ";
            } else {
                oss << squares_[square_idx].piece << " ";
            }
        }
        oss << std::endl;
    }
    
    return oss.str();
}

// method to generate chessboard representation as 2D array
std::vector<std::vector<std::string>> ChessboardAnalyzer::generateChessboard2D() {
    std::vector<std::vector<std::string>> board(8, std::vector<std::string>(8));
    
    for (int rank = 0; rank < 8; rank++) {
        for (int file = 0; file < 8; file++) {
            int square_idx = rank * 8 + file;
            board[7-rank][file] = squares_[square_idx].piece; 
        }
    }
    
    return board;
}

// method to print chessboard to console
void ChessboardAnalyzer::printChessboard() {
    std::cout << std::endl << "=== CHESSBOARD REPRESENTATION ===" << std::endl;
    std::cout << "   a        b        c        d        e        f        g        h" << std::endl;
    
    auto board = generateChessboard2D();
    for (int rank = 0; rank < 8; rank++) {
        std::cout << (8-rank) << " ";
        for (int file = 0; file < 8; file++) {
            std::cout << std::left << std::setw(9) << board[rank][file];
        }
        std::cout << " " << (8-rank) << std::endl;
    }
    
    std::cout << "   a        b        c        d        e        f        g        h" << std::endl;
    std::cout << "===================================" << std::endl;
}

// method to save chessboard representation to the text file
void ChessboardAnalyzer::saveChessboardToFile(const std::string& output_file_path) {
    std::ofstream file(output_file_path);
    if (!file.is_open()) {
        std::cerr << "Error: Could not create output file: " << output_file_path << std::endl;
        return;
    }
    
    file << generateChessboardRepresentation();
    file.close();
    
    std::cout << "Chessboard representation saved to: " << output_file_path << std::endl;
}

