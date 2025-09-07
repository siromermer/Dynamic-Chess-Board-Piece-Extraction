cmake_minimum_required(VERSION 3.10)
project(Chess-Organized)

# Set C++ standard
find_package(OpenCV REQUIRED)

# Get all source files except main executables
file(GLOB LIB_SOURCES 
    src/config.cpp
    src/display_squares_from_csv.cpp
    src/find_biggest_contour.cpp
    src/find_contours.cpp
    src/find_valid_squares.cpp
    src/image_processing.cpp
    src/inverse_transform.cpp
    src/perspective_transform.cpp
    src/save_squares_to_csv.cpp
    src/chessboard_analyzer.cpp
    src/fen_board_generator.cpp
)

# Create original main executable
add_executable(main src/main.cpp ${LIB_SOURCES})


# Link OpenCV libraries
target_link_libraries(main ${OpenCV_LIBS})

# Include headers from include/
target_include_directories(main PRIVATE include)
