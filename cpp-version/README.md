# Chess Board Analysis - C++ Version

This is the C++ implementation of the dynamic chess board piece extraction system. It uses YOLOv5 for piece detection, and chessboard and square extraction is identical to the Python implementation. 

This is the first phase of the C++ version, so I will update this repository.

For now, only the perspective transform method is implemented, and the model is only for demo purposes. A better model will be trained in the near future.

In C++, there are not many libraries like in Python. That is why the chessboard is prepared directly, and additional chess piece images are used (chess-pieces folder).


## Prerequisites

- OpenCV 4.x
- CMake 3.10 or higher
- C++17 compatible compiler

## Project Structure

### Folders
- `src/` - Source code files
- `include/` - Header files
- `models/` - YOLOv5 model files
- `test-images/` - Sample images for testing
- `extracted-data/` - Output results and CSV files
- `chess-pieces/` - Individual piece images
- `build/` - Build directory (generated)
- `.vscode/` - VS Code configuration

### Source Files
- `main.cpp` - Main application entry point
- `config.cpp` - Configuration settings and paths
- `image_processing.cpp` - Basic image processing operations
- `find_contours.cpp` - Contour detection algorithms
- `find_biggest_contour.cpp` - Largest contour identification
- `find_valid_squares.cpp` - Chess square validation
- `perspective_transform.cpp` - Perspective transformation operations
- `inverse_transform.cpp` - Inverse transformation operations
- `save_squares_to_csv.cpp` - CSV export functionality
- `display_squares_from_csv.cpp` - Visualization from CSV data
- `chessboard_analyzer.cpp` - Chess board analysis logic
- `fen_board_generator.cpp` - FEN notation generation

### Header Files
Each source file has a corresponding header file in the `include/` directory.

### Model Files
- `chess-yolov5m.onnx` - YOLOv5 model in ONNX format
- `chess-yolov5m.pt` - YOLOv5 PyTorch model
- `classes.txt` - Chess piece class labels

## Compilation

### Build Steps
```bash
mkdir build
cd build
cmake ..
cmake --build .
./Debug/main.exe
```

### Build System
The project uses CMake for cross-platform building. The `CMakeLists.txt` file:
- Sets C++ standard requirements
- Links OpenCV libraries
- Includes header directories
- Compiles all source files into the main executable

## Usage

1. Place your chess board image in the `test-images/` folder
2. Update the image path in `config.cpp` if needed
3. Run the compiled executable
4. Results will be saved in the `extracted-data/` folder

## Output

The program generates:
- Processed chess board images
- CSV files with square coordinates
- FEN notation of the board position
- Individual piece images


