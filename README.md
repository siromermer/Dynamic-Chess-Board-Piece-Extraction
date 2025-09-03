# Dynamic Chess Board-Pieces Extraction

**Computer vision system that automatically extracts chess board positions and piece placements from images or real-time video, converting them into standard FEN (Forsyth-Edwards Notation) format.** This project uses YOLO object detection and  image processing techniques to recognize chess pieces and their positions on the board.

## Project Versions

There are two versions of this project:

1) Extract the chessboard and pieces from an image (.jpeg, .png, etc.).
2) Extract the chessboard and pieces using an OAK-D Lite Camera (utilizing DepthAI for real-time video and inference).

<br>

For more project, you can check my personal blog website: https://visionbrick.com/

<br>
<p align="center">
<img src="https://github.com/user-attachments/assets/109289d6-49d7-48d5-87eb-9c62ffa7aed9" alt="Image 5" height="350" style="display: inline-block; ">
</p>
<p align="center">
  Version 1: Conversion from image
</p>
 
<br><br>
<p align="center">
<img src="https://github.com/siromermer/Dynamic-Chess-Board-Piece-Extraction/assets/113242649/200cffa0-fa19-49fa-892b-dca6b4914e89" alt="Image 5" width="350" style="display: inline-block; ">
</p>
<p align="center">
  Version 2: real-time conversion with OAK D Lite camera (depthai)
</p>
<br>

There are two files for converting images to FEN format: the first uses a square-filling algorithm, and the second uses perspective transformation. <br>
* The square-filling algorithm works better with non-angled (straight) images. <br>
* Perspective transformation works better with images taken from different angles.
<br>

### Perspective transformation Method
  
![chess-perspective_transform](https://github.com/user-attachments/assets/25b7af18-932e-4dc4-a8ad-82499cfb945c)

<br>

### Square Filling Method
  
![chess-diagram-square](https://github.com/user-attachments/assets/f62b1cf0-6724-4254-b40d-259fe7ee4c58)

<br><br>

## Installation

### CPU Version (Recommended for most users)
```bash
pip install -r requirements.txt
```

### GPU Version (For CUDA-compatible systems)
```bash
pip install -r requirements-gpu.txt
```

**Note:** The GPU version requires NVIDIA CUDA toolkit to be installed on your system. If you're unsure, start with the CPU version.

### Files 
* square_filling.py: Script for conversion using the square-filling algorithm.
* square_filling-step-by-step.ipynb: Jupyter notebook for visualizing the entire process step by step.
 
* perspective_transformation.py: Script for conversion using perspective transformation.
* perspective_transformation-step-by-step.ipynb: Jupyter notebook for visualizing the entire process step by step.  
* chess-model-yolov8m.pt --> Trained YOLOv8 model for chess piece detection.
  <br>

### Folders
* extracted-data --> It contains result (converted image), and all the information (coordinates,board ..)
  
* test-images -->  Collection of images for testing purposes.
* example-results --> Contains various images along with their corresponding results.
* Dephtai-chess (Folder) --> It contains real-life camera version with depthai library , It is not updated but it can still be used. I have different and better algorithms but this depthai version is not using them , it is old . I will update it
(not updated)
<br><br>

Important Note: I didn't train the model enough because the first phase of this project was extracting the board and pieces dynamically with changing positions and different boards. As a result, the model cannot predict all the pieces correctly, but the positions are nearly perfect. You can train better models and use them with this code. In the future, I will train better models.
<br><br>

### Example Images 
for more example, check  example-results folder
<br><br> 


<p align="center">
  <img src="https://github.com/user-attachments/assets/f6659085-cd22-448a-9429-96fa23842f84" alt="sample7">
</p>
<p align="center">Example of image using the square-filling algorithm.</p>

<br><br>

<p align="center">
  <img src="https://github.com/user-attachments/assets/6106d193-86da-43f1-b263-14a7c4b25eaf" alt="perspective_transformation_result_5">
</p>
<p align="center">Example of conversion using perspective transformation.</p>

<br><br>

<p align="center">
  <img src="https://github.com/user-attachments/assets/325fdd0d-337c-46f2-87d7-a6641b594aaf" alt="sample5">
</p>
<p align="center">Example of square-filling algorithm on an image.</p>

<br><br>


