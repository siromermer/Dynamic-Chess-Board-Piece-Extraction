 


## Dynamic Chess Board-Pieces Extraction



* Extracting of Chess Board and Pieces to Classic Chess Format (lichess.com, chess.com)
* There is 2 version of this project --> with image or with camera(real-time extraction)
1) Extract chess board and and pieces with normal image(.jpeg,.png ...) ( Make inference with Ultralytics)
2) Extract chess board and and pieces with OAK-D Lite Camera (using depthai for real-time video and inferencing) ( not updated)


--> OpenCV , Ultralytics ,YOLO , Numpy , Pandas , chess , matplotlib  , Depthai
 
<br>
<p align="center">
<img src="https://github.com/siromermer/Dynamic-Chess-Board-Piece-Extraction/assets/113242649/200cffa0-fa19-49fa-892b-dca6b4914e89" alt="Image 5" width="350" style="display: inline-block; ">
</p>
<br>
<br>
<p align="center">
<img src="https://github.com/user-attachments/assets/109289d6-49d7-48d5-87eb-9c62ffa7aed9" alt="Image 5" height="350" style="display: inline-block; ">
</p>




### FILES & FOLDERS
* main.py --> All the process in one step ( Board extraction , piece prediction and conversion of board to 2D chess format
* step-by-step.ipynb --> It is nearly same with main.py , but it explain all the steps in notebook . I highly recommend to read this notebook, if you want to see all process with images and explanations
* extracted-data --> It contains result (converted image), and all the information (coordinates,board ..)
* test-images --> images for testing
* chess-model-yolov8m.pt --> trained Yolov8 model for inferencing
* example-results --> You can see different images and results
  
* Dephtai-chess (Folder) --> It contains real-life camera version with depthai library , It is not updated but it can still be used. I have different and better algorithms but this depthai version is not using them , it is old . I will update it
(I dont recommend to use, not updated)
<br>
<br><br>

### Example Images 
<br><br>

![sample7](https://github.com/user-attachments/assets/7a20e12c-e686-42b3-90a6-66865601bf0a)
<br><br><br><br>
![sample5](https://github.com/user-attachments/assets/4043c07d-6b24-4419-9cca-d9448595e8eb)

 
