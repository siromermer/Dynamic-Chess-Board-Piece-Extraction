 


## Dynamic Chess Board-Pieces Extraction



* Extracting of Chess Board and Pieces to Classic Chess Format (lichess.com, chess.com)
* Different chess boards , different pieces can be used ( check below of the page )
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
* extracted-data --> It contains result (converted image), and all the information (coordinates,board ..)
* test-images --> images for testing
* chess-model-yolov8m.pt --> trained Yolov8 model for inferencing
* example-results --> You can see different images and results
  
* Dephtai-chess (Folder) --> It contains real-life camera version with depthai library , It is not updated but it can still be used. I have different and better algorithms but this depthai version is not using them , it is old . I will update it
(I dont recommend to use, not updated)
<br><br>

VERY IMPORTANT: I didn't train the model enough because the first phase of this project was extracting the board and pieces dynamically with changing positions and different boards. As a result, the model cannot predict all the pieces correctly, but the positions are nearly perfect. You can train better models and use them with this code. In the future, I will train better models.
<br><br>

### Example Images 
for more example, check  example-results folder
<br><br> 

![sample7](https://github.com/user-attachments/assets/f6659085-cd22-448a-9429-96fa23842f84)
<br><br><br><br>
![sample5](https://github.com/user-attachments/assets/325fdd0d-337c-46f2-87d7-a6641b594aaf)

 
