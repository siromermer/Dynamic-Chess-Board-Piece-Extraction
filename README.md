 


## Dynamic Chess Board-Pieces Extraction



* Extracting of Chess Board and Pieces to Classic Chess Format (lichess.com, chess.com)
* --> with image or with camera(real-time extraction)
* There is 2 version of this project
1) Extract chess board and and pieces with normal image(.jpeg,.png ...) ( Make inference with Ultralytics)
2) Extract chess board and and pieces with OAK-D Lite Camera (using depthai for real-time video and inferencing) ( not updated)


--> OpenCV , Ultralytics ,YOLO , Numpy , Pandas , chess , matplotlib  , Depthai
 

<p align="center">
<img src="https://github.com/siromermer/Dynamic-Chess-Board-Piece-Extraction/assets/113242649/200cffa0-fa19-49fa-892b-dca6b4914e89" alt="Image 5" width="350" style="display: inline-block; ">
</p>

<p align="center">
<img src="https://github.com/user-attachments/assets/fa2f9c74-87de-4449-9c22-2311b6729355" alt="Image 5" width="350" style="display: inline-block; ">
</p>
 


### FILES & FOLDERS
* Steps.txt ----> If you want to see all the steps( image processing , camera calibration , board square extracting, piece prediction , mapping coordinates ) ,you can check
* Board-Square-Extraction.ipynb ---> Extraction of Square Positions
* Piece-Extraction.ipynb ----> Extraction of Pieces and their squares , and obtaining classical Chess Format Image (lichess,chess.com)
* Blob-Model -----> It contains converted Yolov8 chess model  

<br>

### Below , I explained all the steps in sequence of images ---> all the steps : check Steps.txt file

<br>

<p align="center">
  <img src="https://github.com/siromermer/Dynamic-Chess-Board-Piece-Extraction/assets/113242649/0e54a33c-b7e9-4eaa-8a5a-8041e529e54f" alt="Image 1" width="500" style="display: inline-block; margin-right: 10px;">
  <img src="https://github.com/siromermer/Dynamic-Chess-Board-Piece-Extraction/assets/113242649/da070ed8-979a-436d-a0cb-de6755b4b4d4" alt="Image 2" width="500" style="display: inline-block; margin-right: 10px;">
  <img src="https://github.com/siromermer/Dynamic-Chess-Board-Piece-Extraction/assets/113242649/2f7e52f1-ad8c-42d0-a5b4-3a768c6f435b" alt="Image 3" width="500" style="display: inline-block; padding-top: 20px; padding-bottom: 20px; margin-right: 10px;">
  <img src="https://github.com/siromermer/Dynamic-Chess-Board-Piece-Extraction/assets/113242649/ec66ecab-f30c-41f8-b543-4080facddd9c" alt="Image 5" width="500" style="display: inline-block; padding-top: 20px; padding-bottom: 20px;">
</p>

 
