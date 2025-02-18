import cv2
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
from ultralytics import YOLO
import  math
import ultralytics
import csv
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
from ultralytics import YOLO
from PIL import Image
import os
import chess
import chess.svg

## Extracting Chess Squares with Perspective Transformation ( image --> fen format)

# Path of Image that you want to convert
image_path = r"test-images/test-14.jpeg"
# read image and convert it to different color spaces 
image = cv2.imread(image_path)
gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
rgb_image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)


## Processing Image  -->  OTSU Threshold , Canny edge detection , dilate , HoughLinesP 

# OTSU threshold
ret, otsu_binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Canny edge detection
canny_image = cv2.Canny(otsu_binary, 20, 255)

# Dilation
kernel = np.ones((7, 7), np.uint8)
dilation_image = cv2.dilate(canny_image, kernel, iterations=1)

# Hough Lines
lines = cv2.HoughLinesP(dilation_image, 1, np.pi / 180, threshold=500, minLineLength=150, maxLineGap=100)


# Create an image that contains only black pixels
black_image = np.zeros_like(dilation_image)

# Draw only lines that are output of HoughLinesP function to the "black_image"
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # draw only lines to the "black_image"
        cv2.line(black_image, (x1, y1), (x2, y2), (255, 255, 255), 2)

# Dilation
kernel = np.ones((3, 3), np.uint8)
black_image = cv2.dilate(black_image, kernel, iterations=1)



"""
 Find Contours , sort contours points(4 point), and display valid squares on new fully black image
 By saying "valid squares" , I mean geometrically. With some threshold value , 4 length of a square must be close to each other 
  4 point --> bottomright , topright , topleft , bottomleft
"""  

# Look for valid squares and check if squares are inside of board

# find contours
board_contours, hierarchy = cv2.findContours(black_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# blank image for displaying all contours
all_contours_image= np.zeros_like(black_image)

# Copy blank image for displaying all squares 
squares_image = np.copy(image) 

# blank image for displaying valid contours (squares)
valid_squares_image = np.zeros_like(black_image)

 

# loop through contours and filter them by deciding if they are potential squares
for contour in board_contours:
    if 2000 < cv2.contourArea(contour) < 20000:

        # Approximate the contour to a simpler shape
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # if polygon has 4 vertices
        if len(approx) == 4:

            # 4 points of polygon
            pts = [pt[0].tolist() for pt in approx]

            # create same pattern for points , bottomright(1) , topright(2) , topleft(3) , bottomleft(4)
            index_sorted = sorted(pts, key=lambda x: x[0], reverse=True)

            #  Y values
            if index_sorted[0][1]< index_sorted[1][1]:
                cur=index_sorted[0]
                index_sorted[0] =  index_sorted[1]
                index_sorted[1] = cur

            if index_sorted[2][1]> index_sorted[3][1]:
                cur=index_sorted[2]
                index_sorted[2] =  index_sorted[3]
                index_sorted[3] = cur

            # bottomright(1) , topright(2) , topleft(3) , bottomleft(4)
            pt1=index_sorted[0]
            pt2=index_sorted[1]
            pt3=index_sorted[2]
            pt4=index_sorted[3]

            # find rectangle that fits 4 point 
            x, y, w, h = cv2.boundingRect(contour)
            # find center of rectangle 
            center_x=(x+(x+w))/2
            center_y=(y+(y+h))/2

            

            # calculate length of 4 side of rectangle
            l1 = math.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
            l2 = math.sqrt((pt2[0] - pt3[0])**2 + (pt2[1] - pt3[1])**2)
            l3 = math.sqrt((pt3[0] - pt4[0])**2 + (pt3[1] - pt4[1])**2)
            l4 = math.sqrt((pt1[0] - pt4[0])**2 + (pt1[1] - pt4[1])**2)
 
 
            # Create a list of lengths
            lengths = [l1, l2, l3, l4]
            
            # Get the maximum and minimum lengths
            max_length = max(lengths)
            min_length = min(lengths)

            # Check if this length values are suitable for a square , this threshold value plays crucial role for squares ,  
            if (max_length - min_length) <= 35: # 20 for smaller boards  , 50 for bigger , 35 works most of the time 
                valid_square=True
            else:
                valid_square=False
 
            if valid_square:

                # Draw the lines between the points
                cv2.line(squares_image, pt1, pt2, (255, 255, 0), 7)
                cv2.line(squares_image, pt2, pt3, (255, 255, 0), 7)
                cv2.line(squares_image, pt3, pt4, (255, 255, 0), 7)
                cv2.line(squares_image, pt1, pt4, (255, 255, 0), 7)

                # Draw only valid squares to "valid_squares_image"
                cv2.line(valid_squares_image, pt1, pt2, (255, 255, 0), 7)
                cv2.line(valid_squares_image, pt2, pt3, (255, 255, 0), 7)
                cv2.line(valid_squares_image, pt3, pt4, (255, 255, 0), 7)
                cv2.line(valid_squares_image, pt1, pt4, (255, 255, 0), 7)
            
            # Draw only valid squares to "valid_squares_image"
            cv2.line(all_contours_image, pt1, pt2, (255, 255, 0), 7)
            cv2.line(all_contours_image, pt2, pt3, (255, 255, 0), 7)
            cv2.line(all_contours_image, pt3, pt4, (255, 255, 0), 7)
            cv2.line(all_contours_image, pt1, pt4, (255, 255, 0), 7)
            


#### Dilation to the image that contains only valid squares (gemoetrically valid)

# Apply dilation to the valid_squares_image
kernel = np.ones((7, 7), np.uint8)
dilated_valid_squares_image = cv2.dilate(valid_squares_image, kernel, iterations=1)


#### Find biggest contour of image 

# Find contours of dilated_valid_squares_image
contours, _ = cv2.findContours(dilated_valid_squares_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
# take biggest contour 
largest_contour = max(contours, key=cv2.contourArea)

# create black image
biggest_area_image = np.zeros_like(dilated_valid_squares_image)

# draw biggest contour to the image
cv2.drawContours(biggest_area_image,largest_contour,-1,(255,255,255),10)


#### Find 4 extreme point of chess board

# Initialize variables to store extreme points
top_left = None
top_right = None
bottom_left = None
bottom_right = None

# Loop through the contour to find extreme points
for point in largest_contour[:, 0]:
    x, y = point

    if top_left is None or (x + y < top_left[0] + top_left[1]):
        top_left = (x, y)

    if top_right is None or (x - y > top_right[0] - top_right[1]):
        top_right = (x, y)

    if bottom_left is None or (x - y < bottom_left[0] - bottom_left[1]):
        bottom_left = (x, y)

    if bottom_right is None or (x + y > bottom_right[0] + bottom_right[1]):
        bottom_right = (x, y)

# Draw the contour and the extreme points
extreme_points_image = np.zeros_like(dilated_valid_squares_image, dtype=np.uint8)
cv2.drawContours(extreme_points_image, [largest_contour], -1, (255, 255, 255), thickness=2)

# Mark the extreme points
# Mark the extreme points
cv2.circle(extreme_points_image, top_left, 15, (255, 255, 255), -1)  # red for top-left
cv2.circle(extreme_points_image, top_right, 15, (255, 255, 255), -1)  # green for top-right
cv2.circle(extreme_points_image, bottom_left, 15, (255, 255,255), -1)  # blue for bottom-left
cv2.circle(extreme_points_image, bottom_right, 15, (255, 255, 255), -1)  # yellow for bottom-right



#### Apply Perspective Transformation
 
# read image and convert it to different color spaces 
image = cv2.imread(image_path)
rgb_image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

# Define the four source points (replace with actual coordinates)
extreme_points_list = np.float32([top_left, top_right, bottom_left, bottom_right])

threshold = 0  # Extra space on all sides

width, height = 1200 , 1200 

# Define the destination points (shifted by 'threshold' on all sides)
dst_pts = np.float32([
    [threshold, threshold], 
    [width + threshold, threshold], 
    [threshold, height + threshold], 
    [width + threshold, height + threshold]
])

# Compute the perspective transform matrix
M = cv2.getPerspectiveTransform(extreme_points_list, dst_pts)

# Apply the transformation with extra width and height
warped_image = cv2.warpPerspective(rgb_image, M, (width + 2 * threshold, height + 2 * threshold))

cv2.circle(warped_image, (threshold, threshold), 15, (0, 0, 255), -1)   
cv2.circle(warped_image, (width + threshold, threshold), 15, (0, 0, 255), -1)   
cv2.circle(warped_image, (threshold, height + threshold), 15, (0, 0,255), -1)  
cv2.circle(warped_image, (width + threshold, height + threshold), 15, (0, 0, 255), -1)   



#### Divide board to 64 square

# Assuming area_warped is already defined
# Define number of squares (8x8 for chessboard)
rows, cols = 8, 8

# Calculate the width and height of each square
square_width = width // cols
square_height = height // rows

# Draw the squares on the warped image
for i in range(rows):
    for j in range(cols):
        # Calculate top-left and bottom-right corners of each square
        top_left = (j * square_width, i * square_height)
        bottom_right = ((j + 1) * square_width, (i + 1) * square_height)
        
        # Draw a rectangle for each square
        cv2.rectangle(warped_image, top_left, bottom_right, (0, 255, 0), 4)  # Green color, thickness 2




#### Display extracted squares on original image with inverse transformation
  
# read image and convert it to different color spaces 
image = cv2.imread(image_path)
rgb_image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

# Compute the inverse perspective transformation matrix
M_inv = cv2.invert(M)[1]  # Get the inverse of the perspective matrix

rows, cols = 8, 8  # 8x8 chessboard

# Calculate the width and height of each square in the warped image
square_width = width // cols
square_height = height // rows

# List to store squares' data in the correct order (bottom-left first)
squares_data_warped = []

for i in range(rows - 1, -1, -1):  # Start from bottom row and move up
    for j in range(cols):  # Left to right order
        # Define the 4 corners of each square
        top_left = (j * square_width, i * square_height)
        top_right = ((j + 1) * square_width, i * square_height)
        bottom_left = (j * square_width, (i + 1) * square_height)
        bottom_right = ((j + 1) * square_width, (i + 1) * square_height)

        # Calculate center of the square
        x_center = (top_left[0] + bottom_right[0]) // 2
        y_center = (top_left[1] + bottom_right[1]) // 2

        # Append to list in the correct order
        squares_data_warped.append([
            (x_center, y_center),
            bottom_right,
            top_right,
            top_left,
            bottom_left
        ])

# Convert to numpy array for transformation
squares_data_warped_np = np.array(squares_data_warped, dtype=np.float32).reshape(-1, 1, 2)

# Transform all points back to the original image
squares_data_original_np = cv2.perspectiveTransform(squares_data_warped_np, M_inv)

# Reshape back to list format
squares_data_original = squares_data_original_np.reshape(-1, 5, 2)  # (num_squares, 5 points, x/y)


for square in squares_data_original:
    x_center, y_center = tuple(map(int, square[0]))  # Convert to int
    bottom_right = tuple(map(int, square[1]))
    top_right = tuple(map(int, square[2]))
    top_left = tuple(map(int, square[3]))
    bottom_left = tuple(map(int, square[4]))

    # Draw necessary lines only (to form grid)
    cv2.line(rgb_image, top_left, top_right, (0, 255, 0), 6)  # Top line
    cv2.line(rgb_image, top_left, bottom_left, (0, 255, 0), 6)  # Left line

    # Draw bottom and right lines only for last row/column
    if j == cols - 1:
        cv2.line(rgb_image, top_right, bottom_right, (0, 255, 0), 8)  # Right line
    if i == 0:
        cv2.line(rgb_image, bottom_left, bottom_right, (0, 255, 0), 8)  # Bottom line

cv2.circle(rgb_image, (int(extreme_points_list[0][0]),int(extreme_points_list[0][1])), 25, (255, 255, 255), -1)   
cv2.circle(rgb_image,  (int(extreme_points_list[1][0]),int(extreme_points_list[1][1])), 25, (255, 255, 255), -1)   
cv2.circle(rgb_image,  (int(extreme_points_list[2][0]),int(extreme_points_list[2][1])), 25, (255, 255,255), -1)   
cv2.circle(rgb_image,  (int(extreme_points_list[3][0]),int(extreme_points_list[3][1])), 25, (255, 255, 255), -1)   



#### Write coordinate of squares to a csv file

# Write coordinates to CSV file 
with open('extracted-data/board-square-positions-demo.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # columns
    writer.writerow(['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4'])


    for coordinate in squares_data_original:
        center, bottom_right, top_right, top_left, bottom_left = coordinate
        
        writer.writerow([
                bottom_right[0], bottom_right[1],  # x1, y1
                top_right[0], top_right[1],        # x2, y2
                top_left[0], top_left[1],          # x3, y3
                bottom_left[0], bottom_left[1]     # x4, y4
            ])


#### Check coordinates of squares that are inside of CSV file

# Check CSV coordinates
data = pd.read_csv("extracted-data/board-square-positions-demo.csv") # true Coordinatesa

# Read the image

image = cv2.imread(image_path) 

# Loop through each row in the DataFrame and draw polygons
for i, row in data.iterrows():
    pts = []
    for j in range(0, 8, 2):
        pts.append((int(row[j]), int(row[j+1])))
    pts = np.array(pts, np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.circle(image, (int(squares_data_original[i][0][0]),int(squares_data_original[i][0][1])), 3, (0,255,0), 3)
    cv2.polylines(image,[pts],True,(255,255,255),thickness=8)  # Change color and thickness as needed



# for creating csv files for coordinates --> Chess-Board/Board_to_csv.ipynb
coordinates=pd.read_csv("extracted-data/board-square-positions-demo.csv")
coordinates.tail()



# dictionary for every cell's boundary coordinates 
# [[334, 1231], [344, 1139], [262, 1137], [247, 1228]] -->x1,y1,x2,y2,x3,y3,x4,y4
# 64 cell_value in total --> 8x8 board
coord_dict={}

cell=1
for row in coordinates.values:
    coord_dict[cell]=[[row[0],row[1]],[row[2],row[3]],[row[4],row[5]],[row[6],row[7]]]
    cell+=1
    
 

# class values , these values are decided before training
names: ['black-bishop', 'black-king', 'black-knight', 'black-pawn', 'black-queen', 'black-rook', 'white-bishop', 'white-king', 'white-knight', 'white-pawn', 'white-queen', 'white-rook'] # type: ignore
class_dict={0:'black-bishop',1:'black-king',2:'black-knight',3:'black-pawn',4: 'black-queen',5: 'black-rook',
            6:'white-bishop',7:'white-king',8: 'white-knight',9: 'white-pawn',10: 'white-queen',11:'white-rook'}

print("\n\n") 

# YOLOv8  model  
model = YOLO("chess-model-yolov8m.pt") 

# make prediction
results = model(image_path) # path to test image
im_array = results[0].plot(); # plot a BGR numpy array of predictions

print("\n\n") 

# list for cell number and piece id (class value)
game_list=[]

for result in results:  # results is model's prediction     
    for id,box in enumerate(result.boxes.xyxy) : # box with xyxy format, (N, 4)
            
            x1,y1,x2,y2=int(box[0]),int(box[1]),int(box[2]),int(box[3]) # take coordinates 

            # find middle of bounding boxes for x and y 
            x_mid=int((x1+x2)/2) 
            # add padding to y values
            y_mid=int((y1+y2)/2)+25

            for cell_value, coordinates in coord_dict.items():
                x_values = [point[0] for point in coordinates]
                y_values = [point[1] for point in coordinates]
                 
                if (min(x_values) <= x_mid <= max(x_values)) and (min(y_values) <= y_mid <= max(y_values)):
                    a=int(result.boxes.cls[id])

                    print(f" cell :  {cell_value} --> {a} ")
                    # add cell values and piece cell_value(class value
                    game_list.append([cell_value,a]) 
                    break

print("\n\n\n")        

# show game , if cell value exist in game_list , then print piece in that cell , otherwise print space 
chess_str=""
for i in range(1, 65):
    
    for slist in game_list:
        if slist[0] == i:
            print(class_dict[slist[1]], end=" ")
            chess_str+=f" {class_dict[slist[1]]} "
            break
    else:
        print("space", end=" ")
        chess_str+=" space "

    if i % 8 == 0:
        print("\n")
        chess_str+="\n"
 

def parse_coordinates(input_str):
    """
    Parse the input string to extract the positions of the chess pieces.
    """
    rows = input_str.strip().split('\n')
    chess_pieces = []
    for row in rows:  # Reversing rows to invert ranks
        pieces = row.strip().split()
        chess_pieces.extend(pieces)
    return chess_pieces

 
input_str=chess_str

chess_pieces = parse_coordinates(input_str)

board = chess.Board(None)

piece_mapping = {
    'white-pawn': chess.PAWN,
    'black-pawn': chess.PAWN,
    'white-knight': chess.KNIGHT,
    'black-knight': chess.KNIGHT,
    'white-bishop': chess.BISHOP,
    'black-bishop': chess.BISHOP,
    'white-rook': chess.ROOK,
    'black-rook': chess.ROOK,
    'white-queen': chess.QUEEN,
    'black-queen': chess.QUEEN,
    'white-king': chess.KING,
    'black-king': chess.KING,
    'space': None
}

for rank in range(8):
    for file in range(8):
        piece = chess_pieces[rank * 8 + file]
        if piece != 'space':
            color = chess.WHITE if piece.startswith('white') else chess.BLACK
            piece_type = piece_mapping[piece]
            board.set_piece_at(chess.square(file, rank), chess.Piece(piece_type, color))  # Not inverting rank

svgboard = chess.svg.board(board)
with open("extracted-data/2Dboard.svg", "w") as f:
    f.write(svgboard)

 

# Function to convert SVG to PNG
def convert_svg_to_png(svg_file_path, png_file_path):
    # Read the SVG file and convert it to a ReportLab Drawing
    drawing = svg2rlg(svg_file_path)
    # Render the drawing to a PNG file
    renderPM.drawToFile(drawing, png_file_path, fmt='jpeg')
    print(f"Converted {svg_file_path} to {png_file_path}")

# Example usage
svg_file = 'extracted-data/2Dboard.svg'
png_file = 'extracted-data/Extracted-Board.jpeg'
convert_svg_to_png(svg_file, png_file)

original_image = cv2.imread(image_path)
original_image=cv2.cvtColor(original_image,cv2.COLOR_BGR2RGB)
 



plt.figure(figsize=(14, 10))  # Increase the figure size to 18x6 inches


plt.subplot(131)
plt.title(f"{image_path}")
plt.imshow(original_image)

plt.subplot(132)
plt.title("Extracted Squares")
plt.imshow(image)

plt.subplot(133)
plt.title("Converted Image")
plt.imshow(cv2.cvtColor(cv2.imread(png_file),cv2.COLOR_BGR2RGB))

# Save the figure as a PNG file
output_path = 'output_figure.png'
plt.savefig(output_path)

plt.show()  
