import cv2
from enum import Enum

# Notes for integration:
# - image being graded should be in a variable called image, or you can change the variable name

# Enum for solder errors
class sError(Enum):
    excessive = 'E'
    insufficient = 'I'
    shifted = 'R' # R for Rotation
    short = 'S'

# Get PCB image
# image = cv2.imread("<Insert Absolute Path to Image>")

# Dummy List for debugging
# Each element in the list will be a 3 element tuple
# (Upper left corner coordinate (2 int tuple), bottom right corner coordinate(2 int tuple), classification (string))
list_seg_class = [((30,30), (100, 100), "excessive"), ((500,450), (678, 567), "insufficient"), ((1000,600), (1200, 700), "shifted"), ((450,700), (800, 945), "short")]

# Loop thorugh the list of segments and classfications
for seg in list_seg_class:
    # Check that there is a classification
    if(seg[2] != ""):
        # Get the top left x and y coordinates and the bottom left x and y coordinates
        x = seg[0][0]
        y = seg[0][1]
        x2 = seg[1][0]
        y2 = seg[1][1]

        # Draw bounding box
        # Sidenote: Ew, it's bgr not rgb
        image = cv2.rectangle(image, (x, y), (x2, y2), (0,0,255), 1)

        #Assign label to bounding box
        cv2.putText(image, str(sError[seg[2]].value), (x, y-2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)


#Save image
cv2.imwrite("saved_img.jpg", image)