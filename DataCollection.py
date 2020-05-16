# Import necessary libraries
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import os 

# Select camera input (0 for PC internal cam, 1 for external USB cam)
capture = cv2.VideoCapture(1)

#Global Variables
img_size = 128
map_size = 10
grid = np.zeros((map_size,map_size))
rows = 0
cols = 0 
prediction = 0

edgemap = 30
letter = "B"

counter = 1

CATEGORIES = ["Nonpressed", "Pressed"]
model = tf.keras.models.load_model("ContactClassifier-CNN.model")
reshaped = np.zeros((1,img_size,img_size,1))
path_edgemaps = r"C:\Users\alexk\Desktop\Y4\Final Year Project\OpenCV\Extra\{}\edgemaps".format(letter)
A = r"C:\Users\alexk\Desktop\Y4\Final Year Project\OpenCV\Extra\{0}\frames\{1}".format(letter,edgemap)

# Infinite while loop to continuously capture video input
while(True):
    
    ret, frame = capture.read()     # Capture frames from the camera at each loop iteration
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert to grayscale
    cv2.imshow('frame', gray)   # Display the frame
    
    # Break the loop when the escape key is pressed
    key = cv2.waitKey(1) & 0xFF
    
    if (key == ord('c')):   # if key "c" is pressed execute the following code
                
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert to grayscale
        
        file_name1 = "frame{}.png".format(counter) # prepare the frame file name
        counter = counter + 1   #increment counter for filename for next iteration
        cv2.imwrite(os.path.join(A, file_name1),gray) # Save the frame
        
        gray_resize = cv2.resize(gray, (img_size,img_size)) # resize the frame
        reshaped[0,:,:,0] = np.array(gray_resize,'f')# transfer the data into a 4-dimensional array
        prediction = model.predict(reshaped)         # use the saved model to determine if contact occured
        print(CATEGORIES[int(prediction)],rows,cols) # Print the result for validation next to the cell coordinates
        
        # multiply the result by 255 for grayscale and store it in the respective cell in the virtual grid
        grid[rows][cols] = int(prediction) * 255 
        plt.imshow(grid, cmap='gray') # Display the edge map once finished
        
        if (rows == map_size-1 & cols == map_size-1):                 # If scan is finished:
            file_name2 = "edgemap{0}{1}.png".format(letter,edgemap)   # Prepare the file name of the edge map
            cv2.imwrite(os.path.join(path_edgemaps, file_name2),grid) # Save the edge map, and
            break                                                     # Exit the loop
        
        elif (cols == map_size-1): # if last column is reached:
            cols = 0               # Reset the column number and 
            rows = rows + 1        # increment the row number/go to next row
        else:                      # else, 
            cols = cols + 1        # increment the column number/ go to next cell
         
    elif (key == 27):
        break

#Release Video capture and close windows
capture.release()
cv2.destroyAllWindows()
