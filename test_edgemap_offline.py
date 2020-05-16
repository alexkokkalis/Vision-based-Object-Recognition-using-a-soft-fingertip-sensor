import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np 

# Select camera input (0 for PC internal cam, 1 for external USB cam)
capture = cv2.VideoCapture(1)

# Global Variables
img_size = 128
map_size = 10
grid = np.zeros((map_size,map_size))
rows = 0
cols = 0 
reshaped1 = np.zeros((1,img_size,img_size,1))
reshaped2 = np.zeros((1,map_size,map_size,1), dtype=np.uint8) # Select this for FFNN classifier
#reshaped2 = np.zeros((1,map_size,map_size,1), 'f') # Select this for CNN classifier
prediction1 = 0
prediction2 = 0

# Categories
CATEGORIES1 = ["Nonpressed", "Pressed"]
CATEGORIES2 = ["A","B","C"]

# Neural Network Models
model1 = tf.keras.models.load_model("ContactClassifier-CNN.model")
model2 = tf.keras.models.load_model("EdgemapClassifier-FFNN.model") # Select this for FFNN classifier
#model2 = tf.keras.models.load_model("EdgemapClassifier-CNN.model") # Select this for CNN classifier

# Infinite while loop to continuously capture video input
while(True):
    
    ret, frame = capture.read()     # Capture frames from the camera at each loop iteration
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert to grayscale
    cv2.imshow('frame', gray)   # Display the frame
    
    # Wait for a key to be pressed
    key = cv2.waitKey(1) & 0xFF
    
    # Use "c" key to scan the grid, classify letter when done
    if (key == ord('c')):
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_resize = cv2.resize(gray, (img_size,img_size))
        reshaped1[0,:,:,0] = np.array(gray_resize,'f')
        prediction1 = model1.predict(reshaped1)
        print(CATEGORIES1[int(prediction1)],rows,cols)
        grid[rows][cols] = int(prediction1) * 255
        plt.imshow(grid, cmap='gray')
        
        if (rows == map_size-1 & cols == map_size-1):
            reshaped2[0,:,:,0] = np.array(grid, dtype=np.uint8) 
            prediction2 = model2.predict(reshaped2)
            print("The letter predicition is: ", CATEGORIES2[np.argmax(prediction2)])
            break
        elif (cols == map_size-1):
            cols = 0 
            rows = rows + 1
        else:
            cols = cols + 1
        
    # Break the loop when the escape key is pressed
    elif (key == 27):
        break

#Release Video capture and close windows
capture.release()
cv2.destroyAllWindows()