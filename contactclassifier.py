# Import necessary libraries
import numpy as np
import cv2
import tensorflow as tf

# Select camera input (0 for PC internal cam, 1 for external USB cam)
capture = cv2.VideoCapture(1)

#Global Variables
img_size = 128

CATEGORIES = ["Nonpressed", "Pressed"]
model = tf.keras.models.load_model("ContactClassifier-CNN.model")
reshaped = np.zeros((1,img_size,img_size,1))

# Infinite while loop to continuously capture video input
while(True):
    
    ret, frame = capture.read()     # Capture frames from the camera at each loop iteration
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert to grayscale
    cv2.imshow('frame', gray)   # Display the frame
    
    # Break the loop when the escape key is pressed
    key = cv2.waitKey(1) & 0xFF
    if (key == 27):
        break
    
    elif (key == ord('c')               # if key "c" is pressed execute the following code
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert to grayscale
        gray_resize = cv2.resize(gray, (img_size,img_size)) # Resize image
        reshaped[0,:,:,0] = np.array(gray_resize,'f') # transfer the data into a 4-dimensional array
        prediction = model.predict(reshaped)    # use the saved model to determine if contact occured
        print(CATEGORIES[int(prediction)]) # print the outcome
        
#Release Video capture and close windows
capture.release()
cv2.destroyAllWindows()
