# Import necessary libraries
import cv2
import os

# Select camera input (0 for PC internal cam, 1 for external USB cam)
capture = cv2.VideoCapture(0)

#Global Variables
counter1 = 1
counter2 = 1

path_non = r"C:\Users\alexk\Desktop\Y4\Final Year Project\OpenCV\Nonpressed"
path_pre = r"C:\Users\alexk\Desktop\Y4\Final Year Project\OpenCV\Pressed"

# Infinite while loop to continuously capture video input

while(True):
    
    ret, frame = capture.read()                             # Capture frames from the camera at each loop iteration
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)          # Convert to grayscale
    cv2.imshow('frame', gray)                               # Display the frame
    
    # Break the loop when the escape key is pressed
    key = cv2.waitKey(1) & 0xFF
    if (key == 27):
        break
    
    # Record current frame when key "n" is pressed (for non pressed tactip)
    elif (key == ord('n')):
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        file_name = "nonpressed{}.png".format(counter1)
        cv2.imwrite(os.path.join(path_non, file_name),gray)
        counter1 += 1
        
    # Record current frame when key "p" is pressed  (for pressed tactip)
    elif (key == ord('p')):
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        file_name = "pressed{}.png".format(counter2)
        cv2.imwrite(os.path.join(path_pre, file_name),gray)
        counter2 += 1

#Release Video capture and close windows
capture.release()
cv2.destroyAllWindows()
