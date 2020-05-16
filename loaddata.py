import numpy as np 
import os
import cv2
import random
import pickle

# Set working directory & classes
DATADIR = r"C:\Users\alexk\Desktop\Y4\Final Year Project\OpenCV"
CATEGORIES = ["Nonpressed","Pressed"]

# Initialize global variables
img_size = 128
training_data = []

def create_training_data():
    for category in CATEGORIES:                 # Iterate over all categories
        path = os.path.join(DATADIR, category)  # Connect directory path to path variable
        class_num = CATEGORIES.index(category)  # Convert class from string to number
                                     
        for img in os.listdir(path):            # Iterate over all images in folder
            # Read each image in gray scale and store in img_array variable:
            img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
            # Resize the image according to img_size value
            new_array = cv2.resize(img_array, (img_size,img_size))
            # Store the image data and respective class label into a list
            training_data.append([new_array,class_num])

create_training_data()
# Shuffle the data in the list in order to train properly
random.shuffle(training_data)


data = []
labels = []
# Separate data and labels to different lists
for features, label in training_data:
    data.append(features)
    labels.append(label)
# Convert to numpy array for training
data = np.array(data).reshape(-1, img_size, img_size, 1)
labels = np.array(labels)
# Create pickle files (serialization)
pickle_out = open("data.pickle","wb")
pickle.dump(data,pickle_out)
pickle_out.close()

pickle_out = open("labels.pickle","wb")
pickle.dump(labels,pickle_out)
pickle_out.close()


