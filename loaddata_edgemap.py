# Letter Classifier 1: Using edgemaps as inputs
import numpy as np 
#import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle

# Set working directory
DATADIR = r"C:\Users\alexk\Desktop\Y4\Final Year Project\OpenCV\Edge Maps"
# Set classes
CATEGORIES = ["A","B","C"]

# Initialize global variables
img_size = 10
training_data = []

# Iterate over all images in the database and convert to grayscale, resize and add to training data list along with its class value
# (0 for non pressed, 1 for pressed)
def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
                                     
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (img_size,img_size))
            training_data.append([new_array,class_num])
    
    for category in CATEGORIES:
        datadir = r"C:\Users\alexk\Desktop\Y4\Final Year Project\OpenCV\Extra\{}\edgemaps".format(category)
        class_num = CATEGORIES.index(category)
        
        for img in os.listdir(datadir):
            img_array = cv2.imread(os.path.join(datadir,img), cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (img_size,img_size))
            training_data.append([new_array,class_num])
        
create_training_data()

#Shuffle the data in the list in order to train properly
random.shuffle(training_data)

# Separate data and labels to different lists
data = []
labels = []

for features, label in training_data:
    data.append(features)
    labels.append(label)

# Convert to numpy array for training
data = np.array(data).reshape(-1, img_size, img_size,1)
labels = np.array(labels)

# Create pickle files (serialization)
pickle_out = open("data2.pickle","wb")
pickle.dump(data,pickle_out)
pickle_out.close()

pickle_out = open("labels2.pickle","wb")
pickle.dump(labels,pickle_out)
pickle_out.close()