# Vision-based-Object-Recognition-using-a-soft-fingertip-sensor
This repository contains all software modules created for my Final Year Project at the University of Bath

A short description for all modules on this folder is given below:

contact_data.py:
Receives live USB camera image. Saves current frame when either keys “n” or “p” are pressed. 
Saves frame in the “Non-pressed” or “Pressed” folder directory based on which key was pressed.	
A database of pressed and non-pressed images is created, with 100 frames in each one.

loaddata.py:
Obtains the images from the database (Pressed and Non-pressed folders) and transfers the data in a list, along with their respective labels.
The list is then shuffled, serialised, and saved on pickle files.
The pickle files created are ready to be used for training of neural networks.

CNN.py:
A CNN with two convolutional layers and a dense layer is trained to perform binary classification on the pressed and non-pressed classes.
5 epochs are used to reach a training accuracy of 99%. 	
A model able to distinguish between a pressed and non-pressed tactile sensor tip is saved for later use (ContactClassifier-CNN)

contactclassifier.py:	
A module that obtains the live camera input and tests whether the sensor is pressed or not. 
When the key “c” is pressed, the current frame is passed on the ContactClassifier-CNN model and a printed prediction is output to inform of the outcome.	
Validation that the ContactClassifier module performs correctly.(Offline Testing)

DataCollection.py:
A data collection routine which scans through all cells of the grid using the tactile sensor. 
The key “c” is pressed at each grid point to obtain the current frame and determine if it is pressed or not using the ContactClassifier-CNN model. 
At each cell, the virtual grid updates its value based on the existence of a part of a letter in the experimental grid. 
Edge maps are stored in the database when scanning is finished.
The edge map is also plotted to validate that the scanning had no errors. 	
A data collection routine that produces data for 2 databases (Frames and edge maps) with a single scan oven a letter and grid.
30 scans were performed for each letter, for a total of 90 scans.

loaddata_edgemap.py:
Obtains edge map images from the database and transfers data into a list along with their labels. 
The list is then shuffled, serialised, and saved on pickle files.	
The pickle files created are ready to be used for training of neural networks.

CNN_edgemap.py:
A CNN with two convolutional layers and a dense layer is trained to perform letter classification by observing an unknown edge map. 
5 epochs are used to reach a training accuracy of 100%. 	
A model able to recognise the letter depicted in an unknown edge map that is fed into it. (EdgemapClassifier-CNN)

FFNN_edgemap.py:
A FFNN with 3 dense layers is trained to perform letter classification by observing an unknown edge map.
5 epochs are used to reach a training accuracy of 100%.
A model able to recognise the letter depicted in an unknown edge map that is fed into it. (EdgemapClassifier-FFNN)

LSTM_edgemap.py:
An LSTM with 2 LSTM layers and 2 dense layers is trained to perform letter classification by observing an unknown edge map. 
30 epochs are used to reach a training accuracy of about 100%.
A model able to recognise the letter depicted in an unknown edge map that is fed into it. (EdgemapClassifier-LSTM)

test_edgemap_offline.py:
A scanning module that creates a newly formed edge map and feeds it into the edge map classifiers to determine the letter scanned. 
Validation that the edge map classifiers perform correctly. (Offline Testing)

DecisionMatrix.py:
A module which essentially splits the data into training and validation sets, trains and tests them and compares the predictions with their labels to produce confusion matrices. 
Obtain confusion matrices for the edge map classifier modules, as part of performance comparison.
