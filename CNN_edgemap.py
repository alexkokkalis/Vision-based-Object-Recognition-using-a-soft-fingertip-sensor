from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
import pickle 
from tensorflow.keras.callbacks import TensorBoard
import datetime
import os

# Create variables to be used in the tensorboard callback
localtime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = os.path.join("EdgemapClassifier-CNN-{}".format(localtime))

# Create tensorboard callback
tensorboard = TensorBoard(log_dir="edgemap_comparison/{}/".format(logdir), histogram_freq=1, profile_batch=100000000)

# Load pickle files (training data)
data = pickle.load(open("data2.pickle", "rb"))
labels = pickle.load(open("labels2.pickle", "rb"))


#Normalize
data = data/255.0


# Optimisation: Check performance with all parameters in the following lists
convolutional_layers = [1, 2]
layer_sizes = [32, 64, 128]
dense_layers = [0, 1, 2]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for convolutional_layer in convolutional_layers:
            
            NAME = os.path.join("{}-conv-{}-nodes-{}-dense-{}".format(convolutional_layer, layer_size, dense_layer, localtime))
            tensorboard = TensorBoard(log_dir="Optimisation/EdgemapClassifier-CNN/{}/".format(NAME), histogram_freq=1, profile_batch=100000000)
            
            # CNN framework
            model = Sequential()
            model.add(Conv2D(layer_size, (3,3), input_shape = data.shape[1:] ))
            model.add(Activation("relu"))
            model.add(MaxPooling2D(pool_size=(2,2)))
            
            for l in range(convolutional_layer - 1):
                model.add(Conv2D(layer_size, (3,3)))
                model.add(Activation("relu"))
                model.add(MaxPooling2D(pool_size=(2,2)))
            
            model.add(Flatten())
            
            for l in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation("relu"))
            
            model.add(Dense(3))
            model.add(Activation("sigmoid"))
            
            model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
            model.fit(data, labels, batch_size=32, epochs=30, callbacks=[tensorboard], validation_split=0.333)

#Save the trained model
#model.save("EdgemapClassifier-CNN.model")

