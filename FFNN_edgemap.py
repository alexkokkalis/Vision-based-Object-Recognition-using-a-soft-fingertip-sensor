import tensorflow as tf
import pickle
from tensorflow.keras.callbacks import TensorBoard
import datetime
import os

localtime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#logdir = os.path.join("EdgemapClassifier-FFNN-{}".format(localtime))
#tensorboard = TensorBoard(log_dir="edgemap_comparison/{}/".format(logdir), histogram_freq=1, profile_batch=100000000)

data = pickle.load(open("data2.pickle", "rb"))
labels = pickle.load(open("labels2.pickle", "rb"))
#test_data = pickle.load(open("test_data.pickle", "rb"))
#test_labels = pickle.load(open("test_labels.pickle", "rb"))

# Optimisation: Check performance with all parameters in the following lists
layers = [1, 2, 3]
nodes = [64, 128, 256]

for layer in layers:
    for node in nodes:
            
        NAME = os.path.join("{}-layers-{}-nodes-{}".format(layer, node, localtime))
        tensorboard = TensorBoard(log_dir="Optimisation/EdgemapClassifier-FFNN/{}/".format(NAME), histogram_freq=1, profile_batch=100000000)
        
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Flatten())    #input layer
        
        for l in range(layer):
            model.add(tf.keras.layers.Dense(node,activation=tf.nn.relu)) 
        
        model.add(tf.keras.layers.Dense(3,activation=tf.nn.softmax)) # Output Layer
        
        model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
        model.fit(data, labels, epochs=30, callbacks=[tensorboard], validation_split=0.333)
            
model.summary()
#model.save("EdgemapClassifier-FFNN.model")