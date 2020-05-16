import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, LSTM, Dropout, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle 
import datetime
import os

localtime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

#logdir = os.path.join("EdgemapClassifier-LSTM-{}".format(localtime))
#tensorboard = TensorBoard(log_dir="edgemap_comparison/{}/".format(logdir), histogram_freq=1, profile_batch=100000000)

# Load pickle files (training data)
data = pickle.load(open("data4.pickle", "rb"))
labels = pickle.load(open("labels4.pickle", "rb"))


#Normalize
data = data/255.0

# Optimisation: Check performance with all parameters in the following lists

LSTM_sizes = [64, 128, 256]
dense_sizes = [16, 32, 64]
dense_layers = [0, 1, 2]

for LSTM_size in LSTM_sizes:
    for dense_layer in dense_layers:
        for dense_size in dense_sizes:
            
            NAME = os.path.join("{}-LSTMsize-{}-Layers-{}-nodes-{}".format(LSTM_size, dense_layer, dense_size, localtime))
            tensorboard = TensorBoard(log_dir="Optimisation/{}/".format(NAME), histogram_freq=1, profile_batch=100000000)

            # LSTM framework
            model = Sequential()
            
            model.add(LSTM(LSTM_size, input_shape = (data.shape[1:]), activation='relu', return_sequences=True))
            model.add(Dropout(0.2))
            
            model.add(LSTM(LSTM_size, activation='relu'))
            model.add(Dropout(0.2))
            
            for l in range(dense_layer):
                model.add(Dense(dense_size, activation='relu'))
                model.add(Dropout(0.2))
            
            model.add(Dense(3, activation='softmax'))
            
            opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5)
            
            model.compile(loss='sparse_categorical_crossentropy',
                          optimizer=opt,
                          metrics=['accuracy'])
            
            model.fit(data, labels, epochs=30, callbacks=[tensorboard], validation_split=0.333)

#Save the trained model
#model.save("EdgemapClassifier-LSTM.model")
