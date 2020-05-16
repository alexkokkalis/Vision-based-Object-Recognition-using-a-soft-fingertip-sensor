import matplotlib.pyplot as plt
import tensorflow as tf
import pickle 
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools
import io
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, Dropout, LSTM

data = pickle.load(open("data4.pickle", "rb"))
labels = pickle.load(open("labels4.pickle", "rb"))
data = data/255.0

validation_split = 0.33
split_at = round(data.shape[0] - (data.shape[0] * validation_split))

train_data = data[:split_at]
test_data = data[split_at:]
train_labels = labels[:split_at]
test_labels = labels[split_at:]

# Use this region for the desired neural network
###############################################################################

model = tf.keras.models.Sequential()

model.add(LSTM(64, input_shape = (train_data.shape[1:]), activation='relu', return_sequences=True))
model.add(Dropout(0.2))
            
model.add(LSTM(64, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(3, activation='softmax'))

opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5)
###############################################################################
model.compile(optimizer=opt, loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(train_data, train_labels, epochs=30)

predictions = model.predict(test_data)
rounded_predictions = []

for i in predictions:
    rounded = np.argmax(i)
    rounded_predictions.append(rounded)

rounded_predictions = np.array(rounded_predictions)

# Converts the matplotlib plot specified by 'figure' to a PNG image and
# returns it. The supplied figure is closed and inaccessible after this call.
def plot_to_image(figure):

  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
  image = tf.expand_dims(image, 0)
  return image


# This function prints and plots the confusion matrix.
# Normalization can be applied by setting `normalize=True`.
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
   
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("LSTMcm.png")

cm = confusion_matrix(rounded_predictions, test_labels)
cm_plot_labels = ['A', 'B', 'C']
figure = plot_confusion_matrix(cm, cm_plot_labels, title='LSTM Confusion Matrix')





