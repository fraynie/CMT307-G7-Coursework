# -*- coding: utf-8 -*-
"""cnn_v3_xception_final.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/11Z_8dl6y2JDVH-8uojn3q1TEDunIvWsW
"""

# import TensorFlow, Keras
import tensorflow as tf
from tensorflow import keras

"""Load data - augmented images and labels have been serialized to pickles file and are available at https://drive.google.com/drive/folders/1QLFdWRIwaXv_PX99BT9_g7A3WCprs03y?usp=sharing

To run this code, copy X_data1.pickle and y_data1.pickle to your the root of your Google drive.

These files contain 5130  images in each category, of size 100x100
"""

# load data
import pickle
import numpy as np
import matplotlib.pyplot as plt
from google.colab import drive

drive.mount('/content/drive')
with open('/content/drive/My Drive/X_data1.pickle', 'rb') as f1:     
    X = pickle.load(f1)

with open('/content/drive/My Drive/y_data1.pickle', 'rb') as f2:
    y = pickle.load(f2)

"""Split the images and labels to Test, Train and Validation sets"""

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)
X_valid,X_test,y_valid,y_test = train_test_split(X_test,y_test,test_size=0.4)
np.random.seed(200)
np.random.shuffle(X_train)
np.random.seed(200)
np.random.shuffle(X_test)
np.random.seed(200)
np.random.shuffle(y_train)
np.random.seed(200)
np.random.shuffle(y_test)
np.random.seed(200)
np.random.shuffle(X_valid)
np.random.seed(200)
np.random.shuffle(y_valid)

#Get the shape of X_train and y_train
print('X_train shape:', X_train.shape) #4D array 30,780 rows 100x100 pixel image with depth = 3 visible wave lenghts (RGB)
print('y_train shape:', y_train.shape) #2D array 30,780 rows and 1 column

#Get the shape of X_valid and y_valid
print('X_valid shape:', X_valid.shape) #4D array 6,156 rows 100x100 pixel image with depth = 3 visible wave lenghts (RGB)
print('y_valid shape:', y_valid.shape) #2D array 6,156 rows and 1 column

#Get the shape of X_test and y_test
print('X_test shape:', X_test.shape) #4D array 4,104 rows 100x100 pixel image with depth = 3 visible wave lenghts (RGB)
print('y_test shape:', y_test.shape) #2D array 4,104 rows and 1 column

"""Show the first image and label from the training set"""

import matplotlib.pyplot as plt
img = plt.imshow(X_train[0])
plt.show()
print('The label is:', y_train[0])

"""Normalize the data"""

X_train = X_train / 255.0
X_test = X_test / 255.0
X_valid = X_valid / 255.0

"""Fine-tuning xception model"""

base_model = keras.applications.xception.Xception(weights="imagenet", include_top=False)

avg = keras.layers.GlobalAveragePooling2D()(base_model.output)
output = keras.layers.Dense(8, activation="softmax")(avg)
model = keras.Model(inputs=base_model.input, outputs=output)

for layer in base_model.layers:
  layer.trainable = False
optimizer = keras.optimizers.SGD(lr=0.2, momentum=0.9, decay=0.01)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

"""Fit the model"""

history = model.fit(X_train.reshape((X_train.shape[0], 100, 100, 3)), y_train, epochs=5, validation_data=(X_valid.reshape((X_valid.shape[0], 100, 100, 3)),y_valid))

# save the model
model.save("model_xception.hdf5")

"""Train the model further with the base layers unfrozen"""

# The model can be further trained with base layers unfrozen
for layer in base_model.layers:
  layer.trainable = True

optimizer = keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=0.01)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

history = model.fit(X_train.reshape((X_train.shape[0], 100, 100, 3)), y_train, epochs=40, validation_data=(X_valid.reshape((X_valid.shape[0], 100, 100, 3)),y_valid))

# save the model
model.save("model_xception.hdf5")

"""Evaluate the model"""

model.evaluate(X_test.reshape(X_test.shape[0], 100, 100, 3), y_test)

"""Draw chart"""

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
model2 = load_model("model_xception.hdf5")

plt.plot(history.history['accuracy']) 
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])

plt.title('Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')

plt.show()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')

plt.show()