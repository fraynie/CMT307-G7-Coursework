#TODO: Description: see https://github.com/randerson112358/Python/blob/master/Classify_Images/cnn.ipynb

import pickle
import numpy as np

#TODO: I noticed that there were duplicate images in the data set - these need to be removed and pickle files need to be re-created
with open('X_data.pickle', 'rb') as f1:     
    X = pickle.load(f1)

with open('y_data.pickle', 'rb') as f2:
    y = pickle.load(f2)

print(type(X))
print(type(y))

print('X shape:', X.shape) #4D array 21,829 rows 100x100 pixel image with depth = 3 visible wave lenghts (RGB)
print('y shape:', y.shape) #2D array 21,829 rows and 1 column

#**************************************************************************
# NOTE: 
# This is the point I was aiming to reach - to be easily able to split up 
# the data into test and training data sets in order to start building the
# CNN.
#
#**************************************************************************

#Spilt the data into training and test sets using the sklearn.model_selection utility
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)


print(type(X_train)) 
print(type(X_test))
print(type(y_train))
print(type(y_test))


#Get the shape of x_train
print('X_train shape:', X_train.shape) #4D array 14,625 rows 100x100 pixel image with depth = 3 visible wave lenghts (RGB)
#Get the shape of y_train
print('_train shape:', y_train.shape) #2D array 14,625 rows and 1 column
#Get the shape of x_train
print('X_test shape:', X_test.shape) #4D array 7,204 rows 100x100 pixel image with depth = 3 visible wave lenghts (RGB)
#Get the shape of y_train
print('y_test shape:', y_test.shape) #2D array 7,204 rows and 1 column

#Show the first image and label from the training set
import matplotlib.pyplot as plt
img = plt.imshow(X_train[0])
plt.show()
print('The label is:', y_train[0])

#TODO: The data set is imbalanced - i.e. we do not have equal numbers of images in each category - this is a problem to fix...
#TODO: See https://datascience.stackexchange.com/questions/13490/how-to-set-class-weights-for-imbalanced-classes-in-keras

#One-Hot Encoding 
#Convert the labels into a set of 8 numbers to input into the neural network
from keras.utils import to_categorical
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

#Print an example of the new labels, e.g. the label 6 = [0,0,0,0,0,0,1,0]
print('The one hot label is:', y_train_one_hot[0])

#Normalize the pixels in the images to be a value between 0 and 1 , they are normally values between 0 and 255
#doing this will help the neural network.
X_train = X_train / 255
X_test = X_test / 255

#Build The CNN
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

model = Sequential() #Create the architecture

#Convolution layer to extract features from the input image, and create 32 ReLu
#5x5 convolved features/layers aka feature map.
#Note:You must input the input shape only in this first layer.
# number of output channels or convolution filters = 64
# number of rows in the convolution kernel
# number of cols in the convolution kernel
# input shape 100x100 RGB image, so spacially it's 3-Dimensional
# activation function Rectifier Linear Unit aka (ReLu)
model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(100,100,3))) 


#Pooling layer with a 2x2 filter to get the max element from the convolved features , 
#this reduces the dimensionality by half e.g. 50x50, aka sub sampling
#Note: the default for stride is the pool_size
model.add(MaxPooling2D(pool_size=(2, 2)))


#2nd Convolution layer with 64 channels
model.add(Conv2D(64, (5, 5), activation='relu'))

#Adding second Max Pooling layer
model.add(MaxPooling2D(pool_size=(2, 2)))

#Flattening, Flattens the input. Does not affect the batch size. 
#(Flattening occurs when you reduce all layers to one background layer), 
#this makes the image a linear array or 1D Array or 1D Vector to 
#feed into or connect with the neural network
model.add(Flatten())
model.add(Dense(1000, activation='relu')) # a layer with 1000 neurons and activation function ReLu
model.add(Dense(8, activation='softmax')) #a layer with 8 output neurons for each label using softmax activation function

model.compile(loss='categorical_crossentropy', # loss function used for classes that are greater than 2)
              optimizer='adam',
              metrics=['accuracy'])

#Batch: Total number of training examples present in a single batch
#Epoch:The number of iterations when an ENTIRE dataset is passed forward and 
#      backward through the neural network only ONCE.
#Fit: Another word for train

#NOTE: We don't need to use validation_data, so we didn't have to split the data 
#into a validation sets. We just put in 0.2 and this splits the data 20% for us.
hist = model.fit(X_train, y_train_one_hot, 
           batch_size=256, epochs=10, validation_split=0.2 )

#Get the models accuracy
model.evaluate(X_test, y_test_one_hot)[1]
#test_loss, test_acc = model.evaluate(test_images, test_labels)

#Visualize the model's accuracy
plt.plot(hist.history['accuracy']) #it broke here...  TODO: save the model...
plt.plot(hist.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

#Visualize the models loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()