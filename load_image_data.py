# Description: extact training images into arrays suitable for ml 
# see https://www.youtube.com/watch?v=j-3vuBynnOE

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random



print('Start')

DATA_DIR = 'C:\\Users\\ma000310\\source\\repos\\CMT307\\SEM2\\data\\Flickr'
CATEGORIES = ['amusement', 'anger', 'awe', 'contentment', 'disgust', 'excitement', 'fear', 'sadness']
IMG_SIZE = 100

def create_training_data():
    return_list = []
    for category in CATEGORIES:
        print('processing category: ', category)
        path = os.path.join(DATA_DIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.cvtColor(cv2.imread(os.path.join(path, img)), cv2.COLOR_BGR2RGB)
                img_resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                return_list.append([img_resized_array, class_num])
            except Exception as e:
                print('Corrupt file: ', category + '\\' + img)
                pass
    random.shuffle(return_list)
    return(return_list)

training_data = create_training_data()
print(len(training_data))

for sample in training_data[:10]:
    print(sample[1])

X = []
y = []

for features, labels in training_data:
    X.append(features)
    y.append(labels)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
y = np.array(y).reshape(-1, 1)

import pickle
with open('X_data.pickle', 'wb') as f1:
    pickle.dump(X, f1)

with open('y_data.pickle', 'wb') as f2:
    pickle.dump(y, f2)

del X
del y

