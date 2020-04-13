# Notes

## load_image_data.py

The first file **load_image_data.py** imports the data into a #4D array containing a 100x100 pixel image with depth = 3 visible wave lenghts (RGB) value for each image in the training set, together with the labels: 'amusement', 'anger', 'awe', etc.

The size of the image files can be changed via the variable IMG_SIZE - e.g.:

    IMG_SIZE = 150

Tha data is saved into pickle files: **X_data.pickle** and **y_data.pickle** which are also uploaded. Pickle files are a simple way of serialising the numpy arrays of training data into a file. There's no need to run the code in this file - just need to download these pickle file and create the data from these files as explained below - which takes no time at all.
*NOTE: pickle files too large to upload to GitHib - download from https://drive.google.com/drive/folders/1QLFdWRIwaXv_PX99BT9_g7A3WCprs03y?usp=sharing*

The following files are available:
* **X_data.pickle** and **y_data.pickle** are the files containing the original data set and labels
* **X_data1.pickle** and **y_data1.pickle** contain the additional images files created via the image augmentation script. There are 5130 images for each category (41,040 images in total).
* **X_data_150x150.pickle** and **y_data_150x150.pickle** contain the additional images files, but with the image size set to 150x150 pixels



## first_go.py

The second file **first_go.py** unpacks the data from these pickle files using the following code *(Note: I've done this as a Python script because that's what I'm familiar with - I'll convert to a Jupyter notebook)*:

    with open('X_data.pickle', 'rb') as f1:
        X = pickle.load(f1)

    with open('y_data.pickle', 'rb') as f2:
        y = pickle.load(f2)`

From these X and y variables we can then easily split the training data into training and testing data sets using the sklearn test_train_split utility as we have done in class

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33) 

From this data we can build the CNN (starting on line 70). I've only done a basic CNN based on a simple example - but following this example we should be able to jointly work to create a better model based on what we've learnt in class.

The basic CNN I've created provides about 35% accuracy after 6 epochs - we should be able to improve this...

![Model accuracy](https://github.com/fraynie/CMT307-G7-Coursework/blob/master/images/Figure_1.png)
