# Notes

## load_image_data.py

The first file **load_image_data.py** imports the data into a #4D array containing a 100x100 pixel image with depth = 3 visible wave lenghts (RGB) value for each image in the training set, together with the labels: 'amusement', 'anger', 'awe', etc.

Tha data is saved into two pickle files: **X_data.pickle** and **y_data.pickle** which are also uploaded. Pickle files are a simple way of serialising the numpy arrays of training data into a file. There's no need to run the code in this file - just need to download these pickle file and create the data from these files as explained below - which takes no time at all.
*NOTE: pickle files too large to upload to GitHib - download from <https://1drv.ms/u/s!AniRgLUlIrYtgdo-NmV3Ja2f9Dg07w?e=kAp7C4>*

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

![Model accuracy](images\Figure_1.png)
