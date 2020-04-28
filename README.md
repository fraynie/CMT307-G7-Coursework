# Notes

## load_image_data.py

The first file **load_image_data.py** imports the data into a #4D array containing a 100x100 pixel image with depth = 3 visible wave lenghts (RGB) value for each image in the training set, together with the labels: 'amusement', 'anger', 'awe', etc.

The size of the image files can be changed via the variable IMG_SIZE - e.g.:

    IMG_SIZE = 150

The data is saved into pickle files: **X_data.pickle** and **y_data.pickle** which are also uploaded. Pickle files are a simple way of serialising the numpy arrays of training data into a file. There's no need to run the code in this file - just need to download these pickle file and create the data from these files as explained below - which takes no time at all.
*NOTE: pickle files too large to upload to GitHib - download from https://drive.google.com/drive/folders/1QLFdWRIwaXv_PX99BT9_g7A3WCprs03y?usp=sharing*

The following files are available:
* **X_data.pickle** and **y_data.pickle** are the files containing the original data set and labels
* **X_data1.pickle** and **y_data1.pickle** contain the additional images files created via the image augmentation script **augmentation.py**. There are 5130 images for each category (41,040 images in total).
* **X_data_150x150.pickle** and **y_data_150x150.pickle** contain the additional images files, but with the image size set to 150x150 pixels

## augmentation.py
This file generates additional image files by making one or more of the following changes:
* introducing a random degree of rotation between 25% on the left and 25% on the right
* adding random noise to the image
* flipping the image howizontally
* adding a random degree of blurr to the image

The is achieving via the functions below:

    def random_rotation(image_array: ndarray):
        # pick a random degree of rotation between 25% on the left and 25% on the right
        random_degree = random.uniform(-25, 25)
        return sk.transform.rotate(image_array, random_degree)

    def random_noise(image_array: ndarray):
        # add random noise to the image
        return sk.util.random_noise(image_array)

    def horizontal_flip(image_array: ndarray):
        # horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !
        return image_array[:, ::-1]

    def blur_image(image_array):
        return cv2.GaussianBlur(image_array, (9,9),0)

The number of additional images craeted in each category is set via the TARGET_IMAGE_NUMBER variable, which was set to equal the number of images within the category with the highest number of images. The table below summarises the number of additional images created in each category:

|Category|Original number of images|Addtional images|Total images|
|:---|:---:|:---:|:---:|
|amusement|4724|406|5130|
|anger|1176|3954|5130|
|awe|2881|2249|5130|
|contentment|5130|0|5130|
|disgust|1581|3549|5130|
|excitement|2725|2405|5130|
|fear|969|4161|5130|
|sadness|2633|2497|5130|

## check_duplicates.py
The original data set included a small number of duplicate images. Also it is possible that the same random changes have been applied to the same image during the image augentation process resulting in further duplicate images in our training data which would have affected the model's learning.

This script checks for and removes duplicate images in each of our eight categories.

## CNN Models
The following Python scripts contain the models described in the report:
* __cnn_v1.py__ - _this is the initial attempt_
* __cnn_v2.py__ - _the enhanced CNN_
* __cnn_v3_xception_final.py__ - _this is the final model described in the report base on the Xception pre-trained model_
* __cnn_v3_xception_final.ipynb__ - _this is the final model as a Jupyter notebook_
