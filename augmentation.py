# Augments images... from https://gist.github.com/tomahim/9ef72befd43f5c106e592425453cb6ae
#import cv2
import os
import random
from scipy import ndarray

# image processing library
import skimage as sk
from skimage import transform
from skimage import util
from skimage import io
import cv2

DATA_DIR = 'C:\\Users\\ma000310\\source\\repos\\CMT307\\SEM2\\data\\Flickr'
CATEGORIES = ['amusement', 'anger', 'awe', 'contentment', 'disgust', 'excitement', 'fear', 'sadness']
TARGET_IMAGE_NUMBER = 5130
RUN = 3


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

# dictionary of the transformations we defined earlier
available_transformations = {
    'rotate': random_rotation,
    'noise': random_noise,
    'horizontal_flip': horizontal_flip,
    'blur': blur_image
}

for category in CATEGORIES:
    print('processing category: ', category)
    path = os.path.join(DATA_DIR, category)

    # find all files paths from the folder
    images = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

    # determine the number of additional images to create based on our target
    num_files_desired = TARGET_IMAGE_NUMBER - len(images)
    print('files to create:', num_files_desired)

    num_generated_files = 0
    while num_generated_files < num_files_desired:
        # random image from the folder
        image_path = random.choice(images)
        # read image as an two dimensional array of pixels
        image_to_transform = sk.io.imread(image_path)
        # random num of transformation to apply
        num_transformations_to_apply = random.randint(1, len(available_transformations))

        num_transformations = 0
        transformed_image = None
        while num_transformations <= num_transformations_to_apply:
            # random transformation to apply for a single image
            key = random.choice(list(available_transformations))
            transformed_image = available_transformations[key](image_to_transform)
            num_transformations += 1

        new_file_path = '%s/augmented_image5_%s.jpg' % (path, num_generated_files)

        transformed_image = sk.img_as_ubyte(transformed_image, force_copy=False)

        # write image to the disk
        io.imsave(new_file_path, transformed_image)
        num_generated_files += 1