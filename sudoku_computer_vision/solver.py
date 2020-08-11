import matplotlib.pyplot as plt
from number_extract import *
import tensorflow as tf
import cv2
import numpy as np
# import keras
# from keras.layers import Dense, Dropout, Flatten
# from keras.layers import Conv2D, MaxPooling2D
# from keras import backend as K
# from keras.datasets import mnist
# from keras.models import Sequential, load_model
# from keras.layers.core import Dense, Dropout, Activation
# from keras.utils import np_utils



# evaluate loaded model on test data
def identify_number(image):
    image_resize = cv2.resize(image, (28,28))   
    image_resize_2 = image_resize.reshape(28,28,1) 
    loaded_model_pred = loaded_model.predict_classes(image_resize_2 , verbose = 0)
    return loaded_model_pred[0]

def model():
    model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(28, 28, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2), 
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'), 
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(), 
    tf.keras.layers.Dense(512, activation='relu'), 
    tf.keras.layers.Dense(9, activation='sigmoid')  
])
    model.load_weights(r"C:\Users\Lenovo\Workspace\Machine Learning\Computer vision\Sudoku\models\model_1.h5")  
    print(model.summary())
    return model

def preprocess(image, skip_dilate= False):
    # Converting to a grey scale iage
    image_grey = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    # Using gaussian blur to reduce thw noise from the image
    image_blur = cv2.GaussianBlur(image_grey, (9,9), 0)
    image_thresh = cv2.adaptiveThreshold(image_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C , cv2.THRESH_BINARY, 11, 2)
    image_bitwise = cv2.bitwise_not(image_thresh, image_thresh)

    if not skip_dilate:
		# Dilate the image to increase the size of the grid lines.
	    kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], np.uint8)
	    image_bitwise = cv2.dilate(image_bitwise, kernel)

    return image_bitwise

