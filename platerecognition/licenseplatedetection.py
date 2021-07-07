import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import xml.etree.ElementTree as xmlet
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# main load start
# model load
model = tf.keras.models.load_model('./models/object_detection.h5')
# main load end

def detect_license_plate(image):
    image = load_img(image) # PIL Object
    image = np.array(image, dtype=np.uint8) # 8 bit array (0 to 255)
    h, w, d = image.shape

    image_for_arr = load_img(image, target_size=(224, 224))
    image_arr_224 = img_to_array(image_for_arr) / 255.0 # convert into array and get the normalized output

    test_arr = image_arr_224.reshape(1, 224, 224, 3)
    test_arr.shape
    
    # make predictions
    coords = model.predict(test_arr)

    #denormalized
    denorm = np.array([w, w, h, h])
    coords = coords * denorm
    coords = coords.astype(np.int32)

    # crop image and return
    x_min, x_max, y_min, y_max = coords[0]
    image = image[y_min:y_max, x_min:x_max]

    return image
