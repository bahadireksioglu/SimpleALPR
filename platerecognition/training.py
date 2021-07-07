import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import xml.etree.ElementTree as xmlet
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2, InceptionV3, InceptionResNetV2
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, Convolution2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf

datafile = pd.read_csv('labels.csv') #data file
datafile.head() #data columns

filename = datafile['filepath'][0]

def get_filename(filename):
    filename_image = xmlet.parse(filename).getroot().find('filename').text
    filepath_image = os.path.join('./images', filename_image)
    return filepath_image

image_path = list(datafile['filepath'].apply(get_filename))

labels = datafile.iloc[:,1:].values
#print(labels)

data = []
output = []

for i in range(len(image_path)):
    image = image_path[i]
    image_arr = cv2.imread(image)

    h, w, d = image_arr.shape

    #preprocessing
    load_image = load_img(image, target_size=(224, 224))
    load_image_arr = img_to_array(load_image)
    norm_load_image_arr = load_image_arr / 255.0 #normalization
    
    #normalization to labels 
    x_min, x_max, y_min, y_max = labels[i]

    #normalized x and normalized y min max
    nx_min, nx_max = x_min / w, x_max / w
    ny_min, ny_max = y_min / h, y_max / h

    label_norm = (nx_min, nx_max, ny_min, ny_max) # normalized output

    data.append(norm_load_image_arr)
    output.append(label_norm)

X = np.array(data, dtype=np.float32)
y = np.array(output, dtype=np.float32)


x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)
#print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

inception_resnet = InceptionResNetV2(weights="imagenet", include_top=False, 
                                    input_tensor=Input(shape=(224,224,3)))
inception_resnet.trainable=True

# model
head_model = inception_resnet.output
head_model = Flatten()(head_model)
head_model = Dense(500, activation="relu")(head_model)
head_model = Dense(250, activation="relu")(head_model)
head_model = Dense(4, activation='sigmoid')(head_model)

model = Model(inputs=inception_resnet.input, outputs=head_model)

#compile model
model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))
model.summary()

tensorboard = TensorBoard('object_detection')

history = model.fit(x=x_train, y=y_train, 
    batch_size=10, epochs=75, 
    validation_data=(x_test, y_test), 
    callbacks=[tensorboard])

history = model.fit(x=x_train, y=y_train, 
    batch_size=10, epochs=150, 
    validation_data=(x_test, y_test), 
    callbacks=[tensorboard], initial_epoch=76)

model.save('./models/object_detection.h5')

