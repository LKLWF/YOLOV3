import os,sys
import numpy as np
import scipy
from scipy import ndimage
from tensorflow import keras
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.optimizers import SGD
from PIL import Image
import random


def DataSet():
    
    train_have_mask ='F:/AI/MaskRecognition/yolo3/dataSet/dataset with label/train/have_mask/'
    train_no_mask = 'F:/AI/MaskRecognition/yolo3/dataSet/dataset with label/train/no_mask/'
    
    test_have_mask ='F:/AI/MaskRecognition/yolo3/dataSet/dataset with label/test/have_mask/'
    test_no_mask = 'F:/AI/MaskRecognition/yolo3/dataSet/dataset with label/test/have_mask/'
    
    imglist_train_havemask = os.listdir(train_have_mask)
    imglist_train_nomask = os.listdir(train_no_mask)
    
    imglist_test_havemask = os.listdir(test_have_mask)
    imglist_test_nomask = os.listdir(test_no_mask)
        
    X_train = np.empty((len(imglist_train_havemask) + len(imglist_train_nomask), 224, 224, 3))
    Y_train = np.empty((len(imglist_train_havemask) + len(imglist_train_nomask), 2))
    count = 0
    for img_name in imglist_train_havemask:
        
        img_path = train_have_mask + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        
        X_train[count] = img
        Y_train[count] = np.array((1,0))
        count+=1
        
    for img_name in imglist_train_nomask:

        img_path = train_no_mask + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        
        X_train[count] = img
        Y_train[count] = np.array((0,1))
        count+=1
        
    X_test = np.empty((len(imglist_test_havemask) + len(imglist_test_nomask), 224, 224, 3))
    Y_test = np.empty((len(imglist_test_havemask) + len(imglist_test_nomask), 2))
    count = 0
    for img_name in imglist_test_havemask:

        img_path = test_have_mask + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        
        X_test[count] = img
        Y_test[count] = np.array((1,0))
        count+=1
        
    for img_name in imglist_test_nomask:
        
        img_path = test_no_mask + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        
        X_test[count] = img
        Y_test[count] = np.array((0,1))
        count+=1
        
    index = [i for i in range(len(X_train))]
    random.shuffle(index)
    X_train = X_train[index]
    Y_train = Y_train[index]
    
    index = [i for i in range(len(X_test))]
    random.shuffle(index)
    X_test = X_test[index]    
    Y_test = Y_test[index]

    return X_train,Y_train,X_test,Y_test


X_train,Y_train,X_test,Y_test = DataSet()
print('X_train shape : ',X_train.shape)
print('Y_train shape : ',Y_train.shape)
print('X_test shape : ',X_test.shape)
print('Y_test shape : ',Y_test.shape)


# # model


model = VGG16(
    weights='imagenet',
    classes=1000
)
# model1 = keras.applications.vgg16.VGG16(include_top=True, weights='imagenet',
#                                 input_tensor=None, input_shape=None,
#                                 pooling=None,
#                                 classes=1000)


# optimizer = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
# model.compile(optimizer = optimizer,
#     loss = keras.losses.categorical_crossentropy,
#     metrics=['accuracy'] )

sgd = SGD(lr=0.0001, decay=0.0, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd,
    loss = keras.losses.categorical_crossentropy,
    metrics=['accuracy'] )

# # train
# cp_callback= keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only= True, verbose=1 )

# model.fit(X_train, Y_train, epochs=1, validation_data=(X_train, Y_train), callbacks=[cp_callback] )

model.fit(X_train, Y_train, epochs=1, batch_size=6)

# # evaluate


model.evaluate(X_test, Y_test, batch_size=32)
print('Evaluate the training model and output the accuracy and loss')
print(model.evaluate(X_test, Y_test, batch_size=32))

# # save


model.save('./my_vgg16_model.h5')

# # restore


model = tf.keras.models.load_model('./my_vgg16_model.h5', compile=False)
print('model saved============')