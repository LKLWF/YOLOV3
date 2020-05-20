import os,sys
import numpy as np
from tensorflow import keras
import tensorflow as tf
import cv2
import time  # 引入time模块
import matplotlib.pyplot as plt
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageDraw, ImageFont
import random

model = tf.keras.models.load_model('my_vgg16_model.h5', compile=False)
def put_prey(pre_y):
    # output = []
    Y = []
    for y in pre_y:
        if y[0] < 1 : 
            Y = 'have_mask'
        else:
            Y = 'no_mask'
    return Y

# # test


# img_path = "F:/AI/MaskRecognition/yolo3/dataSet/dataset with label/test/have_mask/0185.jpg"
#[[9.994616e-01 5.383726e-04]]
img_path = "F:/AI/MaskRecognition/yolo3/dataSet/dataset with label/test/no_mask/0185.jpg"
#[[0.9987527  0.00124726]]
img = image.load_img(img_path, target_size=(224,224))
plt.imshow(img)
x = image.img_to_array(img)/ 255.0
x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)
pre_y = model.predict(x)
print(pre_y)
# 输出预测概率
print('=============')
print(np.argmax(model.predict(x)))

# path = '../test'
# start = time.time()
# for filename in os.listdir(path):
#     img_path = path+'/'+filename
#     # 加载图像
#     img = image.load_img(img_path, target_size=(224, 224))
#     plt.imshow(img)
#     # 图像预处理
#     img = image.img_to_array(img)/ 255.0
#     img = np.expand_dims(img, axis=0)  # 为batch添加第四维
#     # 对图像进行分类
#     pre_y = model.predict(img)
#     print(pre_y)
#     # 输出预测概率
#     # np.argmax(model.predict(img))
    
# end = time.time()
# print('running time====================')
# print(end-start)