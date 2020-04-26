import os,sys
import numpy as np
from tensorflow import keras
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageDraw, ImageFont
import random

model = tf.keras.models.load_model('my_resnet_model.h5', compile=False)

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
img = image.load_img(img_path, target_size=(224, 224))
plt.imshow(img)
img = image.img_to_array(img)/ 255.0
img = np.expand_dims(img, axis=0)  # 为batch添加第四维
print('Actual forecast, output target value')
pre_y = model.predict(img)
print(pre_y)
print('================')
# Y = put_prey(pre_y)
# print(Y + '===================')


# 加载图像
# img = image.load_img(img_path, target_size=(224, 224))

# 图像预处理
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)

# 对图像进行分类
# preds = model.predict(x)

# 输出预测概率
# print('==============')
# print(preds)
# print(np.argmax(model.predict(img)))

# import cv2
# import numpy as np

# image = "F:/AI/MaskRecognition/yolo3/dataSet/dataset with label/test/no_mask/0185.jpg"

# img = cv2.imread(image)
# x0, y0, x1, y1 = [300, 300, 500, 500]
# # rectangle参数说明：图片，(边框左上角坐标)，(边框右下角坐标)，边框颜色，边框厚度
# cv2.rectangle(img,(int(x0),int(y0)),(int(x1),int(y1)),(0,255,0),3)
# # putText参数说明：图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细
# result = np.asarray(img)
# cv2.putText(result, 'mask', (500, 500), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
#                     fontScale=0.50, color=(255, 0, 0), thickness=2)
# cv2.imshow('result', result)
# cv2.waitKey(0) # 解决图片一闪而过
# # 保存图片
# cv2.imwrite('1.png',result,[int(cv2.IMWRITE_PNG_COMPRESSION),9])