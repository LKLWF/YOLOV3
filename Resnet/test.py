# import os,sys
# import numpy as np
# import scipy
# from scipy import ndimage
# from tensorflow import keras
# import tensorflow as tf
# import cv2
# import matplotlib.pyplot as plt
# from tensorflow.keras.applications.resnet50 import ResNet50
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
# from PIL import Image
# import random

# model = tf.keras.models.load_model('Resnet/my_resnet_model.h5', compile=False)
# print('model saved============')

# # # test


# img_path = "F:/AI/MaskRecognition/yolo3/dataSet/dataset with label/test/have_mask/0185.jpg"

# img_path = "F:/AI/MaskRecognition/yolo3/dataSet/dataset with label/test/no_mask/0185.jpg"

# img = image.load_img(img_path, target_size=(224, 224))

# plt.imshow(img)
# img = image.img_to_array(img)/ 255.0
# img = np.expand_dims(img, axis=0)  # 为batch添加第四维
# print('hahhahahha======')
# print(model.predict(img))
# np.argmax(model.predict(img))


# 处理标注框
"""
在很多图像识别的数据集中，图像中需要关注的物体通常会呗标注框圈出来，tsnroflow提供了一些工具来处理标注框。下面这段代
码展示了如何通过 tf.image.draw_bounding_boxes函数在图像中加入标注框
"""
# matplotlib.pyplot 是一个python的画图工具。可以可视化tensorflow对图像的处理过程
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

# import matplotlib
# matplotlib.use('TkAgg')
# print('===============')
# print(matplotlib.get_backend())

# 读取图像的原始数据
image_raw_data = tf.io.gfile.GFile('./0007.jpg', 'rb').read()

with tf.Session() as sess:
    # 将图像使用JPEG的格式解码从而得到图像对应的三维矩阵。Tensorflow还提供了 tf.image.decode_png函数对png格式的图像进行编码。
    # 解码之后的结果为一个张量， 在使用他的取值之前需要明确调用运行的过程。
    image_data = tf.image.decode_jpeg(image_raw_data)
    # Decode a JPEG-encoded image to a uint8 tensor 所以这里的 image_data 已经是一个tsnsor

    # tf.image.draw_bounding_boxes函数要求图像矩阵中的数字为实数，所以需要先将图像矩阵转化为实数类型。
    # tf.image.draw_bounding_boxes函数图像的输入是一个batch的数据，也就是多张图像组成的四维矩阵，所以需要将解码之后的图像
    # 矩阵加一维。
    batched = tf.expand_dims(
        tf.image.convert_image_dtype(image_data, tf.float32), 0
    )
    # 给出每一张图像的所有标注框。一个标注框有四个数字，分别代表[ Ymin,Xmin,Ymax,Xmax]
    # have_mask 0.83 518 166 592 235
    # have_mask 0.90 771 194 860 320
    # have_mask 0.92 157 126 236 215
    # have_mask 0.94 316 96 390 211
    # have_mask 0.99 593 149 670 228
    # have_mask 0.99 423 150 480 222
    # have_mask 0.99 44 148 110 227
    boxes = tf.constant([[[0.125, 0.125, 0.75, 0.75], [0.375, 0.375, 0.75, 0.75]]])
    result = tf.image.draw_bounding_boxes(batched, boxes)
    result = tf.reduce_sum(result, 0)  # 这里显示的时候需要进行降维处理
    plt.imshow(result.eval())
    plt.show()

