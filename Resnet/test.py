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
    boxes = tf.constant([[[0.125, 0.125, 0.75, 0.75], [0.375, 0.375, 0.75, 0.75]]])
    result = tf.image.draw_bounding_boxes(batched, boxes)
    result = tf.reduce_sum(result, 0)  # 这里显示的时候需要进行降维处理
    plt.imshow(result.eval())
    plt.show()

# 用opencv画标注框
def draw_box() :
    import cv2
    import numpy as np
    image = "F:/AI/MaskRecognition/yolo3/dataSet/dataset with label/test/no_mask/0185.jpg"
    img = cv2.imread(image)
    x0, y0, x1, y1 = [300, 300, 500, 500]
    # rectangle参数说明：图片，(边框左上角坐标)，(边框右下角坐标)，边框颜色，边框厚度
    cv2.rectangle(img,(int(x0),int(y0)),(int(x1),int(y1)),(0,255,0),3)
    # putText参数说明：图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细
    result = np.asarray(img)
    cv2.putText(result, 'mask', (500, 500), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.50, color=(255, 0, 0), thickness=2)
    cv2.imshow('result', result)
    cv2.waitKey(0) # 解决图片一闪而过
    # 保存图片
    cv2.imwrite('1.png',result,[int(cv2.IMWRITE_PNG_COMPRESSION),9])
