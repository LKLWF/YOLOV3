"""
Retrain the YOLO model for your own dataset.
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
# from keras.utils import multi_gpu_model
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from yolo3.utils import get_random_data


def _main():
    annotation_path = '2007_train.txt'
    log_dir = 'logs/'
    classes_path = 'model_data/voc_classes.txt'
    anchors_path = 'model_data/tiny_yolo_anchors.txt'
    # anchors_path = 'model_data/yolo_anchors.txt'
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)

    input_shape = (416,416) # multiple of 32, hw

    is_tiny_version = len(anchors)==6 # 6是使用yolov3-tiny.h5,9是使用yolo.h5
    # is_tiny_version = False
    if is_tiny_version:
        model = create_tiny_model(input_shape, anchors, num_classes,
            freeze_body=2, weights_path='model_data/yolov3-tiny.h5')
    else:
        model = create_model(input_shape, anchors, num_classes, freeze_body=2, weights_path='model_data/yolo.h5') # make sure you know what you freeze
    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(
        log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss',
        save_weights_only=True, #只保存模型权重，否则将保存整个模型
        save_best_only=True, #s只保存在验证集上性能最好的模型
        period=3
    )
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    val_split = 0.2
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val
    # num_train = int(len(lines)/2)
    #Train with frozen layers first, to get a stable loss.
    #Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    if True:
        model.compile(
            optimizer=Adam(lr=1e-3),
            loss={'yolo_loss': lambda y_true, y_pred: y_pred}, # use custom yolo_loss Lambda layer.
            metrics = ['accuracy', fmeasure, recall, precision]
        )

        batch_size = 2
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        hist = model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                steps_per_epoch=max(1, num_train//batch_size),
                validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
                validation_steps=max(1, num_val//batch_size),
                epochs=10,
                initial_epoch=0,
                # epochs=2,
                # initial_epoch=0,
                callbacks=[ logging, checkpoint, reduce_lr, early_stopping ]
                # callbacks=[ logging, checkpoint ]
        )
        model.save_weights(log_dir + 'trained_weights_stage_1.h5')
        training_vis(hist, epochs=10)

    #model.load_weights("logs/ep034-loss6.105-val_loss6.205.h5")
    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    if True:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        model.compile(
            optimizer=Adam(lr=1e-4),
            loss={'yolo_loss': lambda y_true,y_pred: y_pred},
            metrics = ['accuracy', fmeasure, recall, precision]
        ) # recompile to apply the change
        print('Unfreeze all of the layers.')

        batch_size = 2 # note that more GPU memory is required after unfreezing the body
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        hist = model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
            validation_steps=max(1, num_val//batch_size),
            # epochs=100,
            # initial_epoch=16,
            epochs=20,
            initial_epoch=10,
            callbacks=[ logging, checkpoint, reduce_lr, early_stopping ]
            # callbacks=[ logging, checkpoint]
        )
        # model.save_weights(log_dir + 'trained_weights_final.h5')
        training_vis(hist, epochs=20)
    # Further training if needed.


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)

def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/yolo_weights.h5'):
    '''create the training model'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
        num_anchors//3, num_classes+5)) for l in range(3)]

    model_body = yolo_body(image_input, num_anchors//3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))
    # model_body = multi_gpu_model(model_body,gpus=2)
    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers)-3)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model

def create_tiny_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/tiny_yolo_weights.h5'):
    '''create the training model, for Tiny YOLOv3'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16}[l], w//{0:32, 1:16}[l], \
        num_anchors//2, num_classes+5)) for l in range(2)]

    model_body = tiny_yolo_body(image_input, num_anchors//2, num_classes)
    print('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze the darknet body or freeze all but 2 output layers.
            num = (20, len(model_body.layers)-2)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.7})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model

def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        # 训练：get_random_data函数，在读取训练图像与标签数据时，也对图像和标签数据进行了等比缩放
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            i = (i+1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)

def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)

def precision(y_true, y_pred):
    # 计算精准率
    # TP=tf.reduce_sum(y_true*tf.round(y_pred))
    # TN=tf.reduce_sum((1-y_true)*(1-tf.round(y_pred)))
    # FP=tf.reduce_sum((1-y_true)*tf.round(y_pred))
    # FN=tf.reduce_sum(y_true*(1-tf.round(y_pred)))
    # precision=TP/(TP+FP)

    # true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    # predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    # precision = true_positives / (predicted_positives + K.epsilon())
    
    TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    N = (-1)*K.sum(K.round(K.clip(y_true-K.ones_like(y_true), -1, 0)))
    TN=K.sum(K.round(K.clip((y_true-K.ones_like(y_true))*(y_pred-K.ones_like(y_pred)), 0, 1)))
    FP=N-TN
    precision = TP / (TP + FP + K.epsilon())
    return precision

def recall(y_true, y_pred):
    # 计算召回率
    # TP=tf.reduce_sum(y_true*tf.round(y_pred))
    # TN=tf.reduce_sum((1-y_true)*(1-tf.round(y_pred)))
    # FP=tf.reduce_sum((1-y_true)*tf.round(y_pred))
    # FN=tf.reduce_sum(y_true*(1-tf.round(y_pred)))
    # recall=TP/(TP+FN)
    
    # true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    # possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    # recall = true_positives / (possible_positives + K.epsilon())
    
    TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    P=K.sum(K.round(K.clip(y_true, 0, 1)))
    FN = P-TP
    recall = TP / (TP + FN + K.epsilon())
    return recall

def fbeta_score(y_true, y_pred, beta=1):
    # # Calculates the F score, the weighted harmonic mean of precision and recall.
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')
    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score
    # TP=tf.reduce_sum(y_true*tf.round(y_pred))
    # TN=tf.reduce_sum((1-y_true)*(1-tf.round(y_pred)))
    # FP=tf.reduce_sum((1-y_true)*tf.round(y_pred))
    # FN=tf.reduce_sum(y_true*(1-tf.round(y_pred)))
    # precision=TP/(TP+FP)
    # recall=TP/(TP+FN)
    # F1score=2*precision*recall/(precision+recall)
    # return F1score

def fmeasure(y_true, y_pred):
    # Calculates the f-measure, the harmonic mean of precision and recall.
    return fbeta_score(y_true, y_pred, beta=1)

# define the function
def training_vis(hist, epochs):
    # 画loss曲线
    history = pd.DataFrame(hist.history)
    max = int(history['loss'][0]) + 10
    # min = history['loss'][epochs-1]
    min = 0
    print(history, '====================================history')
    plt.plot(history['loss'])
    # 画val_loss曲线
    plt.plot(history['val_loss'], color='blue', linewidth=5.0, linestyle='--')
    plt.title('model loss')
    # 坐标轴范围
    plt.xlim((0,6))
    plt.ylim((10, 500))
    # 坐标轴名称
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['val_loss', 'loss'], loc='upper left')
    #设置坐标轴刻度
    my_x_ticks = np.arange(0, epochs, 1)
    # my_y_ticks = np.arange(0, max, int(max/10))
    my_y_ticks = np.arange(0, 18, 0.8)
    plt.xticks(my_x_ticks)
    plt.yticks(my_y_ticks)
    if (epochs == 10):
        plt.savefig('./results/epochs10.jpg')
    else:
        plt.savefig('./results/epochs20.jpg')
    #显示出所有设置
    # plt.show()

if __name__ == '__main__':
    _main()
