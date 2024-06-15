#coding=utf-8
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPool2D
from keras.optimizers import SGD
from keras import backend as K

K.set_image_dim_ordering('tf')


import cv2
import numpy as np


'''
这段代码似乎是用于识别车牌类型的模型。以下是代码中的主要部分：

1. **模型结构定义（`Getmodel_tensorflow`）**：使用了卷积神经网络（CNN）来识别车牌类型。模型包括了卷积层、池化层、全连接层和softmax输出层。输入图像的尺寸为 9x34 像素。

2. **模型训练与加载**：模型已经被训练，并通过加载预先训练好的权重来使用。预先训练好的权重被加载到模型中，以便进行预测。

3. **`SimplePredict` 函数**：这个函数接受一个图像作为输入，并对其进行预测。它首先将输入图像调整为模型所需的大小（34x9），然后对其进行归一化处理。最后，使用训练好的模型对图像进行预测，并返回预测结果的索引，表示车牌的类型。

代码用于创建、训练和使用一个简单的卷积神经网络模型，用于识别车牌的类型，包括蓝牌、单层黄牌、新能源车牌、白色和黑色-港澳。
'''

plateType  = ["蓝牌","单层黄牌","新能源车牌","白色","黑色-港澳"]
def Getmodel_tensorflow(nb_classes):
    # nb_classes = len(charset)

    img_rows, img_cols = 9, 34
    # number of convolutional filters to use
    nb_filters = 32
    # size of pooling area for max pooling
    nb_pool = 2
    # convolution kernel size
    nb_conv = 3

    # x = np.load('x.npy')
    # y = np_utils.to_categorical(range(3062)*45*5*2, nb_classes)
    # weight = ((type_class - np.arange(type_class)) / type_class + 1) ** 3
    # weight = dict(zip(range(3063), weight / weight.mean()))  # 调整权重，高频字优先

    model = Sequential()
    model.add(Conv2D(16, (5, 5),input_shape=(img_rows, img_cols,3)))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(nb_pool, nb_pool)))
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

model = Getmodel_tensorflow(5)
model.load_weights("./model/plate_type.h5")
model.save("./model/plate_type.h5")
def SimplePredict(image):
    image = cv2.resize(image, (34, 9))
    image = image.astype(float) / 255
    res = np.array(model.predict(np.array([image]))[0])
    return res.argmax()


