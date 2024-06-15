
from keras import backend as K
from keras.models import *
from keras.layers import *
from . import e2e

'''
这段代码是用于构建端到端（End-to-End）的OCR（光学字符识别）模型的函数。主要功能包括：

1. 导入必要的Keras库和模块。
2. 定义了一个CTC（Connectionist Temporal Classification）损失函数 `ctc_lambda_func`，用于在模型训练中计算损失。
3. 定义了一个构建模型的函数 `construct_model`，用于构建端到端OCR模型。
   - 输入尺寸为(None, 40, 3)，表示输入图像的高度为40像素，宽度不固定（None），通道数为3（RGB）。
   - 使用卷积神经网络（CNN）提取图像特征，包括多层卷积、批量归一化和激活函数。
   - 最后一层卷积层输出的通道数为目标字符集的大小加一（`len(e2e.chars)+1`），用于预测字符概率。
   - 使用softmax激活函数对最后一层卷积输出进行概率归一化。
   - 加载预训练的模型权重。
4. 这个模型的结构适用于端到端的OCR任务，其中模型直接从原始图像中学习到字符序列的表示，无需手动进行字符分割和识别。

可用于端到端的OCR任务，例如车牌识别、文本检测和识别等应用场景。
'''


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def construct_model(model_path):
    input_tensor = Input((None, 40, 3))
    x = input_tensor
    base_conv = 32

    for i in range(3):
        x = Conv2D(base_conv * (2 ** (i)), (3, 3),padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(256, (5, 5))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(1024, (1, 1))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(len(e2e.chars)+1, (1, 1))(x)
    x = Activation('softmax')(x)
    base_model = Model(inputs=input_tensor, outputs=x)
    base_model.load_weights(model_path)
    return base_model
