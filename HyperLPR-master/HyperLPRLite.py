import cv2
import numpy as np
from keras import backend as K
from keras.models import *
from keras.layers import *

chars = [u"京", u"沪", u"津", u"渝", u"冀", u"晋", u"蒙", u"辽", u"吉", u"黑", u"苏", u"浙", u"皖", u"闽", u"赣", u"鲁", u"豫", u"鄂", u"湘", u"粤", u"桂",
         u"琼", u"川", u"贵", u"云", u"藏", u"陕", u"甘", u"青", u"宁", u"新", u"0", u"1", u"2", u"3", u"4", u"5", u"6", u"7", u"8", u"9", u"A",
         u"B", u"C", u"D", u"E", u"F", u"G", u"H", u"J", u"K", u"L", u"M", u"N", u"P", u"Q", u"R", u"S", u"T", u"U", u"V", u"W", u"X",
         u"Y", u"Z",u"港",u"学",u"使",u"警",u"澳",u"挂",u"军",u"北",u"南",u"广",u"沈",u"兰",u"成",u"济",u"海",u"民",u"航",u"空"
         ]

# 常见车牌颜色的HSV范围
color_ranges = {
    'blue': [(100, 43, 46), (124, 255, 255)],
    'green': [(35, 43, 46), (77, 255, 255)],
    'white': [(0, 0, 221), (180, 30, 255)],
    'yellow': [(26, 43, 46), (34, 255, 255)],
    'black': [(0, 0, 0), (180, 255, 46)]

}


class LPR():
    def __init__(self, model_detection, model_finemapping, model_seq_rec):
        self.watch_cascade = cv2.CascadeClassifier(model_detection)
        self.modelFineMapping = self.model_finemapping()
        self.modelFineMapping.load_weights(model_finemapping)
        self.modelSeqRec = self.model_seq_rec(model_seq_rec)

    # 计算一个矩形区域在图像中的安全范围，防止出现超出图像边界的情况
    def computeSafeRegion(self, shape, bounding_rect):
        top = bounding_rect[1]
        bottom = bounding_rect[1] + bounding_rect[3]
        left = bounding_rect[0]
        right = bounding_rect[0] + bounding_rect[2]
        min_top = 0
        max_bottom = shape[0]
        min_left = 0
        max_right = shape[1]
        if top < min_top:
            top = min_top
        if left < min_left:
            left = min_left
        if bottom > max_bottom:
            bottom = max_bottom
        if right > max_right:
            right = max_right
        return [left, top, right-left, bottom-top]

    # 根据给定的矩形区域裁剪图像，返回裁剪后的图像区域
    def cropImage(self, image, rect):
        x, y, w, h = self.computeSafeRegion(image.shape, rect)
        return image[y:y+h, x:x+w]

    # 对图像进行车牌的粗略定位，返回包含粗定位结果的图像区域列表
    def detectPlateRough(self, image_gray, resize_h=720, en_scale=1.08, top_bottom_padding_rate=0.05):
        if top_bottom_padding_rate > 0.2:
            print("error:top_bottom_padding_rate > 0.2:", top_bottom_padding_rate)
            exit(1)
        height = image_gray.shape[0]
        padding = int(height*top_bottom_padding_rate)
        scale = image_gray.shape[1]/float(image_gray.shape[0])
        image = cv2.resize(image_gray, (int(scale*resize_h), resize_h))
        image_color_cropped = image[padding:resize_h-padding, 0:image_gray.shape[1]]
        image_gray = cv2.cvtColor(image_color_cropped, cv2.COLOR_RGB2GRAY)
        watches = self.watch_cascade.detectMultiScale(image_gray, en_scale, 2, minSize=(36, 9), maxSize=(36*40, 9*40))
        cropped_images = []
        for (x, y, w, h) in watches:
            x -= w * 0.14
            w += w * 0.28
            y -= h * 0.15
            h += h * 0.3
            cropped = self.cropImage(image_color_cropped, (int(x), int(y), int(w), int(h)))
            cropped_images.append([cropped, [x, y+padding, w, h]])
        return cropped_images

    # 将神经网络输出的概率序列快速解码为车牌号码和置信度
    def fastdecode(self, y_pred):
        results = ""
        confidence = 0.0
        table_pred = y_pred.reshape(-1, len(chars)+1)
        res = table_pred.argmax(axis=1)
        for i, one in enumerate(res):
            if one < len(chars):
                results += chars[one]
                confidence += table_pred[i][one]
        confidence /= len(results)
        return results, confidence

    # 定义了一个序列识别模型，用于识别车牌号码
    def model_seq_rec(self, model_path):
        width, height, n_len, n_class = 164, 48, 7, len(chars) + 1
        rnn_size = 256
        input_tensor = Input((164, 48, 3))
        x = input_tensor
        base_conv = 32
        for i in range(3):
            x = Conv2D(base_conv * (2 ** (i)), (3, 3))(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)
        conv_shape = x.get_shape()
        x = Reshape(target_shape=(int(conv_shape[1]), int(conv_shape[2] * conv_shape[3])))(x)
        x = Dense(32)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        gru_1 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru1', reset_after=False)(x)
        gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b',
                     reset_after=False)(x)
        gru1_merged = add([gru_1, gru_1b])
        gru_2 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru2', reset_after=False)(gru1_merged)
        gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b',
                     reset_after=False)(gru1_merged)
        x = concatenate([gru_2, gru_2b])
        x = Dropout(0.25)(x)
        x = Dense(n_class, kernel_initializer='he_normal', activation='softmax')(x)
        base_model = Model(inputs=input_tensor, outputs=x)
        base_model.load_weights(model_path)
        return base_model

    # 定义了一个细分割模型，用于对车牌图像进行细粒度的定位
    def model_finemapping(self):
        input = Input(shape=[16, 66, 3])
        x = Conv2D(10, (3, 3), strides=1, padding='valid', name='conv1')(input)
        x = Activation("relu", name='relu1')(x)
        x = MaxPool2D(pool_size=2)(x)
        x = Conv2D(16, (3, 3), strides=1, padding='valid', name='conv2')(x)
        x = Activation("relu", name='relu2')(x)
        x = Conv2D(32, (3, 3), strides=1, padding='valid', name='conv3')(x)
        x = Activation("relu", name='relu3')(x)
        x = Flatten()(x)
        output = Dense(2, name="dense")(x)
        output = Activation("relu", name='relu4')(output)
        model = Model([input], [output])
        return model

    # 用于对车牌图像进行垂直方向的细分割，以获取更准确的车牌图像区域
    def finemappingVertical(self, image, rect):
        resized = cv2.resize(image, (66, 16))
        resized = resized.astype(float)/255
        res_raw = self.modelFineMapping.predict(np.array([resized]))[0]
        res = res_raw * image.shape[1]
        res = res.astype(int)
        H, T = res
        H -= 3
        if H < 0:
            H = 0
        T += 2
        if T >= image.shape[1]-1:
            T = image.shape[1]-1
        rect[2] -= rect[2] * (1 - res_raw[1] + res_raw[0])
        rect[0] += res[0]
        image = image[:, H:T+2]
        image = cv2.resize(image, (int(136), int(36)))
        return image, rect

    # 用于对给定的车牌图像进行字符识别，返回识别结果和置信度
    def recognizeOne(self, src):
        x_tempx = src
        x_temp = cv2.resize(x_tempx, (164, 48))
        x_temp = x_temp.transpose(1, 0, 2)
        y_pred = self.modelSeqRec.predict(np.array([x_temp]))
        y_pred = y_pred[:, 2:, :]
        return self.fastdecode(y_pred)

    # 用于检测车牌的颜色，以辅助车牌的识别
    def detect_plate_color(self, image):
        """检测车牌颜色"""
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        for color, (lower, upper) in color_ranges.items():
            mask = cv2.inRange(hsv_image, np.array(lower), np.array(upper))
            if cv2.countNonZero(mask) > 0:
                return color
        return "unknown"

    # 用于对给定的图像进行端到端的车牌识别，包括粗定位、细定位、字符识别和颜色检测，最终返回识别结果列表
    def SimpleRecognizePlateByE2E(self, image):
        images = self.detectPlateRough(image, image.shape[0], top_bottom_padding_rate=0.1)
        res_set = []
        for j, plate in enumerate(images):
            plate, rect = plate
            image_rgb, rect_refine = self.finemappingVertical(plate, rect)
            res, confidence = self.recognizeOne(image_rgb)
            plate_color = self.detect_plate_color(image_rgb)
            res_set.append([res, confidence, rect_refine, plate_color])
        return res_set