import time
import cv2
import numpy as np
from PIL import ImageFont, Image, ImageDraw
import HyperLPRLite as pr  # 引入LPR大类

fontC = ImageFont.truetype("./Font/platech.ttf", 14, 0)

def drawRectBox(image, rect, addText):  # 定义划定车牌矩形框函数
    cv2.rectangle(image, (int(rect[0]), int(rect[1])), (int(rect[0] + rect[2]), int(rect[1] + rect[3])), (0, 0, 255), 2, cv2.LINE_AA)
    cv2.rectangle(image, (int(rect[0]-1), int(rect[1])-16), (int(rect[0] + 115), int(rect[1])), (0, 0, 255), -1, cv2.LINE_AA)
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    draw.text((int(rect[0] + 1), int(rect[1] - 16)), addText.encode('utf-8').decode('utf-8'), (255, 255, 255), font=fontC)
    imagex = np.array(img)
    return imagex

def process_frame(model, frame):
    results = model.SimpleRecognizePlateByE2E(frame)
    for pstr, confidence, rect, plate_color in results:
        if confidence > 0.7:
            frame = drawRectBox(frame, rect, f"{pstr} {plate_color} {round(confidence, 3)}")
            print("plate_str:", pstr)
            print("plate_confidence:", confidence)
            print("plate_color:", plate_color)
    return frame

def recognize_from_video(video_path, model):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = process_frame(model, frame)
        cv2.imshow("Video Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def recognize_from_camera(model):
    cap = cv2.VideoCapture(0)  # 0表示第一个摄像头
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = process_frame(model, frame)
        cv2.imshow("Camera Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def recognize_from_photo(photo_path, model):
    grr = cv2.imread(photo_path)
    for pstr, confidence, rect, plate_color in model.SimpleRecognizePlateByE2E(grr):
        if confidence > 0.7:
            image = drawRectBox(grr, rect, f"{pstr} {plate_color} {round(confidence, 3)}")
            print("plate_str:", pstr)
            print("plate_confidence:", confidence)
            print("plate_color:", plate_color)  # 输出车牌颜色

    cv2.imshow("image", image)
    cv2.waitKey(0)

def SpeedTest(image_path, model):  # 定义测试速度函数
    grr = cv2.imread(image_path)
    model.SimpleRecognizePlateByE2E(grr)
    t0 = time.time()
    for _ in range(20):
        model.SimpleRecognizePlateByE2E(grr)
    t = (time.time() - t0)/20.0  # 计算运行时间
    print(f"Image size: {grr.shape[1]}x{grr.shape[0]} need {round(t*1000, 2)}ms")

if __name__ == "__main__":
    model = pr.LPR("model/cascade.xml", "model/model12.h5", "model/ocr_plate_all_gru.h5")  # 输入之前训练好的模型权重

    choice = input("输入‘photo’图像识别；输入‘video’视频识别；输入‘camera’摄像头实时识别: ")

    if choice == 'photo':
        photo_path = input("请输入图片路径: ")
        recognize_from_photo(photo_path, model)
    elif choice == 'video':
        video_path = input("请输入视频路径: ")
        recognize_from_video(video_path, model)
    elif choice == 'camera':
        recognize_from_camera(model)
    else:
        print("无效选择!")

