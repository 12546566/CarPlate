import tkinter as tk
from tkinter import filedialog, Label, Button, LEFT, RIGHT, TOP, BOTTOM, X, Y, BOTH
from PIL import Image, ImageTk
import cv2
import HyperLPRLite as pr
import threading

# Sample implementations of the recognize functions
def recognize_from_photo(file_path, model):
    image = cv2.imread(file_path)
    results = model.SimpleRecognizePlateByE2E(image)
    return results

def recognize_from_video(video_path, model):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.SimpleRecognizePlateByE2E(frame)
        for result in results:
            plate, confidence, rect, color = result
            x1, y1, x2, y2 = rect
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{plate} {color} {confidence:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow('Video Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def recognize_from_camera(model):
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.SimpleRecognizePlateByE2E(frame)
        for result in results:
            plate, confidence, rect, color = result
            x1, y1, x2, y2 = rect
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{plate} {color} {confidence:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow('Camera Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

class LicensePlateRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("车牌识别")
        self.root.state('zoomed')  # Maximize the window

        self.label = Label(root, text="原图", font=("Arial", 16))
        self.label.pack(anchor='nw', padx=20, pady=20)

        self.image_label = Label(root)
        self.image_label.pack(side=LEFT, fill=BOTH, expand=True, padx=20, pady=20)

        self.result_frame = tk.Frame(root)
        self.result_frame.pack(side=RIGHT, fill=Y, padx=20, pady=20)

        self.result_label = Label(self.result_frame, text="车牌识别结果：", font=("Arial", 14))
        self.result_label.pack(anchor='n', pady=10)

        self.plate_result_label = Label(self.result_frame, text="", font=("Arial", 12))
        self.plate_result_label.pack(anchor='n', pady=10)

        self.color_label = Label(self.result_frame, text="车牌颜色：", font=("Arial", 14))
        self.color_label.pack(anchor='n', pady=10)

        self.plate_color_label = Label(self.result_frame, text="", font=("Arial", 12))
        self.plate_color_label.pack(anchor='n', pady=10)

        self.button_frame = tk.Frame(root)
        self.button_frame.pack(side=BOTTOM, fill=X, padx=20, pady=20)

        self.select_image_button = Button(self.button_frame, text="来自图片", font=("Arial", 12), command=self.load_image)
        self.select_image_button.pack(side=LEFT, padx=10, pady=10)

        self.select_video_button = Button(self.button_frame, text="来自视频", font=("Arial", 12), command=self.load_video)
        self.select_video_button.pack(side=LEFT, padx=10, pady=10)

        self.camera_button = Button(self.button_frame, text="来自摄像头", font=("Arial", 12), command=self.use_camera)
        self.camera_button.pack(side=LEFT, padx=10, pady=10)

        self.recognize_button = Button(self.button_frame, text="识别车牌", font=("Arial", 12), command=self.recognize_plate)
        self.recognize_button.pack(side=RIGHT, padx=10, pady=10)

        self.model = pr.LPR("model/cascade.xml", "model/model12.h5", "model/ocr_plate_all_gru.h5")
        self.file_path = None
        self.source_type = None

    def load_image(self):
        self.file_path = filedialog.askopenfilename()
        if self.file_path:
            self.source_type = 'image'
            image = Image.open(self.file_path)
            self.display_image(image)

    def display_image(self, image):
        image.thumbnail((self.image_label.winfo_width(), self.image_label.winfo_height()), Image.LANCZOS)
        self.image = ImageTk.PhotoImage(image)
        self.image_label.config(image=self.image)

    def load_video(self):
        self.file_path = filedialog.askopenfilename()
        if self.file_path:
            self.source_type = 'video'
            self.image_label.config(text="已选择视频: " + self.file_path)

    def use_camera(self):
        self.source_type = 'camera'
        self.image_label.config(text="使用摄像头进行实时识别。")

    def recognize_plate(self):
        if self.source_type == 'image' and self.file_path:
            results = recognize_from_photo(self.file_path, self.model)
            self.display_results(results)
        elif self.source_type == 'video' and self.file_path:
            threading.Thread(target=recognize_from_video, args=(self.file_path, self.model)).start()
        elif self.source_type == 'camera':
            threading.Thread(target=recognize_from_camera, args=(self.model,)).start()
        else:
            self.plate_result_label.config(text="未选择有效的来源。")

    def display_results(self, results):
        if results:
            plate_text = "\n".join([f"车牌: {res[0]}, 置信度: {res[1]}" for res in results])
            color_text = "\n".join([f"颜色: {res[3]}" for res in results])
        else:
            plate_text = "未检测到车牌。"
            color_text = ""
        self.plate_result_label.config(text=plate_text)
        self.plate_color_label.config(text=color_text)

if __name__ == "__main__":
    root = tk.Tk()
    app = LicensePlateRecognitionApp(root)
    root.mainloop()