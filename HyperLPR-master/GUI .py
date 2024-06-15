import tkinter as tk
from tkinter import filedialog, Label, ttk, BOTH, LEFT, RIGHT, BOTTOM, X, Y, TOP
from PIL import Image, ImageTk
import cv2
import HyperLPRLite as pr
import threading
import datetime


def recognize_from_photo(file_path, model):
    image = cv2.imread(file_path)
    results = model.SimpleRecognizePlateByE2E(image)
    return results


def recognize_from_video(video_path, model, stop_event, display_results):
    cap = cv2.VideoCapture(video_path)
    cv2.namedWindow('Video Recognition', cv2.WINDOW_NORMAL)

    while cap.isOpened() and not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.SimpleRecognizePlateByE2E(frame)
        for result in results:
            plate, confidence, rect, color = result
            x1, y1, x2, y2 = rect
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{plate} {color} {confidence:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 0), 2)
            display_results([result])

        cv2.imshow('Video Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.getWindowProperty('Video Recognition', cv2.WND_PROP_VISIBLE) < 1:
            stop_event.set()
            break

    cap.release()
    cv2.destroyAllWindows()


def recognize_from_camera(model, stop_event, display_results):
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('Camera Recognition', cv2.WINDOW_NORMAL)

    while cap.isOpened() and not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.SimpleRecognizePlateByE2E(frame)
        for result in results:
            plate, confidence, rect, color = result
            x1, y1, x2, y2 = rect
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{plate} {color} {confidence:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 0), 2)
            display_results([result])

        cv2.imshow('Camera Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.getWindowProperty('Camera Recognition', cv2.WND_PROP_VISIBLE) < 1:
            stop_event.set()
            break

    cap.release()
    cv2.destroyAllWindows()


class LicensePlateRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("车牌识别")
        self.root.state('zoomed')  # Maximize the window

        self.style = ttk.Style()
        self.style.configure('TFrame', background='#f0f0f0')
        self.style.configure('TLabel', background='#f0f0f0', font=("Arial", 12))
        self.style.configure('TButton', font=("Arial", 12), padding=10)
        self.style.configure('TLabelFrame', background='#f0f0f0', font=("Arial", 14))
        self.style.configure('Treeview', font=("Arial", 12))  # 设置Treeview的字体
        self.style.configure('Treeview.Heading', font=("Arial", 14))  # 设置Treeview标题的字体

        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=BOTH, expand=True)

        self.label_frame = ttk.LabelFrame(self.main_frame, text="原图")
        self.label_frame.pack(side=LEFT, fill=BOTH, expand=True, padx=20, pady=20)

        self.image_label = Label(self.label_frame)
        self.image_label.pack(fill=BOTH, expand=True, padx=20, pady=20)

        self.result_frame = ttk.LabelFrame(self.main_frame, text="车牌识别结果")
        self.result_frame.pack(side=RIGHT, fill=Y, padx=20, pady=20)

        self.result_tree = ttk.Treeview(self.result_frame,
                                        columns=("序号", "识别时间", "车牌号", "车牌颜色", "车牌信息", "车牌类型"),
                                        show="headings")
        self.result_tree.heading("序号", text="序号")
        self.result_tree.heading("识别时间", text="识别时间")
        self.result_tree.heading("车牌号", text="车牌号")
        self.result_tree.heading("车牌颜色", text="车牌颜色")
        self.result_tree.heading("车牌信息", text="车牌信息")
        self.result_tree.heading("车牌类型", text="车牌类型")
        self.result_tree.pack(fill=BOTH, expand=True)

        # 设置居中对齐
        for col in self.result_tree["columns"]:
            self.result_tree.column(col, anchor="center")

        # 设置Treeview样式
        self.result_tree.tag_configure("蓝色", background="lightblue")
        self.result_tree.tag_configure("黄色", background="lightyellow")
        self.result_tree.tag_configure("绿色", background="lightgreen")
        self.result_tree.tag_configure("白色", background="white")
        self.result_tree.tag_configure("黑色", background="lightgray")
        self.result_tree.tag_configure("武警车牌", foreground="red")
        self.result_tree.tag_configure("新能源车牌", foreground="green")

        self.button_frame = ttk.Frame(root)
        self.button_frame.pack(side=BOTTOM, fill=X, padx=20, pady=20)

        self.select_image_button = ttk.Button(self.button_frame, text="来自图片", command=self.load_image)
        self.select_image_button.pack(side=LEFT, padx=10, pady=10)

        self.select_video_button = ttk.Button(self.button_frame, text="来自视频", command=self.load_video)
        self.select_video_button.pack(side=LEFT, padx=10, pady=10)

        self.camera_button = ttk.Button(self.button_frame, text="来自摄像头", command=self.use_camera)
        self.camera_button.pack(side=LEFT, padx=10, pady=10)

        self.recognize_button = ttk.Button(self.button_frame, text="识别车牌", command=self.recognize_plate)
        self.recognize_button.pack(side=RIGHT, padx=10, pady=10)

        self.progress_label = ttk.Label(self.button_frame, text="")
        self.progress_label.pack(side=BOTTOM, padx=10, pady=10)

        self.model = pr.LPR("model/cascade.xml", "model/model12.h5", "model/ocr_plate_all_gru.h5")
        self.file_path = None
        self.source_type = None
        self.stop_event = threading.Event()
        self.result_index = 1

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
        self.stop_event.clear()
        self.progress_label.config(text="识别中...")

        if self.source_type == 'image' and self.file_path:
            results = recognize_from_photo(self.file_path, self.model)
            self.display_results(results)
            self.progress_label.config(text="识别完成。")
        elif self.source_type == 'video' and self.file_path:
            threading.Thread(target=recognize_from_video,
                             args=(self.file_path, self.model, self.stop_event, self.display_results)).start()
        elif self.source_type == 'camera':
            threading.Thread(target=recognize_from_camera,
                             args=(self.model, self.stop_event, self.display_results)).start()
        else:
            self.progress_label.config(text="未选择有效的来源。")

    def display_results(self, results):
        for result in results:
            plate, confidence, rect, color = result
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            color_chinese = self.get_color_in_chinese(color)  # 获取车牌颜色的中文描述
            plate_type = self.get_plate_type(plate, color_chinese)  # 修改判断车牌类型的方式
            province_info = self.get_province_info(plate)  # 获取省份信息

            tags = (color_chinese, plate_type)
            self.result_tree.insert("", "end", values=(
            self.result_index, current_time, plate, color_chinese, province_info, plate_type), tags=tags)
            self.result_index += 1

    def get_plate_type(self, plate, color):
        # 根据车牌号和颜色判断车牌类型
        if "红" in color or plate.startswith("WJ"):
            return "武警车牌"
        elif "绿" in color:
            return "新能源车牌"
        else:
            return "普通车牌"

    def get_province_info(self, plate):
        # 返回省份信息
        province_map = {
            "京": "北京市",
            "津": "天津市",
            "沪": "上海市",
            "渝": "重庆市",
            "冀": "河北省",
            "豫": "河南省",
            "云": "云南省",
            "辽": "辽宁省",
            "黑": "黑龙江省",
            "湘": "湖南省",
            "皖": "安徽省",
            "鲁": "山东省",
            "新": "新疆维吾尔自治区",
            "苏": "江苏省",
            "浙": "浙江省",
            "赣": "江西省",
            "鄂": "湖北省",
            "桂": "广西壮族自治区",
            "甘": "甘肃省",
            "晋": "山西省",
            "蒙": "内蒙古自治区",
            "陕": "陕西省",
            "吉": "吉林省",
            "闽": "福建省",
            "贵": "贵州省",
            "粤": "广东省",
            "青": "青海省",
            "藏": "西藏自治区",
            "川": "四川省",
            "宁": "宁夏回族自治区",
            "琼": "海南省",
            "使": "使馆",
            "警": "警牌",
            "学": "学牌",
            "港": "香港特别行政区",
            "澳": "澳门特别行政区",
            "台": "台湾省"
        }
        return province_map.get(plate[0], "未知省份")

    def get_color_in_chinese(self, color):
        color_map = {
            "blue": "蓝色",
            "yellow": "黄色",
            "green": "绿色",
            "white": "白色",
            "black": "黑色",
            "red": "红色"
        }
        return color_map.get(color, "未知颜色")


if __name__ == "__main__":
    root = tk.Tk()
    app = LicensePlateRecognitionApp(root)
    root.mainloop()
