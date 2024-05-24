class RecognitionResult:
    def __init__(self, plate_number, confidence, bounding_box, color):
        """
        初始化识别结果
        :param plate_number: 车牌号码
        :param confidence: 识别置信度
        :param bounding_box: 车牌区域（左上角和右下角坐标）
        :param color: 车牌颜色
        """
        self.plate_number = plate_number
        self.confidence = confidence
        self.bounding_box = bounding_box
        self.color = color

class PlateFormatRules:
    def __init__(self, formats, valid_characters):
        """
        初始化车牌格式规则
        :param formats: 车牌格式规则列表
        :param valid_characters: 合法字符集
        """
        self.formats = formats
        self.valid_characters = valid_characters

class MultiFrameResults:
    def __init__(self):
        """
        初始化多帧识别结果存储
        """
        self.results = []

    def add_result(self, recognition_result):
        """
        添加识别结果
        :param recognition_result: 单帧识别结果
        """
        self.results.append(recognition_result)

    def get_voting_result(self):
        """
        获取多帧投票结果
        :return: 最终识别结果
        """
        from collections import Counter
        plate_numbers = [result.plate_number for result in self.results]
        most_common_plate = Counter(plate_numbers).most_common(1)[0][0]
        return most_common_plate

class ImageData:
    def __init__(self, image, source='file', frame_number=0):
        """
        初始化图像数据
        :param image: 图像数据（OpenCV格式）
        :param source: 图像来源，file表示文件，camera表示摄像头
        :param frame_number: 视频帧号，仅当来源为视频时使用
        """
        self.image = image
        self.source = source
        self.frame_number = frame_number

class UserInterfaceData:
    def __init__(self):
        """
        初始化用户界面数据
        """
        self.image_path = None
        self.video_path = None
        self.camera_stream = None

    def load_image(self, path):
        """
        加载图像文件
        :param path: 图像文件路径
        """
        self.image_path = path
        self.camera_stream = None

    def load_video(self, path):
        """
        加载视频文件
        :param path: 视频文件路径
        """
        self.video_path = path
        self.camera_stream = None

    def start_camera(self, stream):
        """
        启动实时摄像头视频流
        :param stream: 摄像头视频流
        """
        self.camera_stream = stream
        self.image_path = None
        self.video_path = None

