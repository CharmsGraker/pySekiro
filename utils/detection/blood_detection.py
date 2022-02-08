import math
import cv2
import numpy as np

from thirdparty.api import HED_predict
from utils.detection.detector import Detector


def gamma_correction(img_original, gamma=0.8):
    lookUpTable = np.empty((1, 256), np.uint8)
    for i in range(256):
        lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    res = cv2.LUT(img_original, lookUpTable)
    return res


class BloodDetector(Detector):
    def __init__(self, low_tolerance, high_tolerance, name):
        super().__init__()
        self.low_tolerance = low_tolerance
        self.high_tolerance = high_tolerance

        if isinstance(low_tolerance, int) and isinstance(high_tolerance, int):
            self.use_color = False
        else:
            assert len(low_tolerance) == len(high_tolerance)
            if isinstance(low_tolerance, list) and len(low_tolerance) == 3:
                self.low_tolerance = np.array(low_tolerance)
                self.high_tolerance = np.array(high_tolerance)
                self.use_color = True
        self.name = name
        self.blood_capacity = None

    def detect(self, img, show_window=False):
        """
        a BGR image
        :param img:
        :return:
        """
        blood = 0
        if len(img.shape) == 3:
            if self.use_color:
                self.accept(img)
            else:
                self_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                blood_y = math.floor(6 / 7 * self_gray.shape[0])
                height, width = img.shape[:2]

                # if self.name == 'boss':
                #     print(self_gray[blood_y])
                # to initialize blood, and decide sample frequent
                self.blood_capacity = width

                start = width
                end = 0
                for idx, x_pixel in enumerate(self_gray[blood_y]):
                    # self blood gray pixel 80~98
                    if self.low_tolerance < x_pixel < self.high_tolerance:
                        blood += 1
                        start = min(start, idx)
                        end = max(end, idx)

                # dp = 1.0
                # minDist = 20
                # param1 = 399
                # param2 = 200
                # min_radius = 2
                # max_radius = 4
                #
                # circle_color = img  # [:blood_y-10, :80]
                # circle_color = gamma_correction(circle_color, gamma=1.2)
                # sketched = self.edge_detection(cv2.cvtColor(circle_color, cv2.COLOR_BGR2RGB))
                # sketched = cv2.threshold(sketched, 100, 255, cv2.THRESH_BINARY)[1]
                # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 定义矩形结构元素
                # sketched = cv2.morphologyEx(sketched, cv2.MORPH_CLOSE, kernel, iterations=1)  # 开运算1
                #
                # sketched = cv2.Canny(sketched, 50, 100)
                #
                # circles = cv2.HoughCircles(sketched, cv2.HOUGH_GRADIENT, dp, minDist, param1, param2, min_radius,
                #                            max_radius)

                if show_window:
                    cv2.line(self_gray, (0, blood_y), (width, blood_y), (0, 255, 0), 1)

                    cv2.line(self_gray, (start, 0), (start, width), (0, 255, 0), 1)
                    cv2.line(self_gray, (end, 0), (end, width), (0, 255, 0), 1)

                    # if circles is not None:
                    #     circles = np.uint16(np.around(circles))
                    #     for cirle in circles[0, :]:
                    #         try:
                    #             x, y, r = cirle
                    #             # print(x, y, r)
                    #             cv2.circle(circle_color, [x, y], r, [0, 0, 255], 1)
                    #         except Exception as e:
                    #             print(e)
                    # cv2.imshow(f"{self.name}-cannyed_gray", sketched)
                    # cv2.imshow(f"{self.name}-circle", circle_color)
                    cv2.imshow(f'{self.name}_blood_area', self_gray)

        return blood

    def accept(self, img):
        pass

    def edge_detection(self, img):
        """
        input the RGB image
        :param img:
        :return:
        """

        # self_gray_clip = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # cv2.Canny(self_gray_clip, 254, 255)
        return HED_predict(img)

# this param is only take for mistake RGB as BGR
self_blood_detector = BloodDetector(69, 83, name='self')
boss_blood_detector = BloodDetector(50, 75, name='boss')

# here is the correct BGR param
# self_blood_detector = BloodDetector(50, 99, name='self')
# boss_blood_detector = BloodDetector(40, 43, name='boss')


# screen is RGB,  W,H,C
def clip_screen(screen, window,data_format='HW'):
    if data_format == 'WH':
        return np.array(screen[window[0]:window[2],window[1]:window[3], :])
    elif data_format == 'HW':
        return np.array(screen[window[1]:window[3],window[0]:window[2], :])