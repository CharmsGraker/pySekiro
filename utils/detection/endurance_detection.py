from utils.detection.detector import Detector

import numpy as np
import cv2
import math

from utils.detection.sample_endurance_dict import boss_endurance_toler
from win32_utils.check_key_utils import state_capsLock


class EnduranceDetector(Detector):
    def __init__(self, tolerance_dict, name):
        super().__init__()
        self.tolerance_dict = tolerance_dict
        self.name = name

    def detect(self, img, show_window=False):
        """
        a BGR image
        :param img:
        :return:
        """
        endurance = None
        if len(img.shape) == 3:
            # self_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # to take pixel in middle
            endurance_y = math.floor(4 / 7 * img.shape[0])
            endurance_x = math.floor(0.5 * 4 / 5 * img.shape[1])
            left_endurance_x = math.floor(0.5 * 1 / 5 * img.shape[1])

            height, width = img.shape[:2]

            detect_px = img[endurance_y, endurance_x]
            left_px = img[endurance_y, left_endurance_x]
            # if self.name == 'boss':
            #     print(self_gray[blood_y])

            # after unpack, will get a 1-d arr like, not 2-d
            for level, st_bgr in self.tolerance_dict.items():
                if np.max(np.abs(detect_px - st_bgr)) < 5:
                    if level == 'shinobi' and np.max(np.abs(left_px - st_bgr)) < 5:
                        endurance = 'almost_shinobi'
                    else:
                        endurance = level

            print("endurance level: {}".format(endurance))
            if state_capsLock.checkAndIsPress():
                print("sampled color: {}".format(img[endurance_y, endurance_x]))
            if show_window:
                cv2.line(img, (0, endurance_y), (width, endurance_y), (0, 255, 0), 1)
                cv2.line(img, (endurance_x, 0), (endurance_x, height), (0, 255, 0), 1)
                cv2.line(img, (left_endurance_x, 0), (left_endurance_x, height), (0, 255, 0), 1)

                cv2.imshow(f"{self.name}-endurance", img)
        return endurance


boss_endurance_detector = EnduranceDetector(boss_endurance_toler, name='boss')
