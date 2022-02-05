import math
import cv2


class BloodDetector:
    def __init__(self, low_tolerance, high_tolerance, name):
        self.low_gray = low_tolerance
        self.high_gray = high_tolerance
        self.name = name

    def __call__(self, *args, **kwargs):
        return self.detect(*args, **kwargs)

    def detect(self, img):
        """
        a BGR image
        :param img:
        :return:
        """
        blood = 0
        if len(img.shape) == 3:
            self_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blood_y = math.floor(4 / 5 * self_gray.shape[0])
            height, width = img.shape[:2]

            # if self.name == 'boss':
            #     print(self_gray[blood_y])
            start = width
            end = 0
            for idx, x_pixel in enumerate(self_gray[blood_y]):
                # self blood gray pixel 80~98
                if self.low_gray < x_pixel < self.high_gray:
                    blood += 1
                    start = min(start, idx)
                    end = max(end, idx)
            cv2.line(self_gray, (0, blood_y), (width, blood_y), (0, 255, 0), 1)

            cv2.line(self_gray, (start, 0), (start, width), (0, 255, 0), 1)
            cv2.line(self_gray, (end, 0), (end, width), (0, 255, 0), 1)

            # cv2.imshow(f'{self.name}_blood_area', self_gray)

        return blood


self_blood_detector = BloodDetector(71, 83, name='self')
boss_blood_detector = BloodDetector(50, 75, name='boss')
