import time

from settings import window_size, blood_window
from utils.blood_detection import boss_blood_detector, self_blood_detector
from win32_utils.grabscreen import grab_screen

boss_blood_height = 40
boss_blood_h0 = 50

boss_blood_width = 226

self_blood_h0 = 515
self_blood_height = 50
self_blood_width = 270

boss_blood_window = (57,
                     boss_blood_h0,
                     57 + boss_blood_width,
                     boss_blood_h0 + boss_blood_height)
self_blood_window = (55,
                     self_blood_h0,
                     55 + self_blood_width,
                     self_blood_h0 + self_blood_height)

import cv2
if __name__ =='__main__':
    while True:
        last_time = time.time()
        screen = grab_screen(window_size)
        # blood_img = grab_screen(blood_window)
        # cv2.imshow("screen", screen)
        # cv2.imshow("blood_window", blood_img)
        boss_blood_area = grab_screen(boss_blood_window)
        self_blood_area = grab_screen(self_blood_window)

        boss_blood = boss_blood_detector(boss_blood_area)
        self_blood = self_blood_detector(self_blood_area)

        gray_screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)

        print("boss blood: {},self blood: {}".format(boss_blood, self_blood))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        print("took {}".format(time.time()-last_time))
