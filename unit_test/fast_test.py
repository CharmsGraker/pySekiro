import time

from settings import window_size, blood_window, pause_key
from strategy.judge import judge_blood4reward
from utils.blood_detection import boss_blood_detector, self_blood_detector
from win32_utils.getkeys import key_check
from win32_utils.grabscreen import grab_screen
from utils.count_down import CountDown

boss_blood_height = 42
boss_blood_h0 = 50

boss_blood_width = 222

self_blood_h0 = 515
self_blood_height = 50
self_blood_width = 263

boss_blood_window = (57,
                     boss_blood_h0,
                     57 + boss_blood_width,
                     boss_blood_h0 + boss_blood_height)
self_blood_window = (57,
                     self_blood_h0,
                     57 + self_blood_width,
                     self_blood_h0 + self_blood_height)

import cv2


if __name__ =='__main__':
    CountDown(3)
    boss_blood_area = grab_screen(boss_blood_window)
    self_blood_area = grab_screen(self_blood_window)

    boss_blood = boss_blood_detector(boss_blood_area, show_window=True)
    self_blood = boss_blood_detector(boss_blood_area, show_window=True)
    paused = False
    while True:
        while True:
            if pause_key in key_check():
                if paused:
                    paused = False
                else:
                    paused = True
                    print("paused")
                    time.sleep(1)
                    continue
            if paused == False:
                break


        last_time = time.time()
        screen = cv2.cvtColor(grab_screen(window_size),cv2.COLOR_RGBA2RGB)
        # blood_img = grab_screen(blood_window)
        # cv2.imshow("screen", screen)
        # cv2.imshow("blood_window", blood_img)
        boss_blood_area = grab_screen(boss_blood_window)
        self_blood_area = grab_screen(self_blood_window)

        next_boss_blood = boss_blood_detector(boss_blood_area, show_window=True)
        next_self_blood = self_blood_detector(self_blood_area, show_window=True)
        reward, done, stop, emergence_break = judge_blood4reward(boss_blood,
                                                                 next_boss_blood,
                                                                 self_blood,
                                                                 next_self_blood)
        print("r: {},self: {}/{}".format(reward,self_blood,next_self_blood) )
        self_blood = next_self_blood
        boss_blood = next_boss_blood
        cv2.imshow("screen", screen)
        gray_screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)

        print("boss blood: {},self blood: {}".format(boss_blood, self_blood))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        print("took {}".format(time.time()-last_time))
