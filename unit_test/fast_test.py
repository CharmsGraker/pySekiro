import time

from settings import window_size, pause_key
from strategy.evaluate.evaluate_reward import judge_blood4reward
from utils.detection.blood_detection import boss_blood_detector, self_blood_detector, clip_screen
from utils.detection.endurance_detection import boss_endurance_detector
from win32_utils.getkeys import key_check
from win32_utils.grabscreen import grab_screen

boss_blood_height = 35
boss_blood_h0 = 50

boss_blood_width = 220

self_blood_h0 = 515
self_blood_height = 50
self_blood_width = 263

blood_window = (59,
                55,
                59 + self_blood_width,
                563)  # (60, 91, 280, 562)

# w0,h0,w1,h1
boss_blood_abs_window = (57,
                         boss_blood_h0,
                         57 + boss_blood_width,
                         boss_blood_h0 + boss_blood_height)
self_blood_abs_window = (57,
                         self_blood_h0,
                         57 + self_blood_width,
                         self_blood_h0 + self_blood_height)

boss_blood_offset_window = (0,
                            0,
                            boss_blood_width,
                            boss_blood_height)

self_blood_offset_window = (0,
                            blood_window[3] - blood_window[1] - self_blood_height,
                            self_blood_width,
                            blood_window[3] - blood_window[1])

endurance_offset_window = (0,
                           0,
                           window_size[2],  # keep the same len of state window
                           25)

import cv2

if __name__ == '__main__':
    # CountDown(3)
    show_window = False
    boss_blood_area = grab_screen(boss_blood_abs_window)
    self_blood_area = grab_screen(self_blood_abs_window)

    boss_blood = boss_blood_detector(boss_blood_area, show_window=show_window)
    self_blood = boss_blood_detector(boss_blood_area, show_window=show_window)
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
        screen = cv2.cvtColor(grab_screen(window_size), cv2.COLOR_RGBA2RGB)

        endurance_area = clip_screen(screen, endurance_offset_window)

        # after cvtColor, W,H image will convert to H,W image
        blood_screen = cv2.cvtColor(grab_screen(blood_window), cv2.COLOR_RGBA2RGB)
        # blood_img = grab_screen(blood_window)
        # cv2.imshow("screen", screen)
        # cv2.imshow("blood_window", blood_img)
        boss_blood_area = clip_screen(blood_screen, boss_blood_offset_window)
        self_blood_area = clip_screen(blood_screen, self_blood_offset_window)

        # cv2.imshow("endurance_area", endurance_area)

        next_boss_blood = boss_blood_detector(boss_blood_area, show_window=True)
        next_self_blood = self_blood_detector(self_blood_area, show_window=True)
        next_boss_endurance = boss_endurance_detector(endurance_area, show_window=True)
        reward, done = judge_blood4reward(boss_blood,
                                          next_boss_blood,
                                          self_blood,
                                          next_self_blood,
                                          next_boss_endurance
                                          )
        # print("r: {},self: {}/{}".format(reward, self_blood,next_self_blood) )
        self_blood = next_self_blood
        boss_blood = next_boss_blood
        cv2.imshow("blood_screen", blood_screen)
        gray_screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)

        # print("boss blood: {},self blood: {}".format(boss_blood, self_blood))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        print("took {}".format(time.time() - last_time))
