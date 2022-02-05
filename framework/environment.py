import time

import cv2
import numpy as np

from fast_test import boss_blood_window, self_blood_window
from settings import window_size, blood_window, HEIGHT, WIDTH
from strategy.judge import judge_blood4reward
from utils.blood_detection import boss_blood_detector, self_blood_detector
from win32_utils.grabscreen import grab_screen


class Enviroment:
    def __init__(self):
        pass

    @staticmethod
    def make(env_name):
        if env_name == 'Sekiro':
            return SekiroEnv()


class Config:
    def __init__(self, dict):
        self.__dict__.update(dict)


class SekiroEnv:

    def __init__(self):
        self.action_space = np.empty([15])
        self.state = None
        self.boss_blood = None
        self.sekiro_blood = None

    def get_observation_space(self):
        """
        use RGB for feature map inputs
        :return:
        """
        return Config({"state_w": WIDTH,
                       "state_h": HEIGHT,
                       "state_d": 3})

    def reset(self):
        """
        initialize
        :return:
        """
        screen = cv2.cvtColor(grab_screen(window_size), cv2.COLOR_RGBA2RGB)
        # screen_gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)

        # collect station gray graph
        # blood_window_gray = cv2.cvtColor(grab_screen(blood_window), cv2.COLOR_BGR2GRAY)

        # collect blood gray graph for count self and boss blood
        self.state = cv2.resize(screen, (WIDTH, HEIGHT))
        print(self.state.shape)
        self.state = np.array(self.state).reshape([WIDTH, HEIGHT, 3])  # [0]
        # change graph to WIDTH * HEIGHT for station input
        boss_blood_area = grab_screen(boss_blood_window)
        self_blood_area = grab_screen(self_blood_window)

        self.boss_blood = boss_blood_detector(boss_blood_area)
        self.sekiro_blood = self_blood_detector(self_blood_area)

        # count init blood
        self.target_step = 0
        # used to update target Q network
        self.done = 0
        self.total_reward = 0
        self.stop = 0
        # 用于防止连续帧重复计算reward
        self.last_time = time.time()

        self.emergency_break = 0
        return self.state

    def step(self, action):
        """
           call step() after make action, now let env response
        :return: next_obs, reward, done, _
        """
        # get real-time obs
        screen = cv2.cvtColor(grab_screen(window_size), cv2.COLOR_RGBA2RGB)
        # print(screen)

        # cv2.imshow("screen_gray", screen_gray)

        # screen_gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)

        # collect blood graph
        # blood_window_gray = cv2.cvtColor(grab_screen(blood_window), cv2.COLOR_BGR2GRAY)

        boss_blood_area = grab_screen(boss_blood_window)
        self_blood_area = grab_screen(self_blood_window)

        next_boss_blood = boss_blood_detector(boss_blood_area)
        next_sekiro_blood = self_blood_detector(self_blood_area)

        print("blood boss: {}, sekiro: {}".format(next_boss_blood,next_sekiro_blood))
        next_state = cv2.resize(screen, (WIDTH, HEIGHT))
        # next_state = next_state[:, :, None]
        # because opencv use h,w
        next_state = np.array(next_state).reshape([WIDTH, HEIGHT, 3])

        reward, done, self.stop, self.emergency_break = judge_blood4reward(self.boss_blood, self.sekiro_blood,
                                                                           next_boss_blood, next_sekiro_blood,
                                                                           self.stop, self.emergency_break)

        # update all var
        self.cur_boss_blood = next_boss_blood
        self.cur_sekiro_blood = next_sekiro_blood
        self.state = next_state
        print("took time {}".format(time.time() - self.last_time))
        self.last_time = time.time()
        self.target_step += 1
        # cv2.destroyAllWindows()
        # print("next_state shape: {}".format(next_state.shape))

        return next_state, reward, done

    def render(self):
        pass
