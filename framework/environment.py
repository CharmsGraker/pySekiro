import time

import cv2
import matplotlib.pyplot as plt
import numpy as np

from unit_test.fast_test import boss_blood_window, self_blood_window
from settings import window_size, HEIGHT, WIDTH, pause_key
from strategy.judge import judge_blood4reward
from utils.blood_detection import boss_blood_detector, self_blood_detector
from win32_utils.getkeys import key_check
from win32_utils.grabscreen import grab_screen

from settings import logger


class Enviroment:
    def __init__(self):
        pass

    @staticmethod
    def make(env_name, *arg, **kwargs):
        if env_name == 'Sekiro':
            return SekiroEnv(*arg, **kwargs)


class Config:
    def __init__(self, dict):
        self.__dict__.update(dict)


class PauseException(Exception):
    def __init__(self):
        super(PauseException, self).__init__()

    pass


class SekiroEnv:

    def __init__(self, disable_resize=False):
        self.disable_resize = disable_resize
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
        if self.disable_resize:
            self.HEIGHT = window_size[2] - window_size[0] + 1
            self.WIDTH = window_size[3] - window_size[1] + 1
        else:
            self.WIDTH = WIDTH
            self.HEIGHT = HEIGHT

        self.state = cv2.cvtColor(grab_screen(window_size), cv2.COLOR_RGBA2RGB)
        # screen_gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)

        # collect station gray graph
        # blood_window_gray = cv2.cvtColor(grab_screen(blood_window), cv2.COLOR_BGR2GRAY)

        # collect blood gray graph for count self and boss blood
        if not self.disable_resize:
            self.state = cv2.resize(self.state, (self.WIDTH, self.HEIGHT))
            self.state = np.array(self.state).reshape([self.WIDTH, self.HEIGHT, 3])  # [0]

        plt.imshow(self.state)
        plt.show()
        logger.debug(self.state.shape)
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

    def step(self, action, log_info=False):
        """
           call step() after make action, now let env response
        :return: next_obs, reward, done, _
        """
        paused = False
        while True:
            if pause_key in key_check():
                if paused == True:
                    paused = False
                else:
                    paused = True

            try:
                if (paused == False):
                    break
                else:
                    raise PauseException()
            except PauseException as e:
                print("paused")
                time.sleep(0.5)

        # get real-time obs
        next_state = cv2.cvtColor(grab_screen(window_size), cv2.COLOR_RGBA2RGB)
        # print(screen)

        # cv2.imshow("screen_gray", screen_gray)

        # screen_gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)

        # collect blood graph
        # blood_window_gray = cv2.cvtColor(grab_screen(blood_window), cv2.COLOR_BGR2GRAY)

        boss_blood_area = grab_screen(boss_blood_window)
        self_blood_area = grab_screen(self_blood_window)

        next_boss_blood = boss_blood_detector(boss_blood_area)
        next_sekiro_blood = self_blood_detector(self_blood_area)

        # print("blood boss: {}, sekiro: {}".format(next_boss_blood, next_sekiro_blood))
        if not self.disable_resize:
            next_state = cv2.resize(next_state, (self.WIDTH, self.HEIGHT))
            next_state = np.array(next_state).reshape([self.WIDTH, self.HEIGHT, 3])
        # next_state = next_state[:, :, None]
        # because opencv use h,w


        reward, done, self.stop, self.emergency_break = judge_blood4reward(self.boss_blood, next_boss_blood,
                                                                           self.sekiro_blood, next_sekiro_blood)

        # update all var
        self.boss_blood = next_boss_blood
        self.sekiro_blood = next_sekiro_blood
        self.state = next_state
        logger.debug("took time {}".format(time.time() - self.last_time))
        self.last_time = time.time()
        self.target_step += 1
        # cv2.destroyAllWindows()
        # print("next_state shape: {}".format(next_state.shape))
        return next_state, reward, done

    def render(self):
        pass
