import time
import cv2
import matplotlib.pyplot as plt
import numpy as np

from settings import window_size, HEIGHT, WIDTH, pause_key, obs_dim
from strategy.actions import action_table
from strategy.evaluate.evaluate_reward import judge_blood4reward
from unit_test.fast_test import blood_window, boss_blood_offset_window, self_blood_offset_window, \
    endurance_offset_window
from utils.detection.blood_detection import boss_blood_detector, self_blood_detector, clip_screen
from utils.detection.endurance_detection import boss_endurance_detector
from win32_utils.getkeys import key_check
from win32_utils.grabscreen import grab_screen

from settings import logger


class Environment:
    def __init__(self):
        pass

    @staticmethod
    def make(env_name, *arg, **kwargs):
        if env_name == 'Sekiro':
            return SekiroEnv(*arg, **kwargs)


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
        self.use_history = False
        self.history_data = None
        self.cursor = -1

    def get_observation_space(self):
        """
        use RGB for feature map inputs
        :return:
        """
        # give the original shape before input to the net
        return obs_dim

    def get_action_space(self):
        return len(action_table)

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

        # count init blood
        self.target_step = 0
        # used to update target Q network
        self.done = 0
        self.total_reward = 0

        self.last_time = time.time()

        # collect state gray graph
        self.state = self.flush_next_state()

        # collect blood gray graph for count self and boss blood
        if not self.disable_resize:
            # change graph to WIDTH * HEIGHT for station input
            self.state = cv2.resize(self.state, (self.WIDTH, self.HEIGHT))
            self.state = np.array(self.state).reshape([self.WIDTH, self.HEIGHT, 3])  # [0]

        logger.debug(self.state.shape)
        # endurance_area = grab_screen(endurance_window)

        # after cvtColor, W,H image will convert to H,W image
        blood_screen = cv2.cvtColor(grab_screen(blood_window), cv2.COLOR_RGBA2RGB)
        boss_blood_area = clip_screen(blood_screen, boss_blood_offset_window)
        self_blood_area = clip_screen(blood_screen, self_blood_offset_window)

        self.boss_blood = boss_blood_detector(boss_blood_area)
        self.sekiro_blood = self_blood_detector(self_blood_area)

        return self.state

    def step(self, action, log_info=False):
        """
           call step() after make action, now let env response
        :return: next_obs, reward, done, _
        """
        paused = False
        while True:
            if pause_key in key_check():
                if paused:
                    paused = False
                else:
                    paused = True
            try:
                if not paused:
                    break
                else:
                    raise PauseException()
            except PauseException as e:
                print("paused")
                time.sleep(0.5)

        # get real-time obs
        next_state = self.flush_next_state()

        if not self.disable_resize:
            next_state = cv2.resize(next_state, (self.WIDTH, self.HEIGHT))
            next_state = np.array(next_state).reshape([self.WIDTH, self.HEIGHT, 3])

        reward, done = self.flush_reward_done(next_state)
        # because opencv use h,w

        logger.debug("took time {}".format(time.time() - self.last_time))
        self.last_time = time.time()
        self.target_step += 1
        return next_state, reward, done

    def flush_next_state(self, initial=False):
        next_state = cv2.cvtColor(grab_screen(window_size), cv2.COLOR_RGBA2RGB)
        self.endurance_area = clip_screen(next_state, endurance_offset_window)

        return next_state

    def flush_reward_done(self, next_state):
        blood_screen = cv2.cvtColor(grab_screen(blood_window), cv2.COLOR_RGBA2RGB)
        boss_blood_area = clip_screen(blood_screen, boss_blood_offset_window)
        self_blood_area = clip_screen(blood_screen, self_blood_offset_window)

        next_boss_blood = boss_blood_detector(boss_blood_area)
        next_sekiro_blood = self_blood_detector(self_blood_area)
        next_boss_endurance = boss_endurance_detector(self.endurance_area)
        reward, done = judge_blood4reward(self.boss_blood,
                                          next_boss_blood,
                                          self.sekiro_blood,
                                          next_sekiro_blood,
                                          next_boss_endurance)
        # update all var
        self.boss_blood = next_boss_blood
        self.sekiro_blood = next_sekiro_blood
        self.state = next_state
        return reward, done

    def render(self):
        plt.imshow(self.state)
        plt.show()
