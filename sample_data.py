import os
import time

import cv2
import numpy as np
import win32api
import win32con

from settings import MEMORY_WARMUP_SIZE, logger, LEARN_FREQ, BATCH_SIZE, GAMMA, LEARNING_RATE, MEMORY_SIZE, pause_key, \
    sample_data_save_root
from strategy.actions import take_action
from framework.environment import Enviroment

from utils.count_down import CountDown

# 训练一个episode
from win32_utils.check_key_utils import state_lButton, state_rButton, state_space, state_W, state_S, state_A, state_D, \
    state_capsLock, state_leftAlt, state_R


def sample_train_episode(behavior, env, buffer):
    total_reward = 0
    obs = env.reset()
    step = 0
    while True:
        step += 1
        # not one-hot here
        action = behavior.sample(obs)  # is a str liked action,such as attack, jump ...
        # print("action: ", action)
        take_action(action)

        next_obs, reward, done = env.step(action)
        logger.debug("step {}, reward: {}, done:{}".format(step, reward, done))
        buffer.append([obs, action, reward, next_obs, done])

        total_reward += reward
        obs = next_obs
        if done:
            break
    assert len(buffer) > 0

    np.save(os.path.join(sample_data_save_root,
                         'episode_{}-total_step-{}.npy'
                         .format(time.strftime("%Y_%m_%d_%H_%M",
                                               time.localtime()),
                                 step)), np.array(buffer))

    cv2.destroyAllWindows()
    return total_reward


class Behavior:
    def __init__(self):
        pass

    def sample(self, obs):
        actions = []
        if state_lButton.checkAndIsPress():
            actions.append('attack')
        elif state_rButton.checkAndIsPress():
            actions.append('defense')
        if state_space.checkAndIsPress():
            actions.append("jump")
        elif state_leftAlt.checkAndIsPress():
            actions.append("roll_up")
        if state_W.checkAndIsPress():
            actions.append('forward')
        if state_S.checkAndIsPress():
            actions.append("backward")
        if state_A.checkAndIsPress():
            actions.append("left")
        if state_D.checkAndIsPress():
            actions.append('right')
        if state_R.checkAndIsPress():
            actions.append('recover')

        output = "|".join(actions)
        return output


def main():
    env = Enviroment.make('Sekiro', disable_resize=True)
    # a picture
    obs_dim = env.get_observation_space()  # CartPole-v0: (4,)
    act_dim = 5  # env.get_action_space()  # CartPole-v0: 2

    behavior = Behavior()
    buffer = []
    logger.info("sample data...")
    logger.info("already now?")

    ready = 0
    CountDown(3)
    # 先往经验池里存一些数据，避免最开始训练的时候样本丰富度不够
    while len(buffer) < MEMORY_WARMUP_SIZE:
        while True:
            logger.info("waiting ready key...")
            time.sleep(0.1)
            if state_capsLock.checkAndIsPress():
                ready ^= 1
            if ready == 1:
                break
        ready = 0
        sample_train_episode(behavior, env, buffer)
        logger.info("sampled one episode done.")


if __name__ == '__main__':
    main()
