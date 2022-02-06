import math

from settings import logger
from utils.blood_detection import self_blood_detector, boss_blood_detector
import cv2

sample_bleeding_freq = 1


class RewardJudger:
    def __init__(self, sample_freq):
        self.sample_freq = sample_freq
        self.eps = 1e-5
        self.last_sampled = {}
        self.sample_period_counter = 0

    def __call__(self, *args, **kwargs):
        return self.judge(*args, **kwargs)

    def judge(self, boss_blood,
              next_boss_blood,
              self_blood,
              next_self_blood):
        # get action reward
        # emergence_break is used to break down training
        # 用于防止出现意外紧急停止训练防止错误训练数据扰乱神经网络
        reward = -0.01
        done = 0
        stop = 0
        emergence_break = 0
        # print(self_blood_detector.blood_capacity)
        # print(boss_blood_detector.blood_capacity)
        # if self.last_sampled == {}:
        #     self.last_sampled['boss_blood'] = 0
        #     self.last_sampled['self_blood'] = 0

        if next_self_blood <= 1:  # self dead
            # if emergence_break < 2:
            #     reward = -10
            #     done = 1
            #     stop = 0
            #     emergence_break += 1
            #     return reward, done, stop, emergence_break
            # else:
            print("self dead")
            reward -= 20
            done = 1
        if next_boss_blood <= 2 or next_boss_blood < boss_blood - 30:  # boss dead
            print("boss dead")
            reward += 30
            done = 1
            stop = 0

        # print("next boss blood: ", next_boss_blood)
        self_blood_reward = 0
        boss_blood_reward = 0
        # print(next_self_blood - self_blood)
        # print(next_boss_blood - boss_blood)
        if done != 1 and self.sample_period_counter == 0:
            # sample a period, because each attack a little
            if next_self_blood < self_blood - 2:
                print("self bleeding... {}/{}".format(next_self_blood,self_blood))
                self_blood_reward = -3

            if next_boss_blood - boss_blood < -2:
                print("boss bleeding... {}/{}".format(next_boss_blood, boss_blood))
                boss_blood_reward = 5.5

            # print("[boss blood] next/store: {}/{}".format(next_boss_blood, self.last_sampled['boss_blood']))

            # self.last_sampled['self_blood'] = next_self_blood
            # self.last_sampled['boss_blood'] = next_boss_blood

        self.sample_period_counter = (self.sample_period_counter + 1) % self.sample_freq

        # print("boss blood reward: ", boss_blood_reward)
        # print("self blood reward: ", self_blood_reward)
        reward += self_blood_reward
        reward += boss_blood_reward
        logger.info("self: {}/{}, boss: {}/{}, reward: {}, done: {}".format(self_blood,
                                                                            next_self_blood,
                                                                            boss_blood,
                                                                            next_boss_blood,
                                                                            reward,
                                                                            done))
        return reward, done, stop, emergence_break


judge_blood4reward = RewardJudger(sample_bleeding_freq)
