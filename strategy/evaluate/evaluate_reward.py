from settings import logger
from strategy.evaluate.reward_dict import rewardOfBossEndurance

sample_bleeding_freq = 1


class RewardEvaluator:
    def __init__(self, sample_freq, rewardOfBossEndurance):
        """

        :param sample_freq:
        :param rewardOfBossEndurance: a functor
        """
        self.sample_freq = sample_freq
        self.eps = 1e-5
        self.last_sampled = {}
        self.sample_period_counter = 0
        self.rewardOfBossEndurance = rewardOfBossEndurance

    def __call__(self, *args, **kwargs):
        return self.judge(*args, **kwargs)

    def judge(self,
              boss_blood,
              next_boss_blood,
              self_blood,
              next_self_blood,
              next_boss_endurance):
        # get action reward
        # emergence_break is used to break down training
        # 用于防止出现意外紧急停止训练防止错误训练数据扰乱神经网络
        reward = -0.01
        done = 0
        emergence_break = 0
        self_blood_reward = 0
        boss_blood_reward = 0

        # to confirm current status
        if next_self_blood <= 1:  # self dead or low blood state, danger
            logger.debug("self dead")
            reward -= 50
            done = 1

        elif next_boss_blood <= 2:  # boss dead
            logger.debug("boss dead")
            reward += 30
            done = 1

            if next_boss_endurance == 'shinobi':
                reward += self.rewardOfBossEndurance(next_boss_endurance)
        else:
            if next_self_blood - self_blood < - 2:
                print("self bleeding... {}/{}".format(next_self_blood, self_blood))
                self_blood_reward = -7
            elif next_boss_blood < 125:
                self_blood_reward = -3

            if next_boss_blood - boss_blood < -2:
                print("boss bleeding... {}/{}".format(next_boss_blood, boss_blood))
                boss_blood_reward = 5.5

            # the shinobi status only calc when done = 1,
            # so avoid forget this reward, will calc at boss dead
            if next_boss_endurance != 'shinobi':
                reward += self.rewardOfBossEndurance(next_boss_endurance)

        # sample a period, because each attack a little

        # print("[boss blood] next/store: {}/{}".format(next_boss_blood, self.last_sampled['boss_blood']))
        # self.sample_period_counter = (self.sample_period_counter + 1) % self.sample_freq

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
        return reward, done


judge_blood4reward = RewardEvaluator(sample_bleeding_freq, rewardOfBossEndurance)
