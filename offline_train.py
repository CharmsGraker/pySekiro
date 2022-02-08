import argparse
import os.path

import cv2
import numpy as np

from settings import MEMORY_WARMUP_SIZE, LEARN_FREQ, BATCH_SIZE, GAMMA, LEARNING_RATE, MEMORY_SIZE, \
    sample_data_save_root, logger, obs_dim, act_dim
from framework.model import Model
from framework.algorithm import DQN
from framework.agent import Agent

from framework.replay_memory import ReplayMemory
import tensorflow._api.v2.compat.v1 as tf1


# 训练一个episode
def run_train_episode_offline(agent, simulate_env,rpm):
    total_reward = 0
    step = 0
    done = 0
    interrupt_state = 0
    reward = 0
    next_obs = None
    while True:
        step += 1

        action,next_obs, reward, done = simulate_env.step(step)

        # train model
        if (len(rpm) > MEMORY_WARMUP_SIZE) and (step % LEARN_FREQ == 0):
            # interrupt game
            # s,a,r,s',done
            (batch_obs, batch_action, batch_reward, batch_next_obs,
             batch_done) = rpm.sample(BATCH_SIZE)
            train_loss = agent.learn(batch_obs, batch_action, batch_reward,
                                     batch_next_obs, batch_done)
            print("mse_loss: ", train_loss)

        # maybe offline data not larger than MEMORY_WARMUP_SIZE
        if (len(rpm) > MEMORY_WARMUP_SIZE) == False:
            logger.debug("the offline data len {}/{} too short for train".format(len(rpm), MEMORY_WARMUP_SIZE))
            raise Exception("too short data")
        total_reward += reward
        obs = next_obs
        if done:
            break
    cv2.destroyAllWindows()
    return total_reward


def offline_main(obs_dim, act_dim, sampled_data_path=None):
    gpu_options = tf1.GPUOptions(allow_growth=True)
    with tf1.Session(config=tf1.ConfigProto(allow_soft_placement=True, inter_op_parallelism_threads=8,
                                            intra_op_parallelism_threads=8, gpu_options=gpu_options)) as sess:

        rpm = ReplayMemory(MEMORY_SIZE)  # DQN的经验回放池
        if sampled_data_path:
            rpm.loadFrom(sampled_data_path)

        # 根据parl框架构建agent
        model = Model(sess=sess, obs_dim=obs_dim, act_dim=act_dim)
        algorithm = DQN(sess, model, gamma=GAMMA, lr=LEARNING_RATE)
        agent = Agent(
            algorithm,
            act_dim=act_dim,
            e_greed=0.1,  # 有一定概率随机选取动作，探索
            e_greed_decrement=1e-6)  # 随着训练逐步收敛，探索的程度慢慢降低

        # 加载模型
        # save_path = './dqn_model.ckpt'

        # 训练结束，保存模型
        save_path = './dqn_model.ckpt'
        agent.restore(save_path)

        # 先往经验池里存一些数据，避免最开始训练的时候样本丰富度不够
        while len(rpm) < MEMORY_WARMUP_SIZE:
            run_train_episode_offline(agent, simulate_env,rpm)
            agent.save(save_path)

        max_episode = 2000

        # # start train
        # episode = 0
        # while episode < max_episode:  # 训练max_episode个回合，test部分不计算入episode数量
        #     # train part
        #     for i in range(50):
        #         total_reward = run_train_episode_offline(agent, rpm)
        #         episode += 1
        #
        #     # test part       render=True 查看显示效果
        #     # eval_reward = run_evaluate_episodes(agent, render=False)


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('-file', default=None, type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arg()
    file = args.file
    if file is None:
        filename = 'episode_2022_02_08_22_54-total_step-539.npy'
        file = os.path.join(sample_data_save_root, filename)

    offline_main(obs_dim, act_dim, file)
