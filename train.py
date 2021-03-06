import os.path

import cv2
import numpy as np

from settings import MEMORY_WARMUP_SIZE, LEARN_FREQ, BATCH_SIZE, GAMMA, LEARNING_RATE, MEMORY_SIZE, \
    sample_data_save_root, logger
from strategy.actions import take_action
from framework.environment import Environment
from framework.model import Model
from framework.algorithm import DQN
# from parl.algorithms import DQN
from framework.agent import Agent

from framework.replay_memory import ReplayMemory
import tensorflow._api.v2.compat.v1 as tf1

from utils.count_down import CountDown

# 训练一个episode
from win32_utils.key_mapping import interrupt_game, continue_game


def run_train_episode(agent, env, rpm, online=True):
    total_reward = 0
    obs = env.reset()
    step = 0
    done = 0
    interrupt_state = 0
    reward = 0
    next_obs = None
    while True:
        step += 1
        if online:
            # not one-hot here
            action = agent.sample(obs)  # 采样动作，所有动作都有概率被尝试到
            # print("action: ", action)
            take_action(action)

            next_obs, reward, done = env.step(action)
            print("step {}, reward: {}, done:{}".format(step, reward, done))
            rpm.append((obs, action, reward, next_obs, done))

        # train model
        if (len(rpm) > MEMORY_WARMUP_SIZE) and (step % LEARN_FREQ == 0):
            # interrupt game
            interrupt_state = interrupt_game()
            # s,a,r,s',done
            (batch_obs, batch_action, batch_reward, batch_next_obs,
             batch_done) = rpm.sample(BATCH_SIZE)
            train_loss = agent.learn(batch_obs, batch_action, batch_reward,
                                     batch_next_obs, batch_done)
            print("mse_loss: ", train_loss)
        # maybe offline data not larger than MEMORY_WARMUP_SIZE
        if not online and (len(rpm) > MEMORY_WARMUP_SIZE) == False:
            logger.debug("the offline data len {}/{} too short for train".format(len(rpm), MEMORY_WARMUP_SIZE))
        total_reward += reward
        obs = next_obs
        if done:
            break
        interrupt_state = continue_game(interrupt_state)
    cv2.destroyAllWindows()
    return total_reward


# 评估 agent, 跑 5 个episode，总reward求平均
def run_evaluate_episodes(agent, env, render=False):
    eval_reward = []
    for i in range(5):
        obs = env.reset()
        episode_reward = 0
        while True:
            action = agent.predict(obs[np.newaxis, :, :, :])  # 预测动作，只选最优动作
            obs, reward, done = env.step(action)
            episode_reward += reward
            if render:
                env.render()
            if done:
                break
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)


def main(sampled_data_path=None, online=True):
    gpu_options = tf1.GPUOptions(allow_growth=True)
    with tf1.Session(config=tf1.ConfigProto(allow_soft_placement=True, inter_op_parallelism_threads=8,
                                            intra_op_parallelism_threads=8, gpu_options=gpu_options)) as sess:
        env = Environment.make('Sekiro')

        # a picture
        obs_dim = env.get_observation_space()  # CartPole-v0: (4,)
        act_dim = env.get_action_space()  # env.get_action_space()  # CartPole-v0: 2

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

        print("already now?")
        if online:
            CountDown(3)

        # 先往经验池里存一些数据，避免最开始训练的时候样本丰富度不够
        while len(rpm) < MEMORY_WARMUP_SIZE:
            run_train_episode(agent, env, rpm, online)
            agent.save(save_path)

        max_episode = 2000

        # start train
        episode = 0
        while episode < max_episode:  # 训练max_episode个回合，test部分不计算入episode数量
            # train part
            for i in range(50):
                total_reward = run_train_episode(agent, env, rpm, online)
                episode += 1

            # test part       render=True 查看显示效果
            eval_reward = run_evaluate_episodes(agent, env, render=False)


if __name__ == '__main__':
    filename = 'episode_2022_02_08_22_54-total_step-539.npy'
    sampled_data = os.path.join(sample_data_save_root, filename)

    main(sampled_data, online=False)
