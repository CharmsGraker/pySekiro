import os.path

import numpy as np

import tensorflow._api.v2.compat.v1 as tf1

from settings import WIDTH, HEIGHT

tf1.disable_v2_behavior()


class GrakerAgent(object):
    def __init__(self, algorithm, act_dim):
        self.alg = algorithm
        self.act_dim = act_dim

        self.build_program()

    def build_program(self):
        # self.obs = tf1.placeholder(tf1.float32, [None, WIDTH, HEIGHT, 1], name='obs')
        # self.act = tf1.placeholder(tf1.int32, [self.act_dim, 1], name='act')
        # self.reward = tf1.Variable(0.0, name='reward')
        # self.next_obs = tf1.placeholder(tf1.float32, [None, WIDTH, HEIGHT, 1], name='next_obs')
        # self.terminal = tf1.Variable(0.0, name='terminal')
        # self.loss = self.alg._build_graph(feed=(self.obs, self.act, self.reward, self.next_obs, self.terminal))
        self.alg.build_graph()

    def learn(self, obs, act, reward, next_obs, terminal):
        pass

    def save(self, checkpoint_dir):
        checkpoint_dir = os.path.splitext(checkpoint_dir)[0]
        return self.alg.save(checkpoint_dir, self.global_step)

    def restore(self, path):
        could_load, checkpoint_counter = self.alg.load(path)

        if could_load:
            self.global_step = checkpoint_counter + 1

            print(" [*] Load SUCCESS")
        else:
            self.global_step = 0

            print(" [!] Load failed...")


class Agent(GrakerAgent):
    def __init__(self, algorithm, act_dim, e_greed=0.1, e_greed_decrement=1e-6):
        super(Agent, self).__init__(algorithm, act_dim)
        assert isinstance(act_dim, int)

        self.sess = algorithm.sess

        self.global_step = 0
        self.update_target_steps = 200  # 每隔200个training steps再把model的参数复制到target_model中

        self.e_greed = e_greed  # 有一定概率随机选取动作，探索
        self.e_greed_decrement = e_greed_decrement  # 随着训练逐步收敛，探索的程度慢慢降低

    def sample(self, obs):
        """ 根据观测值 obs 采样（带探索）一个动作
            must be one observation!!!!!
        """
        sample = np.random.random()  # 产生0~1之间的小数
        if sample < self.e_greed:
            act = np.random.randint(self.act_dim)  # 探索：每个动作都有概率被选择
        else:
            act = self.predict(obs[np.newaxis, :, :, :]).eval()  # 选择最优动作
        # decay
        self.e_greed = max(
            0.01, self.e_greed - self.e_greed_decrement)  # 随着训练逐步收敛，探索的程度慢慢降低
        return act

    def predict(self, obs):
        """ 根据观测值 obs 选择最优动作
        """
        obs = tf1.convert_to_tensor(obs, dtype='float32')
        pred_q = self.alg.predict(obs)
        act = tf1.argmax(pred_q, axis=1)[0]  # pred_q.argmax().numpy()[0]  # 选择Q最大的下标，即对应的动作
        return act

    def learn(self, obs, act, reward, next_obs, terminal):
        """ 根据训练数据更新一次模型参数
        """
        if self.global_step % self.update_target_steps == 0:
            self.alg.sync_target()
        self.global_step += 1

        act = np.expand_dims(act, axis=-1)
        reward = np.expand_dims(reward, axis=-1)
        terminal = np.expand_dims(terminal, axis=-1)

        # obs = tf1.convert_to_tensor(obs, dtype='float32')
        # act = tf1.convert_to_tensor(act, dtype='int32')
        # reward = tf1.convert_to_tensor(reward, dtype='float32')
        # next_obs = tf1.convert_to_tensor(next_obs, dtype='float32')
        # terminal = tf1.convert_to_tensor(terminal, dtype='float32')
        logit = self.alg.learn(obs, act, reward, next_obs, terminal)

        # v_loss, _ = self.sess.run([self.mse_loss, self.optimizer], feed_dict={
        #     self.pred_value: pred_value,
        #     self.target: target
        # })

        # 训练一次网络
        return logit  # loss.numpy()[0]
