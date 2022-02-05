import os

import tensorflow._api.v2.compat.v1 as tf1

tf1.disable_v2_behavior()


class Algorithm:
    def __init__(self, sess):
        self.sess = sess

        self.log_dir = 'logs'
        self.checkpoint_dir = 'checkpoint'

        self.writer = tf1.summary.FileWriter(self.log_dir + '/' + self.model_dir, self.sess.graph)

    def build_saver(self):
        self.sess.run(tf1.global_variables_initializer())
        self.saver = tf1.train.Saver()

    def predict(self, obs):
        pass

    def learn(self, obs, action, reward, next_obs, terminal):
        pass

    def load(self, checkpoint_dir):
        """
        可能训练非常缓慢，或者说当前条件不允许一次性训练完。这个函数使得可以从上次check point(断点）处继续训练
        :param checkpoint_dir: 断点保存路径
        :return: flag:True/False,counter:last epoch
        """
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf1.train.get_checkpoint_state(checkpoint_dir)  # checkpoint file information

        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)  # first line
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(ckpt_name.split('-')[-1])
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    @property
    def model_dir(self):
        return "{}_{}".format(self.model_name, self.dataset_name)

    @property
    def model_name(self):
        return 'DQN_Algorithm'

    @property
    def dataset_name(self):
        return 'unknown'


# tf 的思想是用tf变量占位，构建图。在运行时在指定dict

class DQN(Algorithm):
    def __init__(self, sess, model, gamma=None, lr=None):
        """ DQN algorithm

        Args:
            model (parl.Model): 定义Q函数的前向网络结构
            gamma (float): reward的衰减因子
            lr (float): learning_rate，学习率.
        """
        # checks
        assert isinstance(gamma, float)
        assert isinstance(lr, float)

        self.gamma = gamma
        self.lr = lr

        super(DQN, self).__init__(sess)

        self.model = model
        # callup the clone() will trigger build graph
        self.target_model = model.clone()
        # will sync later

        # self.target = tf1.placeholder(tf1.float32, [BATCH_SIZE, 1], name='target')
        # self.pred_value = tf1.placeholder(tf1.float32, [BATCH_SIZE, 1], name='pred_value')

        # self.M_loss = tf1.summary.scalar('target_mse_loss', self.mse_loss)

    def predict(self, obs):
        """ 使用self.model的网络来获取 [Q(s,a1),Q(s,a2),...]
        """
        return self.model(obs)

    def build_graph(self):
        # 获取Q预测值
        self.model.build_model()
        self.obs = self.model.state_input
        self.next_obs = self.model.next_state_input

        with tf1.variable_scope("input"):
            self.action = tf1.placeholder(tf1.int32, [None, 1], name='action_raw_input')
            self.reward = tf1.placeholder(tf1.float32, [None, 1], name='reward_input')
            self.terminal = tf1.placeholder(tf1.float32, [None, 1], name='terminal_input')


        pred_values = self.model(self.obs)
        # print(pred_values.eval())
        action_dim = pred_values.shape[-1]
        action = tf1.squeeze(self.action, axis=-1)

        # 将action转one hot向量，比如：3 => [0,0,0,1,0]
        action_onehot = tf1.one_hot(
            action, depth=action_dim)
        # 下面一行是逐元素相乘，拿到action对应的 Q(s,a)
        # 比如：pred_value = [[2.3, 5.7, 1.2, 3.9, 1.4]], action_onehot = [[0,0,0,1,0]]
        #  ==> pred_value = [[3.9]]
        with tf1.variable_scope("target"):
            self.target_predict = tf1.placeholder(tf1.float32, [None, action_dim], name="predict")

        # shape: batch,1
        pred_value = tf1.reduce_sum(tf1.multiply(pred_values, action_onehot), axis=1, keepdims=True)

        # 从target_model中获取 max Q' 的值，用于计算target_Q
        # with paddle.no_grad():
        max_v = tf1.reduce_max(self.target_predict, axis=1, keepdims=True)
        target = self.reward + (1 - self.terminal) * self.gamma * max_v

        self.mse_loss = tf1.losses.mean_squared_error(pred_value, target)

        with tf1.name_scope('train_loss'):
            # use the loss to optimize the network
            self.optimizer = tf1.train.AdamOptimizer(learning_rate=self.lr).minimize(self.mse_loss)  # 使用Adam优化器

        self.build_saver()

    def learn(self, obs, action, reward, next_obs, terminal):
        """ 使用DQN算法更新self.model的value网络
        """
        # need eval next_obs first, then we can get a ndarray instead of tensor,
        # avoid to update target network
        target_predict = self.model.target_Q_value.eval(feed_dict={self.next_obs: next_obs})
        logit, _ = self.sess.run([self.mse_loss, self.optimizer], feed_dict={
            self.obs: obs,
            self.action: action,
            self.reward: reward,
            self.terminal: terminal,
            self.target_predict: target_predict
        })
        # 计算 Q(s,a) 与 target_Q的均方差，得到loss
        return logit

    def sync_target(self):
        """ 把 self.model 的模型参数值同步到 self.target_model
        """
        self.model.sync_weights_to(self.target_model)

    @property
    def dataset_name(self):
        return "Sekiro"

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(self.model_name,
                                    self.dataset_name,
                                    self.gamma,
                                    self.lr)
