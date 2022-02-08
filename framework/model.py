import copy
import tensorflow._api.v2.compat.v1 as tf

from _config import scope, train_model_scope
from net import DQN
from net.vgg19 import Vgg19


class Model(object):
    def __init__(self, sess, obs_dim, act_dim):
        self.sess = sess
        self.vgg = Vgg19()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.base_Q_net_scope = 'base_Q_net'

        # so we can dont store truely model here

        with tf.name_scope('inputs'):
            # first, set the input of networks
            self.state_input = tf.placeholder("float", [None, obs_dim.state_w, obs_dim.state_h, obs_dim.state_d])
            self.next_state_input = tf.placeholder(tf.float32, [None, obs_dim.state_w, obs_dim.state_h, obs_dim.state_d])
        # second, create the current_net

    def base_Q_net(self, x_init, reuse=True):
        with tf.variable_scope(self.base_Q_net_scope, reuse=reuse):
            self.vgg.build(x_init)
            real_feature_map = self.vgg.conv4_4_no_activation
            Q = DQN.Q_net(real_feature_map, self.obs_dim, self.act_dim, self.base_Q_net_scope)
            return Q.Q_value

    def target_Q_net(self, x_init, reuse=True):
        with tf.variable_scope(self.target_Q_net_scope, reuse=reuse):
            self.vgg.build(x_init)
            real_feature_map = self.vgg.conv4_4_no_activation
            Q = DQN.Q_net(real_feature_map, self.obs_dim, self.act_dim, self.target_Q_net_scope)
            return Q.Q_value

    def clone(self):
        """
        :return:
        """
        self.target_Q_net_scope = 'target_Q_net'
        # self.build_model()
        return self.target_Q_net

    def __call__(self, x):
        return self.base_Q_net(x)

    def build_model(self):
        self.Q_value = self.base_Q_net(self.state_input, reuse=False)
        self.target_Q_value = self.target_Q_net(self.next_state_input, reuse=False)

    def sync_weights_to(self, target_q_scope):  # at last, solve the parameters replace problem
        # # the parameters of current_net
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.base_Q_net_scope)
        # the parameters of target_net
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.target_Q_net_scope)

        # define the operation that replace the target_net's parameters by current_net's parameters
        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]
        self.sess.run(self.target_replace_op)
