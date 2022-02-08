import tf_slim

from ops.functional import max_pool_2x2, Conv2D, DownSample, layer_norm, Conv2DNormLReLU, Separable_conv2d
import tensorflow._api.v2.compat.v1 as tf

tf.disable_v2_behavior()


class Q_net(object):
    def __init__(self, inputs, obs_dim, act_dim, scope):
        self.scope = scope
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        with tf.variable_scope(scope):
            inputs = Conv2DNormLReLU(inputs, 64)
            inputs = Conv2DNormLReLU(inputs, 64)
            inputs = Separable_conv2d(inputs, 128, strides=2) + DownSample(inputs, 128)

            inputs = Conv2DNormLReLU(inputs, 128)
            inputs = Separable_conv2d(inputs, 128)
            inputs = Separable_conv2d(inputs, 256, strides=2) + DownSample(inputs, 256)

            inputs = Conv2DNormLReLU(inputs, 32)
            inputs = Conv2DNormLReLU(inputs, 16)
            inputs = tf_slim.flatten(inputs)
            inputs = tf_slim.fully_connected(inputs, self.act_dim)

            inputs = tf.nn.dropout(inputs, rate=0.5)
            self.Q_value = inputs
