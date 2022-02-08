import tf_slim

from ops.functional import max_pool_2x2, Conv2D, DownSample
import tensorflow._api.v2.compat.v1 as tf

tf.disable_v2_behavior()


class Q_net(object):
    def __init__(self, inputs, obs_dim, act_dim, scope):
        self.scope = scope
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        with tf.variable_scope(scope):
            inputs = DownSample(inputs, 32)
            inputs = tf.nn.relu(Conv2D(inputs, 32))
            inputs = Conv2D(inputs, filters=32, kernel_size=1)
            inputs = tf.nn.max_pool2d(inputs, 3, strides=2, padding='VALID')

            inputs = DownSample(inputs, 32)

            inputs = Conv2D(inputs, filters=64, kernel_size=1)
            inputs = tf.nn.relu(Conv2D(inputs, filters=16))
            inputs = tf.nn.max_pool2d(inputs, 3, strides=2, padding='VALID')

            inputs = tf.nn.relu(Conv2D(inputs, filters=32))
            inputs = tf.nn.max_pool2d(inputs, 3, strides=2, padding='VALID')
            inputs = tf.nn.relu(Conv2D(inputs, filters=32))

            inputs = tf.nn.relu(Conv2D(inputs, filters=16))

            inputs = tf_slim.flatten(inputs)
            inputs = tf_slim.fully_connected(inputs, self.act_dim)

            inputs = tf.nn.dropout(inputs, rate=0.5)
            self.Q_value = inputs
