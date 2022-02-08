import tensorflow._api.v2.compat.v1 as tf1

import numpy as np
import time
import sys

# 应该是根据data_man里得到
from settings import VGG_MEAN


class Vgg19:
    def __init__(self, vgg19_npy_path='vgg19_weight/vgg19.npy'):

        if vgg19_npy_path is not None:
            self.data_dict = np.load(vgg19_npy_path, encoding='latin1', allow_pickle=True).item()
            print("npy file loaded ------- ", vgg19_npy_path)
        else:
            self.data_dict = None
            print("npy file load error!")
            sys.exit(1)

    def build(self, rgb, include_fc=False):
        """
        load variable from npy to build the VGG
        input format: bgr image with shape [batch_size, h, w, 3]
        scale: (-1, 1)
        """

        start_time = time.time()
        rgb_scaled = ((rgb + 1) / 2) * 255.0  # [-1, 1] ~ [0, 255]

        # 在通道方向分离三个轴，得到r,g,b
        red, green, blue = tf1.split(axis=3, num_or_size_splits=3, value=rgb_scaled)

        # 消除这批资料风格影响
        bgr = tf1.concat(axis=3, values=[blue - VGG_MEAN[0],
                                         green - VGG_MEAN[1],
                                         red - VGG_MEAN[2]])

        self.conv1_1 = self.conv_layer(bgr, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.conv3_4 = self.conv_layer(self.conv3_3, "conv3_4")
        self.pool3 = self.max_pool(self.conv3_4, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")

        # 用vgg这一层拿来提取feature map
        self.conv4_4_no_activation = self.no_activation_conv_layer(self.conv4_3, "conv4_4")

        self.conv4_4 = self.conv_layer(self.conv4_3, "conv4_4")
        self.pool4 = self.max_pool(self.conv4_4, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        self.conv5_4 = self.conv_layer(self.conv5_3, "conv5_4")
        self.pool5 = self.max_pool(self.conv5_4, 'pool5')

        if include_fc:
            self.fc6 = self.fc_layer(self.pool5, "fc6")
            assert self.fc6.get_shape().as_list()[1:] == [4096]
            self.relu6 = tf1.nn.relu(self.fc6)

            self.fc7 = self.fc_layer(self.relu6, "fc7")
            self.relu7 = tf1.nn.relu(self.fc7)

            self.fc8 = self.fc_layer(self.relu7, "fc8")

            self.prob = tf1.nn.softmax(self.fc8, name="prob")

            self.data_dict = None
        print(("build model finished: %fs" % (time.time() - start_time)))

    def avg_pool(self, bottom, name):
        return tf1.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf1.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf1.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf1.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf1.nn.bias_add(conv, conv_biases)

            relu = tf1.nn.relu(bias)
            return relu

    def no_activation_conv_layer(self, bottom, name):
        with tf1.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf1.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            x = tf1.nn.bias_add(conv, conv_biases)

            return x

    def fc_layer(self, bottom, name):
        with tf1.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf1.reshape(bottom, [-1, dim])

            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf1.nn.bias_add(tf1.matmul(x, weights), biases)

            return fc

    def get_conv_filter(self, name):
        return tf1.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf1.constant(self.data_dict[name][1], name="biases")

    def get_fc_weight(self, name):
        return tf1.constant(self.data_dict[name][0], name="weights")
