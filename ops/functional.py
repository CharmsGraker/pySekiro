# the function that give the weight initial value
import tensorflow._api.v2.compat.v1 as tf
import tf_slim as slim

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# the function that give the bias initial value
def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)



def Conv2D(inputs, filters, kernel_size=3, strides=1, padding='VALID', Use_bias=None):
    if kernel_size == 3:
        inputs = tf.pad(inputs, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="REFLECT")
    # return tf1.nn.conv2d(
    #     inputs,
    #     num_outputs=filters,
    #     #kernel_size=kernel_size,
    #     stride=strides,
    #     biases_initializer=Use_bias,
    #     normalizer_fn=None,
    #     activation_fn=None,
    #     padding=padding)
    return slim.layers.conv2d(inputs,
                              num_outputs=filters,
                              kernel_size=kernel_size,
                              stride=strides,
                              biases_initializer=Use_bias,
                              normalizer_fn=None,
                              activation_fn=None,
                              padding=padding)

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
