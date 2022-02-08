# the function that give the weight initial value
import tensorflow._api.v2.compat.v1 as tf
import tf_slim as slim
tf.disable_v2_behavior()

def lrelu(x, alpha=0.2, scope=None):
    return tf.nn.leaky_relu(x, alpha)


def Separable_conv2d(inputs, filters, kernel_size=3, strides=1, padding='VALID', Use_bias=None):
    """
        关于tf.pad，https://blog.csdn.net/yy_diego/article/details/81563160
        第二个参数表示的是填充行数。
        看样子input的特征有是三维。上下、左右、前后，其中第0维是样本方向
    """
    if kernel_size == 3 and strides == 1:
        inputs = tf.pad(inputs, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="REFLECT")

    if strides == 2:
        inputs = tf.pad(inputs, [[0, 0], [0, 1], [0, 1], [0, 0]], mode="REFLECT")

    # 深度可分离卷积。貌似是对每个通道运用不同的卷积核。故输出深度=通道数
    # https://blog.csdn.net/qq_33278989/article/details/80265657
    # return tf1.layers.separable_conv2d(
    #     inputs,
    #     num_outputs=filters,
    #     kernel_size=kernel_size,
    #     depth_multiplier=1,
    #     stride=strides,
    #     biases_initializer=Use_bias,
    #     normalizer_fn=InstanceNormalization,
    #     activation_fn=lrelu,
    #     padding=padding)
    return slim.layers.separable_conv2d(
        inputs,
        num_outputs=filters,
        kernel_size=kernel_size,
        depth_multiplier=1,
        stride=strides,
        biases_initializer=Use_bias,
        normalizer_fn=slim.instance_norm,
        activation_fn=lrelu,
        padding=padding)


def DownSample(inputs, filters=256, kernel_size=3):
    '''
        An alternative to transposed convolution where we first resize, then convolve.
        See http://distill.pub/2016/deconv-checkerboard/
        For some reason the shape needs to be statically known for gradient propagation
        through tf.image.resize_images, but we only know that for fixed image size, so we
        plumb through a "training" argument
        '''
    # if you use tf.shape, you may lose the tensor shape
    # but actually we can infer that through a static shape
    # input shape: sample * H * W
    input_shape = inputs.get_shape().as_list()
    new_H, new_W = input_shape[1] // 2, input_shape[2] // 2
    inputs = tf.image.resize_images(inputs, [new_H, new_W])

    return Separable_conv2d(filters=filters, kernel_size=kernel_size, inputs=inputs)


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
