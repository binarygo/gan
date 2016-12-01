import math
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python import layers as tf_layers


def _normalize_shape(shape):
    if isinstance(shape, int):
        return [shape, shape]
    assert len(shape) == 2
    return shape
    

def get_conv2d_output_shape(input_shape, kernel_shape, stride_shape, padding):
    input_shape = _normalize_shape(input_shape)
    kernel_shape = _normalize_shape(kernel_shape)
    stride_shape = _normalize_shape(stride_shape)
    if padding == "SAME":
        def get_output_x(input_x, stride_x):
            return int(math.ceil(input_x / float(stride_x)))
        return [
            get_output_x(input_shape[i], stride_shape[i])
            for i in range(2)
        ]
    elif padding == "VALID":
        def get_output_x(input_x, kernel_x, stride_x):
            return int(math.floor((input_x - kernel_x) / float(stride_x))) + 1
        return [
            get_output_x(input_shape[i], kernel_shape[i], stride_shape[i])
            for i in range(2)
        ]
    else:
        raise ValueError("padding must be either VALID or SAME:"
                         " {}".format(padding))


def get_conv2d_input_shape(output_shape, kernel_shape, stride_shape, padding):
    output_shape = _normalize_shape(output_shape)
    kernel_shape = _normalize_shape(kernel_shape)
    stride_shape = _normalize_shape(stride_shape)
    if padding == "SAME":
        def get_input_x(output_x, stride_x):
            return output_x * stride_x
        return [
            get_input_x(output_shape[i], stride_shape[i])
            for i in range(2)
        ]
    elif padding == "VALID":
        def get_input_x(output_x, kernel_x, stride_x):
            return (output_x - 1) * stride_x + kernel_x
        return [
            get_input_x(output_shape[i], kernel_shape[i], stride_shape[i])
            for i in range(2)
        ]
    else:
        raise ValueError("padding must be either VALID or SAME:"
                         " {}".format(padding))


def conv2d(input, kernel_shape, output_channels, stride_shape, padding,
           label=""):
    """Makes a 2D convolution layer.

    Args:
      input: a tensor of shape [batch, in_h, in_w, in_channels].
      kernel_shape: a list of [kernel_h, kernel_w].
      output_channels: number of output channels.
      stride_shape: a list of [stride_h, stride_w].
      padding: padding algorithm, can be "VALID" or "SAME".
      label: string prefix of variable name.

    Returns:
      A convolution layer.
    """
    batch, in_h, in_w, in_channels = input.get_shape().as_list()
    k_h, k_w = _normalize_shape(kernel_shape)
    s_h, s_w = _normalize_shape(stride_shape)
    w = tf.get_variable(label + "w",
                        shape=[k_h, k_w, in_channels, output_channels],
                        dtype=tf.float32)
    b = tf.get_variable(label + "b", shape=[output_channels], dtype=tf.float32)
    conv = tf.nn.conv2d(input, filter=w, strides=[1, s_h, s_w, 1],
                        padding=padding)
    return tf.nn.bias_add(conv, b)


def deconv2d(input, kernel_shape, output_channels, stride_shape, padding,
             label=""):
    """Makes a 2D deconvolution layer.

    Args:
      input: a tensor of shape [batch, in_h, in_w, in_channels].
      kernel_shape: a list of [kernel_h, kernel_w].
      output_channels: number of output channels.
      stride_shape: a list of [stride_h, stride_w].
      padding: padding algorithm, can be "VALID" or "SAME".
      label: string prefix of variable name.

    Returns:
      A deconvolution layer.
    """
    batch, in_h, in_w, in_channels = input.get_shape().as_list()
    k_h, k_w = _normalize_shape(kernel_shape)
    s_h, s_w = _normalize_shape(stride_shape)
    w = tf.get_variable(label + "w",
                        shape=[k_h, k_w, output_channels, in_channels],
                        dtype=tf.float32)
    b = tf.get_variable(label + "b", shape=[output_channels], dtype=tf.float32)
    out_h, out_w = get_conv2d_input_shape(
        [in_h, in_w], [k_h, k_w], [s_h, s_w], padding)
    deconv = tf.nn.conv2d_transpose(
        input, filter=w, output_shape=[batch, out_h, out_w, output_channels],
        strides=[1, s_h, s_w, 1], padding=padding)
    return tf.nn.bias_add(deconv, b)


def flatten(x):
    x_shape = x.get_shape()
    d = int(np.prod(x_shape[1:]))
    return tf.reshape(x, [-1, d])


def linear(x, output_size, label=""):
    w = tf.get_variable(label + "w", shape=[x.get_shape()[1], output_size],
                        dtype=tf.float32)
    b = tf.get_variable(label + "b", shape=[output_size], dtype=tf.float32)
    return tf.matmul(x, w) + b


def leaky_relu(x, alpha=0.01):
    pos_x = tf.nn.relu(x)
    neg_x = tf.nn.relu(-x)
    return pos_x - neg_x * alpha


def batch_norm(train_phase, x, decay=0.99, center=True, scale=True, label=""):
    with tf.variable_scope(label + "bn") as scope:
        bn_train = tf_layers.batch_norm(
            x, decay=decay, center=center, scale=scale,
            updates_collections=None, is_training=True,
            reuse=None, trainable=True, scope=scope)
        bn_test = tf_layers.batch_norm(
            x, decay=decay, center=center, scale=scale,
            updates_collections=None, is_training=False,
            reuse=True, trainable=True, scope=scope)
    return tf.cond(train_phase, lambda: bn_train, lambda: bn_test)


def dropout(train_phase, x, keep_prob):
    return tf.cond(train_phase, lambda: tf.nn.dropout(x, keep_prob),
                   lambda: x)


class ParamsTracker(object):

    def __init__(self):
        self._num_params = 0

    def get_params(self):
        ans = tf.trainable_variables()[self._num_params:]
        self._num_params += len(ans)
        return ans
