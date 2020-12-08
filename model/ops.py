# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2020/11/01
# @Author  : Oscar Chen
# @Email   : 530824679@qq.com
# @File    : ops.py
# Description :base operators.
# --------------------------------------

import tensorflow as tf
import tensorflow.contrib.slim as slim

def fixed_padding(inputs, kernel_size):
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]], mode='CONSTANT')
    return padded_inputs

def leaky_relu(inputs, alpha):
    return tf.nn.leaky_relu(inputs, alpha=alpha, name='leaky_relu')

def conv2d(inputs, filters, kernel_size, strides=1):
    if strides > 1:
        inputs = fixed_padding(inputs, kernel_size)
    inputs = slim.conv2d(inputs, filters, kernel_size, stride=strides, padding=('SAME' if strides == 1 else 'VALID'))
    return inputs

def residual_block(inputs, filters):
    shortcut = inputs
    net = conv2d(inputs, filters * 1, 1)
    net = conv2d(net, filters * 2, 3)

    net = net + shortcut

    return net

def maxpool(inputs, size=2, stride=2):
    pool = tf.layers.max_pooling2d(inputs, pool_size=size, strides=stride, padding='SAME')

    return pool

def route(previous_output, current_output):
    output = tf.concat([current_output, previous_output], axis=-1)

    return output

def upsample(inputs, method="deconv"):
    assert method in ["resize", "deconv"]

    if method == "resize":
        input_shape = tf.shape(inputs)
        output = tf.image.resize_nearest_neighbor(inputs, (input_shape[1] * 2, input_shape[2] * 2))

    if method == "deconv":
        numm_filter = inputs.shape.as_list()[-1]
        output = tf.layers.conv2d_transpose(inputs, numm_filter, kernel_size=2, padding='same', strides=(2,2), kernel_initializer=tf.random_normal_initializer())

    return output