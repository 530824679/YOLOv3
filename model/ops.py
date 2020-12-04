# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2020/11/01
# @Author  : Oscar Chen
# @Email   : 530824679@qq.com
# @File    : ops.py
# Description :base operators.
# --------------------------------------

import tensorflow as tf

def leaky_relu(inputs, alpha):
    return tf.nn.leaky_relu(inputs, alpha=alpha, name='leaky_relu')

def conv2d(inputs, filters_shape, trainable, downsample=False, activate=True, bn=True, scope='conv2d'):
    with tf.variable_scope(scope):
        if downsample:
            pad_h, pad_w = (filters_shape[0] - 2) // 2 + 1, (filters_shape[1] - 2) // 2 + 1
            paddings = tf.constant([[0, 0], [pad_h, pad_h], [pad_w, pad_w], [0, 0]])
            input_data = tf.pad(inputs, paddings, 'CONSTANT')
            strides = (1, 2, 2, 1)
            padding = 'VALID'
        else:
            input_data = inputs
            strides = (1, 1, 1, 1)
            padding = "SAME"

        weight = tf.get_variable(name=scope+'_weight', dtype=tf.float32, trainable=True, shape=filters_shape, initializer=tf.random_normal_initializer(stddev=0.01))
        conv = tf.nn.conv2d(input=input_data, filter=weight, strides=strides, padding=padding, name=scope+'_conv')

        if bn:
            conv = tf.layers.batch_normalization(conv, beta_initializer=tf.zeros_initializer(),
                                                 gamma_initializer=tf.ones_initializer(),
                                                 moving_mean_initializer=tf.zeros_initializer(),
                                                 moving_variance_initializer=tf.ones_initializer(), training=trainable)
        else:
            bias = tf.get_variable(name='bias', shape=filters_shape[-1], trainable=True, dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, bias)

        if activate == True:
            conv = leaky_relu(conv, alpha=0.1)

    return conv

def residual_block(inputs, input_channel, filter_num1, filter_num2, trainable, scope):
    short_cut = inputs

    with tf.variable_scope(scope):
        input_data = conv2d(inputs, filters_shape=(1, 1, input_channel, filter_num1), trainable=trainable, scope='conv1')
        input_data = conv2d(input_data, filters_shape=(3, 3, filter_num1,   filter_num2), trainable=trainable, scope='conv2')

        residual_output = input_data + short_cut

    return residual_output

def maxpool(inputs, size=2, stride=2, name='maxpool'):
    with tf.name_scope(name):
         pool = tf.layers.max_pooling2d(inputs, pool_size=size, strides=stride, padding='SAME')

    return pool

def route(previous_output, current_output, scope):
    with tf.variable_scope(scope):
        output = tf.concat([current_output, previous_output], axis=-1)

    return output

def upsample(inputs, method="deconv", scope="upsample"):
    assert method in ["resize", "deconv"]

    if method == "resize":
        with tf.variable_scope(scope):
            input_shape = tf.shape(inputs)
            output = tf.image.resize_nearest_neighbor(inputs, (input_shape[1] * 2, input_shape[2] * 2))

    if method == "deconv":
        numm_filter = inputs.shape.as_list()[-1]
        output = tf.layers.conv2d_transpose(inputs, numm_filter, kernel_size=2, padding='same', strides=(2,2), kernel_initializer=tf.random_normal_initializer())

    return output