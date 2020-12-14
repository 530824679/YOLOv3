# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2020/11/01
# @Author  : Oscar Chen
# @Email   : 530824679@qq.com
# @File    : convert_weight.py
# Description : process weight
# --------------------------------------

import os
import random
import numpy as np
import tensorflow as tf
from model.network import Network

def load_weights(var_list, weights_file):
    """
    loads and converts pre-trained weight，the first 5 values correspond to
    major version (4 bytes)
    minor version (4 bytes)
    revision      (4 bytes)
    images seen   (8 bytes)
    :param var_list:  list of network variables
    :param weights_file: name of the binary file
    :return:
    """
    with open(weights_file, 'rb') as fp:
        # skip first 5 values
        np.fromfile(fp, dtype=np.int32, count=5)
        weights = np.fromfile(fp, dtype=np.float32)

    ptr = 0
    i = 0
    assign_ops = []

    while i < len(var_list) - 1:
        var1 = var_list[i]
        var2 = var_list[i + 1]
        if 'Conv' in var1.name.split('/')[-2]:
            # check type of next layer
            if 'BatchNorm' in var2.name.split('/')[-2]:
                # load batch norm params
                gamma, beta, mean, var = var_list[i + 1:i + 5]
                batch_norm_vars = [beta, gamma, mean, var]
                for var in batch_norm_vars:
                    shape = var.shape.as_list()
                    num_params = np.prod(shape)
                    var_weights = weights[ptr:ptr + num_params].reshape(shape)
                    ptr += num_params
                    assign_ops.append(tf.assign(var, var_weights, validate_shape=True))
                # we move the pointer by 4, because we loaded 4 variables
                i += 4
            elif 'Conv' in var2.name.split('/')[-2]:
                # load biases
                bias = var2
                bias_shape = bias.shape.as_list()
                bias_params = np.prod(bias_shape)
                bias_weights = weights[ptr:ptr + bias_params].reshape(bias_shape)
                ptr += bias_params
                assign_ops.append(tf.assign(bias, bias_weights, validate_shape=True))
                # we loaded 1 variable
                i += 1
            # we can load weights of conv layer
            shape = var1.shape.as_list()
            num_params = np.prod(shape)

            var_weights = weights[ptr:ptr + num_params].reshape((shape[3], shape[2], shape[0], shape[1]))
            # remember to transpose to column-major
            var_weights = np.transpose(var_weights, (2, 3, 1, 0))
            ptr += num_params
            assign_ops.append(tf.assign(var1, var_weights, validate_shape=True))
            i += 1
    return assign_ops

def weights_to_ckpt():
    num_class = 80
    image_size = 416
    anchors = [[676,197], [763,250], [684,283],
               [868,231], [745,273], [544,391],
               [829,258], [678,316, 713,355]]
    weight_path = '../weights/yolov3.weights'
    save_path = '../weights/yolov3.ckpt'

    model = Network(num_class, anchors, False)
    with tf.Session() as sess:
        inputs = tf.placeholder(tf.float32, [1, image_size, image_size, 3])

        with tf.variable_scope('yolov3'):
            feature_maps = model.build_network(inputs)

        saver = tf.train.Saver(var_list=tf.global_variables(scope='yolov3'))

        load_ops = load_weights(tf.global_variables(scope='yolov3'), weight_path)
        sess.run(load_ops)
        saver.save(sess, save_path=save_path)
        print('TensorFlow model checkpoint has been saved to {}'.format(save_path))

def remove_optimizers_params():
    ckpt_path = ''
    class_num = 2
    save_dir = 'shrinked_ckpt'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    anchors = [[676, 197], [763, 250], [684, 283],
               [868, 231], [745, 273], [544, 391],
               [829, 258], [678, 316, 713, 355]]

    image = tf.placeholder(tf.float32, [1, 416, 416, 3])
    model = Network(class_num, anchors, False)
    with tf.variable_scope('yolov3'):
        feature_maps = model.build_network(image)

    saver_to_restore = tf.train.Saver()
    saver_to_save = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver_to_restore.restore(sess, ckpt_path)
        saver_to_save.save(sess, save_dir + '/shrinked')

if __name__ == '__main__':
    weights_to_ckpt()