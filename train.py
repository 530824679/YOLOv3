# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2020/11/01
# @Author  : Oscar Chen
# @Email   : 530824679@qq.com
# @File    : train.py
# Description : train code
# --------------------------------------

import os
import math
import time
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import tensorflow.contrib.slim as slim

from cfg.config import path_params, model_params, solver_params
from model import network
from data import tfrecord
from utils.process_utils import total_sample


def train():
    start_step = 0
    restore = solver_params['restore']
    pre_train = solver_params['pre_train']
    checkpoint_dir = path_params['checkpoints_dir']
    checkpoints_name = path_params['checkpoints_name']
    tfrecord_dir = path_params['tfrecord_dir']
    tfrecord_name = path_params['train_tfrecord_name']
    log_dir = path_params['logs_dir']
    batch_size = solver_params['batch_size']
    num_class = len(model_params['classes'])

    # 配置GPU
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options)

    # 解析得到训练样本以及标注
    data = tfrecord.TFRecord()
    train_tfrecord = os.path.join(tfrecord_dir, tfrecord_name)
    data_num = total_sample(train_tfrecord)
    batch_num = int(math.ceil(float(data_num) / batch_size))
    dataset = data.create_dataset(train_tfrecord, batch_num, batch_size=batch_size, is_shuffle=True)
    iterator = dataset.make_one_shot_iterator()
    inputs, y_true_13, y_true_26, y_true_52 = iterator.get_next()

    inputs.set_shape([None, 416, 416, 3])
    y_true_13.set_shape([None, 13, 13, 3, 5+num_class])
    y_true_26.set_shape([None, 26, 26, 3, 5+num_class])
    y_true_52.set_shape([None, 52, 52, 3, 5+num_class])

    y_true = [y_true_13, y_true_26, y_true_52]

    # 构建网络
    with tf.variable_scope('yolov3'):
        model = network.Network(len(model_params['classes']), model_params['anchors'], is_train=True)
        logits = model.build_network(inputs)

    # 计算损失函数
    loss = model.calc_loss(logits, y_true)
    l2_loss = tf.losses.get_regularization_loss()

    # restore_include = None
    # restore_exclude = ['yolov3/yolov3_head/Conv_14', 'yolov3/yolov3_head/Conv_6', 'yolov3/yolov3_head/Conv_22']
    # update_part = ['yolov3/yolov3_head']
    # saver_to_restore = tf.train.Saver(var_list=tf.contrib.framework.get_variables_to_restore(include=restore_include, exclude=restore_exclude))
    # update_vars = tf.contrib.framework.get_variables_to_restore(include=update_part)

    global_step = tf.Variable(float(0), trainable=False)#, collections=[tf.GraphKeys.LOCAL_VARIABLES])
    learning_rate = tf.train.exponential_decay(solver_params['lr'], global_step, solver_params['decay_steps'], solver_params['decay_rate'], staircase=True)
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss[0] + l2_loss, var_list=None, global_step=global_step)
        #gvs = optimizer.compute_gradients(loss[0] + l2_loss, var_list=update_vars)
        #clip_grad_var = [gv if gv[0] is None else [tf.clip_by_norm(gv[0], 100.), gv[1]] for gv in gvs]
        #train_op = optimizer.apply_gradients(clip_grad_var, global_step=global_step)

    tf.summary.scalar("learning_rate", learning_rate)
    tf.summary.scalar('total_loss', loss[0])
    tf.summary.scalar('loss_xy', loss[1])
    tf.summary.scalar('loss_wh', loss[2])
    tf.summary.scalar('loss_conf', loss[3])
    tf.summary.scalar('loss_class', loss[4])

    # 配置tensorboard
    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(log_dir, graph=tf.get_default_graph(), flush_secs=60)

    save_variable = tf.global_variables()
    saver_to_restore = tf.train.Saver(save_variable, max_to_keep=50)
    with tf.Session(config=config) as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

        if restore == True:
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                stem = os.path.basename(ckpt.model_checkpoint_path)
                restore_step = int(stem.split('.')[0].split('-')[-1])
                start_step = restore_step
                sess.run(global_step.assign(restore_step))
                saver_to_restore.restore(sess, ckpt.model_checkpoint_path)
                print('Restoreing from {}'.format(ckpt.model_checkpoint_path))
            else:
                print("Failed to find a checkpoint")

        if pre_train == True:
            saver_to_restore.restore(sess, os.path.join(path_params['weights_dir'], 'yolov3.ckpt'))

        summary_writer.add_graph(sess.graph)

        for epoch in range(start_step + 1, solver_params['total_epoches']):
            train_epoch_loss, train_epoch_xy_loss, train_epoch_wh_loss, train_epoch_confs_loss, train_epoch_class_loss = [], [], [], [], []
            for index in tqdm(range(batch_num)):
                _, summary_, loss_, xy_loss_, wh_loss_, confs_loss_, class_loss_, global_step_, lr = sess.run([train_op, summary_op, loss[0], loss[1], loss[2], loss[3], loss[4], global_step, learning_rate])
                print("Epoch: {}, global_step: {}, lr: {:.8f}, total_loss: {:.3f}, diou_loss: {:.3f}, confs_loss: {:.3f}, class_loss: {:.3f}".format(
                        epoch, global_step_, lr, loss_, xy_loss_, wh_loss_, confs_loss_, class_loss_))

                train_epoch_loss.append(loss_)
                train_epoch_xy_loss.append(xy_loss_)
                train_epoch_wh_loss.append(wh_loss_)
                train_epoch_confs_loss.append(confs_loss_)
                train_epoch_class_loss.append(class_loss_)

                summary_writer.add_summary(summary_, global_step_)

            train_epoch_loss, train_epoch_xy_loss, train_epoch_wh_loss, train_epoch_confs_loss, train_epoch_class_loss = np.mean(train_epoch_loss), np.mean(train_epoch_xy_loss), np.mean(train_epoch_wh_loss),np.mean(train_epoch_confs_loss), np.mean(train_epoch_class_loss)
            print("Epoch: {}, global_step: {}, lr: {:.8f}, total_loss: {:.3f}, xy_loss: {:.3f}, wh_loss: {:.3f},confs_loss: {:.3f}, class_loss: {:.3f}".format(epoch, global_step_, lr, train_epoch_loss, train_epoch_xy_loss, train_epoch_wh_loss, train_epoch_confs_loss, train_epoch_class_loss))
            saver_to_restore.save(sess, os.path.join(checkpoint_dir, checkpoints_name), global_step=epoch)

        sess.close()

if __name__ == '__main__':
    train()