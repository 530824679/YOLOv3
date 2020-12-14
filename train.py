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
    log_step = solver_params['log_step']
    restore = solver_params['restore']
    checkpoint_dir = path_params['checkpoints_dir']
    checkpoints_name = path_params['checkpoints_name']
    tfrecord_dir = path_params['tfrecord_dir']
    tfrecord_name = path_params['train_tfrecord_name']
    log_dir = path_params['logs_dir']
    batch_size = solver_params['batch_size']

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
    y_true_13.set_shape([None, 13, 13, 3, 7])
    y_true_26.set_shape([None, 26, 26, 3, 7])
    y_true_52.set_shape([None, 52, 52, 3, 7])

    y_true = [y_true_13, y_true_26, y_true_52]

    # 构建网络
    model = network.Network(len(model_params['classes']), model_params['anchors'], is_train=True)
    with tf.variable_scope('yolov3'):
        logits = model.build_network(inputs)

    # 计算损失函数
    loss = model.calc_loss(logits, y_true)
    l2_loss = tf.losses.get_regularization_loss()

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(solver_params['lr'], global_step, solver_params['decay_steps'], solver_params['decay_rate'], staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss[0], global_step=global_step)

    tf.summary.scalar("learning_rate", learning_rate)
    tf.summary.scalar('total_loss', loss[0])
    tf.summary.scalar('loss_diou', loss[1])
    tf.summary.scalar('loss_conf', loss[2])
    tf.summary.scalar('loss_class', loss[3])
    tf.summary.scalar('loss_l2', l2_loss)
    tf.summary.scalar('loss_ratio', l2_loss / loss[0])

    # 配置tensorboard
    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(log_dir, graph=tf.get_default_graph(), flush_secs=60)

    # 模型保存
    save_variable = tf.global_variables()
    saver = tf.train.Saver(save_variable, max_to_keep=1000)


    with tf.Session(config=config) as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

        if restore == True:
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                stem = os.path.basename(ckpt.model_checkpoint_path)
                restore_step = int(stem.split('.')[0].split('-')[-1])
                start_step = restore_step
                sess.run(global_step.assign(restore_step))
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Restoreing from {}'.format(ckpt.model_checkpoint_path))
            else:
                print("Failed to find a checkpoint")

        summary_writer.add_graph(sess.graph)

        for epoch in range(start_step + 1, solver_params['total_epoches']):
            train_epoch_loss, train_epoch_diou_loss, train_epoch_confs_loss, train_epoch_class_loss = [], [], [], []
            for index in tqdm(range(batch_num)):
                _, summary_, loss_, diou_loss_, confs_loss_, class_loss_, global_step_, lr = sess.run([train_op, summary_op, loss[0], loss[1], loss[2],loss[3], global_step, learning_rate])

                train_epoch_loss.append(loss_)
                train_epoch_diou_loss.append(diou_loss_)
                train_epoch_confs_loss.append(confs_loss_)
                train_epoch_class_loss.append(class_loss_)

                summary_writer.add_summary(summary_, global_step_)

                train_epoch_loss, train_epoch_diou_loss, train_epoch_confs_loss, train_epoch_class_loss = np.mean(train_epoch_loss), np.mean(train_epoch_diou_loss), np.mean(train_epoch_confs_loss), np.mean(train_epoch_class_loss)

                print("Epoch: {}, global_step: {}, lr: {:.8f}, total_loss: {:.3f}, diou_loss: {:.3f}, confs_loss: {:.3f}, class_loss: {:.3f}".format(epoch, global_step_, lr, train_epoch_loss, train_epoch_diou_loss, train_epoch_confs_loss, train_epoch_class_loss))
                saver.save(sess, os.path.join(checkpoint_dir, checkpoints_name), global_step=epoch)

        sess.close()

if __name__ == '__main__':
    train()