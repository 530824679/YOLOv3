# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2020/11/01
# @Author  : Oscar Chen
# @Email   : 530824679@qq.com
# @File    : train.py
# Description : train code
# --------------------------------------

import os
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim

from cfg.config import path_params, model_params, solver_params
from model import network
from data import tfrecord


def train():
    start_step = 0
    log_step = solver_params['log_step']
    restore = solver_params['restore']
    pre_train = solver_params['pre_train']
    checkpoint_dir = path_params['checkpoints_dir']
    checkpoints_name = path_params['checkpoints_name']
    tfrecord_dir = path_params['tfrecord_dir']
    tfrecord_name = path_params['train_tfrecord_name']
    log_dir = path_params['logs_dir']

    # 配置GPU
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options)

    # 解析得到训练样本以及标注
    data = tfrecord.TFRecord()
    train_tfrecord = os.path.join(tfrecord_dir, tfrecord_name)
    dataset = data.create_dataset(train_tfrecord, batch_size=4, is_shuffle=True)
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

    tf.summary.scalar('total_loss', loss[0])
    tf.summary.scalar('loss_xy', loss[1])
    tf.summary.scalar('loss_wh', loss[2])
    tf.summary.scalar('loss_conf', loss[3])
    tf.summary.scalar('loss_class', loss[4])
    tf.summary.scalar('loss_l2', l2_loss)
    tf.summary.scalar('loss_ratio', l2_loss / loss[0])

    # setting restore parts and vars to update
    restore_include = None
    restore_exclude = ['yolov3/yolov3_head/Conv_14', 'yolov3/yolov3_head/Conv_6', 'yolov3/yolov3_head/Conv_22']
    update_part = ['yolov3/yolov3_head']
    saver_to_restore = tf.train.Saver(var_list=tf.contrib.framework.get_variables_to_restore(include=restore_include, exclude=restore_exclude))
    update_vars = tf.contrib.framework.get_variables_to_restore(include=update_part)

    global_step = tf.Variable(float(0), trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
    learning_rate = tf.train.exponential_decay(solver_params['learning_rate'], global_step, 30000, 0.1, name='learning_rate')

    # 设置优化器
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        # 采用的优化方法是随机梯度下降
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
        gvs = optimizer.compute_gradients(loss[0] + l2_loss, var_list=update_vars)
        clip_grad_var = [gv if gv[0] is None else [tf.clip_by_norm(gv[0], 100.), gv[1]] for gv in gvs]
        train_op = optimizer.apply_gradients(clip_grad_var, global_step=global_step)

    # 配置tensorboard
    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(log_dir, graph=tf.get_default_graph(), flush_secs=60)

    saver = tf.train.Saver()
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
        elif pre_train == True:
            saver_to_restore.restore(sess, os.path.join(path_params['weights_dir'], 'yolov3.ckpt'))

        summary_writer.add_graph(sess.graph)

        for epoch in range(start_step + 1, solver_params['total_epoches']):
            _, summary_, loss_, loss_xy_, loss_wh_, confs_loss_, class_loss_, global_step_, lr = sess.run(
                [train_op, summary_op, loss[0], loss[1], loss[2],loss[3], loss[4], global_step,
                 learning_rate])

            print(
                "Epoch: {}, global_step: {}, lr: {:.8f}, total_loss: {:.3f}, loss_xy: {:.3f}, loss_wh: {:.3f}, confs_loss: {:.3f}, class_loss: {:.3f}".format(
                    epoch, global_step_, lr, loss_, loss_xy_, loss_wh_, confs_loss_, class_loss_))

            if epoch % solver_params['save_step'] == 0 and epoch > 0:
                save_path = saver.save(sess, os.path.join(checkpoint_dir, checkpoints_name), global_step=epoch)
                print('Save modle into {}....'.format(save_path))

            if epoch % log_step == 0 and epoch > 0:
                summary_writer.add_summary(summary_, global_step=global_step_)

        sess.close()

if __name__ == '__main__':
    train()