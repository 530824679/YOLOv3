# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2020/11/01
# @Author  : Oscar Chen
# @Email   : 530824679@qq.com
# @File    : configs.py
# Description :config parameters
# --------------------------------------
import os

path_params = {
    'data_path': '/home/chenwei/HDD/Project/datasets/object_detection/VOC2028',
    'pretrain_weights': '/home/chenwei/HDD/Project/CenterNet/weights/resnet34.npy',
    'checkpoints_dir': './checkpoints',
    'weights_dir': './weights',
    'logs_dir': './logs',
    'tfrecord_dir': './tfrecord',
    'checkpoints_name': 'model.ckpt',
    'train_tfrecord_name': 'train.tfrecord',
    'test_output_dir': './test'
}

model_params = {
    'input_height': 416,                                # 图片高度
    'input_width': 416,                                 # 图片宽度
    'channels': 3,                                      # 输入图片通道数
    'anchors': [[11,13], [15,17], [17,21],
               [22,25], [27,32], [37,43],
               [57,65], [104,121], [229,266]],
    'classes': ['person', 'hat'],  # 类别
    'anchor_per_scale': 3,                              # 每个尺度的anchor个数
    'upsample_method': "resize",                        # 上采样的方式
    'iou_threshold': 0.5,
    'max_bbox_per_scale': 150,
    'batch_norm_decay': 0.99,  # decay in bn ops
    'weight_decay': 5e-4,  # l2 weight decay
    'global_step': 0,  # used when resuming training
    'warm_up_epoch': 3
}

solver_params = {
    'gpu': '0',                     # 使用的gpu索引
    'lr': 1e-5,                     # 初始学习率
    'decay_steps': 5000,            # 衰变步数
    'decay_rate': 0.95,             # 衰变率
    'staircase': True,
    'batch_size': 8,               # 每批次输入的数据个数
    'total_epoches': 1000,          # 训练的最大迭代次数
    'save_step': 1000,              # 权重保存间隔
    'weight_decay': 0.0001,         # 正则化系数
    'restore': False,               # 支持restore
    'pre_train': True
}

test_params = {
    'prob_threshold': 0.3,         # 类别置信度分数阈值
    'iou_threshold': 0.45,           # nms阈值，小于0.45被过滤掉
    'max_output_size': 10           # nms选择的边界框最大数量
}

classes_map = {'person': 0, 'hat': 1}
