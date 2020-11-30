# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2020/11/01
# @Author  : Oscar Chen
# @Email   : 530824679@qq.com
# @File    : network.py
# Description :YOLO v3 network architecture
# --------------------------------------

from cfg.config import model_params
from model.loss import Loss
from model.ops import *

class Network(object):
    def __init__(self, inputs, is_train):
        self.is_train = is_train
        self.strides = model_params['strides']
        self.classes = model_params['classes']
        self.class_num = len(self.classes)
        self.anchors = model_params['anchors']
        self.anchor_per_sacle = model_params['anchor_per_sacle']
        self.iou_loss_thresh = model_params['iou_threshold']
        self.upsample_method = model_params['upsample_method']

    def darknet53(self, inputs, scope='darknet53'):
        """
        定义网络特征提取层
        :param inputs:待输入的样本图片
        :param scope: 命名空间
        :return: 三个不同尺度的基础特征图输出
        """
        with tf.variable_scope('darknet'):
            input_data = conv2d(inputs, filters_shape=(3, 3, 3, 32), trainable=self.is_train, scope='conv0')
            input_data = conv2d(input_data, filters_shape=(3, 3, 32, 64), trainable=self.is_train, scope='conv1', downsample=True)

            for i in range(1):
                input_data = residual_block(input_data, 64, 32, 64, trainable=self.is_train, scope='residual%d' % (i + 0))

            input_data = conv2d(input_data, filters_shape=(3, 3, 64, 128), trainable=self.is_train, scope='conv4', downsample=True)

            for i in range(2):
                input_data = residual_block(input_data, 128, 64, 128, trainable=self.is_train, scope='residual%d' % (i + 1))

            input_data = conv2d(input_data, filters_shape=(3, 3, 128, 256), trainable=self.is_train, scope='conv9', downsample=True)

            for i in range(8):
                input_data = residual_block(input_data, 256, 128, 256, trainable=self.is_train, scope='residual%d' % (i + 3))

            route_1 = input_data
            input_data = conv2d(input_data, filters_shape=(3, 3, 256, 512), trainable=self.is_train, scope='conv26', downsample=True)

            for i in range(8):
                input_data = residual_block(input_data, 512, 256, 512, trainable=self.is_train, scope='residual%d' % (i + 11))

            route_2 = input_data
            input_data = conv2d(input_data, filters_shape=(3, 3, 512, 1024), trainable=self.is_train, scope='conv43', downsample=True)

            for i in range(4):
                input_data = residual_block(input_data, 1024, 512, 1024, trainable=self.is_train, scope='residual%d' % (i + 19))

            return route_1, route_2, input_data

    def detect_head(self, route_1, route_2, input_data):
        input_data = conv2d(input_data, (1, 1, 1024, 512), self.is_train, 'conv52')
        input_data = conv2d(input_data, (3, 3, 512, 1024), self.is_train, 'conv53')
        input_data = conv2d(input_data, (1, 1, 1024, 512), self.is_train, 'conv54')
        input_data = conv2d(input_data, (3, 3, 512, 1024), self.is_train, 'conv55')
        input_data = conv2d(input_data, (1, 1, 1024, 512), self.is_train, 'conv56')

        conv_lobj_branch = conv2d(input_data, (3, 3, 512, 1024), self.is_train, scope='conv_lobj_branch')
        conv_lbbox = conv2d(conv_lobj_branch, (1, 1, 1024, 3 * (self.class_num + 5)), trainable=self.is_train,
                            scope='conv_lbbox', activate=False, bn=False)

        input_data = conv2d(input_data, (1, 1, 512, 256), self.is_train, 'conv57')
        input_data = upsample(input_data, scope='upsample0', method=self.upsample_method)

        with tf.variable_scope('route_1'):
            input_data = tf.concat([input_data, route_2], axis=-1)

        input_data = conv2d(input_data, (1, 1, 768, 256), self.is_train, 'conv58')
        input_data = conv2d(input_data, (3, 3, 256, 512), self.is_train, 'conv59')
        input_data = conv2d(input_data, (1, 1, 512, 256), self.is_train, 'conv60')
        input_data = conv2d(input_data, (3, 3, 256, 512), self.is_train, 'conv61')
        input_data = conv2d(input_data, (1, 1, 512, 256), self.is_train, 'conv62')

        conv_mobj_branch = conv2d(input_data, (3, 3, 256, 512), self.is_train, scope='conv_mobj_branch')
        conv_mbbox = conv2d(conv_mobj_branch, (1, 1, 512, 3 * (self.class_num + 5)), trainable=self.is_train,
                            scope='conv_mbbox', activate=False, bn=False)

        input_data = conv2d(input_data, (1, 1, 256, 128), self.is_train, 'conv63')
        input_data = upsample(input_data, scope='upsample1', method=self.upsample_method)

        with tf.variable_scope('route_2'):
            input_data = tf.concat([input_data, route_1], axis=-1)

        input_data = conv2d(input_data, (1, 1, 384, 128), self.is_train, 'conv64')
        input_data = conv2d(input_data, (3, 3, 128, 256), self.is_train, 'conv65')
        input_data = conv2d(input_data, (1, 1, 256, 128), self.is_train, 'conv66')
        input_data = conv2d(input_data, (3, 3, 128, 256), self.is_train, 'conv67')
        input_data = conv2d(input_data, (1, 1, 256, 128), self.is_train, 'conv68')

        conv_sobj_branch = conv2d(input_data, (3, 3, 128, 256), self.is_train, scope='conv_sobj_branch')
        conv_sbbox = conv2d(conv_sobj_branch, (1, 1, 256, 3 * (self.class_num + 5)), trainable=self.is_train,
                            scope='conv_sbbox', activate=False, bn=False)

        return conv_lbbox, conv_mbbox, conv_sbbox

    def build_network(self, inputs, scope='yolo_v3'):
        """
        定义前向传播过程
        :param inputs:待输入的样本图片
        :param scope: 命名空间
        :return: 三个不同尺度的检测层输出
        """
        route_1, route_2, input_data = self.darknet53(inputs, self.is_train)

        try:
            conv_lbbox, conv_mbbox, conv_sbbox = self.detect_head(route_1, route_2, input_data)
        except:
            raise NotImplementedError("Can not build up yolov3 network!")

        with tf.variable_scope('pred_sbbox'):
            self.pred_sbbox = self.reorg_layer(conv_sbbox, self.anchors[0], self.strides[0])

        with tf.variable_scope('pred_mbbox'):
            self.pred_mbbox = self.reorg_layer(conv_mbbox, self.anchors[1], self.strides[1])

        with tf.variable_scope('pred_lbbox'):
            self.pred_lbbox = self.reorg_layer(conv_lbbox, self.anchors[2], self.strides[2])



    def reorg_layer(self, feature_maps, anchors, stride):
        """
        解码网络输出的特征图
        :param feature_maps:网络输出的特征图
        :param anchors:当前层使用的anchor尺度 shape is [3, 2]
        :param stride:特征图相比原图的缩放比例
        :return: 预测层最终的输出 shape=[batch_size, feature_size, feature_size, anchor_per_scale, 5 + num_classes]
        """
        feature_shape = tf.shape(feature_maps)
        batch_size = feature_shape[0]
        feature_height = feature_shape[1]
        feature_width = feature_shape[2]

        # rescale the anchors to the feature map [w, h]
        anchor_per_scale = len(anchors)
        rescaled_anchors = [(anchor[0] / stride, anchor[1] / stride) for anchor in anchors]

        # 网络输出转化——偏移量、置信度、类别概率
        predict = tf.reshape(feature_maps, [batch_size, feature_height * feature_width, anchor_per_scale, self.class_num + 5])
        # 中心坐标相对于该cell左上角的偏移量，sigmoid函数归一化到0-1
        xy_offset = tf.nn.sigmoid(predict[:, :, :, 0:2])
        # 相对于anchor的wh比例，通过e指数解码
        wh_offset = tf.clip_by_value(tf.exp(predict[:, :, :, 2:4]), 1e-9, 50) * rescaled_anchors
        # 置信度，sigmoid函数归一化到0-1
        pred_obj = tf.nn.sigmoid(predict[:, :, :, 4:5])
        # 网络回归的是得分,用softmax转变成类别概率
        pred_class = tf.nn.softmax(predict[:, :, :, 5:])

        # 构建特征图每个cell的左上角的xy坐标
        height_index = tf.range(feature_height, dtype=tf.float32)
        width_index = tf.range(feature_width, dtype=tf.float32)
        x_cell, y_cell = tf.meshgrid(height_index, width_index)
        # shape = [H*W, num_anchors, num_class+5]
        x_cell = tf.reshape(x_cell, [1, -1, 1])
        y_cell = tf.reshape(y_cell, [1, -1, 1])

        # decode to original image scale
        bbox_x = (x_cell + xy_offset[:, :, :, 0]) * stride
        bbox_y = (y_cell + xy_offset[:, :, :, 1]) * stride
        bbox_w = (anchors[:, 0] * wh_offset[:, :, :, 0]) * stride
        bbox_h = (anchors[:, 1] * wh_offset[:, :, :, 1]) * stride

        boxes_xy = tf.concat([bbox_x, bbox_y], axis=3)
        boxes_wh = tf.concat([bbox_w, bbox_h], axis=3)
        pred_xywh = tf.concat([boxes_xy, boxes_wh], axis=-1)

        return tf.concat([pred_xywh, pred_obj, pred_class], axis=-1)