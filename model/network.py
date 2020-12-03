# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2020/11/01
# @Author  : Oscar Chen
# @Email   : 530824679@qq.com
# @File    : network.py
# Description :YOLO v3 network architecture
# --------------------------------------

import numpy as np
import math
from cfg.config import model_params, solver_params
from model.ops import *

class Network(object):
    def __init__(self, is_train):
        self.is_train = is_train
        self.classes = model_params['classes']
        self.class_num = len(self.classes)
        self.anchors = np.array(model_params['anchors'])
        self.anchor_per_sacle = model_params['anchor_per_sacle']
        self.upsample_method = model_params['upsample_method']

        self.batch_size = solver_params['batch_size']
        self.input_height = model_params['input_height']
        self.input_width = model_params['input_width']
        self.input_size = np.array(self.input_height, self.input_width)
        self.iou_threshold = model_params['iou_threshold']

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

        return conv_lbbox, conv_mbbox, conv_sbbox

    def reorg_layer(self, feature_maps, anchors):
        """
        解码网络输出的特征图
        :param feature_maps:网络输出的特征图
        :param anchors:当前层使用的anchor尺度 shape is [3, 2]
        :param stride:特征图相比原图的缩放比例
        :return: 预测层最终的输出 shape=[batch_size, feature_size, feature_size, anchor_per_scale, 5 + num_classes]
        """
        feature_shape = feature_maps.get_shape().as_list()[1:3]
        batch_size = feature_shape[0]
        ratio = tf.cast(self.input_size / feature_shape, tf.float32)
        anchor_per_scale = self.anchor_per_sacle

        # rescale the anchors to the feature map [w, h]
        rescaled_anchors = [(anchor[0] / ratio[1], anchor[1] / ratio[0]) for anchor in anchors]

        # 网络输出转化——偏移量、置信度、类别概率
        feature_maps = tf.reshape(feature_maps, [batch_size, feature_shape[0] * feature_shape[1], anchor_per_scale, self.class_num + 5])
        # 中心坐标相对于该cell左上角的偏移量，sigmoid函数归一化到0-1
        xy_offset = tf.nn.sigmoid(feature_maps[..., 0:2])
        # 相对于anchor的wh比例，通过e指数解码
        wh_offset = tf.clip_by_value(tf.exp(feature_maps[..., 2:4]), 1e-9, 50) * rescaled_anchors
        # 置信度，sigmoid函数归一化到0-1
        obj_probs = tf.nn.sigmoid(feature_maps[..., 4:5])
        # 网络回归的是得分,用softmax转变成类别概率
        class_probs = tf.nn.softmax(feature_maps[..., 5:])

        # 构建特征图每个cell的左上角的xy坐标
        height_index = tf.range(feature_shape[0], dtype=tf.int32)
        width_index = tf.range(feature_shape[1], dtype=tf.int32)
        x_cell, y_cell = tf.meshgrid(height_index, width_index)

        x_cell = tf.reshape(x_cell, [-1, 1])
        y_cell = tf.reshape(y_cell, [-1, 1])
        xy_cell = tf.concat([x_cell, y_cell], axis=-1)
        # shape: [13, 13, 1, 2]
        xy_cell = tf.cast(tf.reshape(xy_cell, [feature_shape[0], feature_shape[1], 1, 2]), tf.float32)

        # decode to raw image norm 0-1
        bboxes_xy = (xy_cell + xy_offset) / tf.cast(feature_shape[::-1], tf.float32)
        bboxes_wh = (rescaled_anchors * wh_offset) / tf.cast(feature_shape[::-1], tf.float32)

        if self.is_train == False:
            # 转变成坐上-右下坐标
            bboxes_xywh = tf.concat([bboxes_xy, bboxes_wh], axis=-1)
            bboxes_corners = tf.stack([bboxes_xywh[..., 0] - bboxes_xywh[..., 2] / 2,
                                       bboxes_xywh[..., 1] - bboxes_xywh[..., 3] / 2,
                                       bboxes_xywh[..., 0] + bboxes_xywh[..., 2] / 2,
                                       bboxes_xywh[..., 1] + bboxes_xywh[..., 3] / 2], axis=3)
            return bboxes_corners, obj_probs, class_probs
        return xy_cell, feature_maps, bboxes_xy, bboxes_wh

    def calc_loss(self, y_logit, y_true):
        '''
        :param y_logit: function: [feature_map_1, feature_map_2, feature_map_3]
        :param y_true: input y_true by the tf.data pipeline
        '''
        loss_xy, loss_wh, loss_conf, loss_class = 0., 0., 0., 0.
        anchor_group = [self.anchors[6:9], self.anchors[3:6], self.anchors[0:3]]

        for i in range(len(y_logit)):
            result = self.loss_layer(y_logit[i], y_true[i], anchor_group[i])
            loss_xy += result[0]
            loss_wh += result[1]
            loss_conf += result[1]
            loss_class += result[2]
        total_loss = loss_xy + loss_wh + loss_conf + loss_class

        return [total_loss, loss_xy, loss_wh, loss_conf, loss_class]

    def loss_layer(self, logits, y_true, anchors):
        '''
        calc loss function from a certain scale
        :param logits: feature maps of a certain scale. shape: [N, 13, 13, 3*(5 + num_class)] etc.
        :param y_true: y_ture from a certain scale. shape: [N, 13, 13, 3, 5 + num_class + 1] etc.
        :param anchors: shape [9, 2]
        '''
        feature_size = tf.shape(logits)[1:3]
        ratio = tf.cast(self.input_size / feature_size, tf.float32)

        # ground truth
        object_coords = y_true[:, :, :, :, 0:4]
        object_masks = y_true[:, :, :, :, 4:5]
        object_probs = y_true[:, :, :, :, 5:]

        # shape: [N, 13, 13, 5, 4] & [N, 13, 13, 5] ==> [V, 4]
        valid_true_boxes = tf.boolean_mask(object_coords, tf.cast(object_masks[..., 0], 'bool'))
        # shape: [V, 2]
        valid_true_box_xy = valid_true_boxes[:, 0:2]
        valid_true_box_wh = valid_true_boxes[:, 2:4]

        # predicts
        xy_offset, predictions, pred_box_xy, pred_box_wh = self.reorg_layer(logits, self.anchors)
        pred_conf_logits = predictions[:, :, :, :, 4:5]
        pred_prob_logits = predictions[:, :, :, :, 5:]

        # calc iou 计算每个pre_boxe与所有true_boxe的交并比.
        # valid_true_box_xx: [V,2]
        # pred_box_xx: [13,13,5,2]
        # shape: [N, 13, 13, 5, V],
        iou = self.broadcast_iou(valid_true_box_xy, valid_true_box_wh, pred_box_xy, pred_box_wh)
        # shape : [N,13,13,5]
        best_iou = tf.reduce_max(iou, axis=-1)

        # get_ignore_mask shape: [N,13,13,5,1] 0,1张量
        ignore_mask = tf.expand_dims(tf.cast(best_iou < self.iou_threshold, tf.float32), -1)

        # 图像尺寸归一化信息转换为特征图的单元格相对信息
        # shape: [N, 13, 13, 3, 2]  # 坐标反归一化
        true_xy = y_true[..., 0:2] * feature_size[::-1] - xy_offset
        pred_xy = pred_box_xy * feature_size[::-1] - xy_offset

        # shape: [N, 13, 13, 3, 2],
        true_tw_th = y_true[..., 2:4] * feature_size[::-1] / anchors
        pred_tw_th = pred_box_wh * feature_size[::-1] / anchors

        # for numerical stability 稳定训练, 为0时不对anchors进行缩放, 在模型输出值特别小是e^out_put为0
        true_tw_th = tf.where(condition=tf.equal(true_tw_th, 0), x=tf.ones_like(true_tw_th), y=true_tw_th)
        pred_tw_th = tf.where(condition=tf.equal(pred_tw_th, 0), x=tf.ones_like(pred_tw_th), y=pred_tw_th)

        true_tw_th = tf.log(tf.clip_by_value(true_tw_th, 1e-9, 1e9))
        pred_tw_th = tf.log(tf.clip_by_value(pred_tw_th, 1e-9, 1e9))

        # shape: [N, 13, 13, 3, 1]
        box_loss_scale = 2. - y_true[..., 2:3] * y_true[..., 3:4]

        xy_loss = tf.square(true_xy - pred_xy) * object_masks * box_loss_scale
        wh_loss = tf.square(true_tw_th - pred_tw_th) * object_masks * box_loss_scale

        # shape: [N, 13, 13, 3, 1]
        conf_pos_mask = object_masks
        conf_neg_mask = (1 - object_masks) * ignore_mask

        conf_loss_pos = conf_pos_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_masks, logits=pred_conf_logits)
        conf_loss_neg = conf_neg_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_masks, logits=pred_conf_logits)
        conf_loss = conf_loss_pos + conf_loss_neg
        # focal_loss
        alpha = 1.0
        gamma = 2.0
        focal_mask = alpha * tf.pow(tf.abs(object_masks - tf.sigmoid(pred_conf_logits)), gamma)
        conf_loss = conf_loss * focal_mask

        # label smooth
        delta = 0.01
        label_target = (1 - delta) * object_probs + delta * 1. / self.class_num
        # shape: [N, 13, 13, 3, 1]
        class_loss = object_masks * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_target, logits=pred_prob_logits)

        xy_loss = tf.reduce_mean(tf.reduce_sum(xy_loss, axis=[1, 2, 3, 4]))
        wh_loss = tf.reduce_mean(tf.reduce_sum(wh_loss, axis=[1, 2, 3, 4]))
        conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1, 2, 3, 4]))
        class_loss = tf.reduce_mean(tf.reduce_sum(class_loss, axis=[1, 2, 3, 4]))

        return xy_loss, wh_loss, conf_loss, class_loss

    def broadcast_iou(self, true_box_xy, true_box_wh, pred_box_xy, pred_box_wh):
        # shape:
        # true_box_??: [V, 2] V:目标数量
        # pred_box_??: [N, 13, 13, 3, 2]

        # shape: [N, 13, 13, 3, 1, 2] , 扩张维度方便进行维度广播
        pred_box_xy = tf.expand_dims(pred_box_xy, -2)
        pred_box_wh = tf.expand_dims(pred_box_wh, -2)

        # shape: [1, V, 2] V:该尺度下分feature_map 下所有的目标是目标数量
        true_box_xy = tf.expand_dims(true_box_xy, 0)
        true_box_wh = tf.expand_dims(true_box_wh, 0)

        # [N, 13, 13, 3, 1, 2] --> [N, 13, 13, 3, V, 2] & [1, V, 2] ==> [N, 13, 13, 3, V, 2] 维度广播
        # 真boxe,左上角,右下角, 假boxe的左上角,右小角,
        intersect_mins = tf.maximum(pred_box_xy - pred_box_wh / + 2.,  # 取最靠右的左上角
                                    true_box_xy - true_box_wh / 2.)
        intersect_maxs = tf.minimum(pred_box_xy + pred_box_wh / 2.,  # 取最靠左的右下角
                                    true_box_xy + true_box_wh / 2.)
        # tf.maximun 去除那些没有面积交叉的矩形框, 置0
        intersect_wh = tf.maximum(intersect_maxs - intersect_mins, 0.)  # 得到重合区域的长和宽

        # shape: [N, 13, 13, 3, V]
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]  # 重合部分面积
        # shape: [N, 13, 13, 3, 1]
        pred_box_area = pred_box_wh[..., 0] * pred_box_wh[..., 1]  # 预测区域面积
        # shape: [1, V]
        true_box_area = true_box_wh[..., 0] * true_box_wh[..., 1]  # 真实区域面积
        # [N, 13, 13, 3, V]
        iou = intersect_area / (pred_box_area + true_box_area - intersect_area)

        return iou

    def bbox_iou(self, boxes_1, boxes_2):
        """
        calculate regression loss using iou
        :param boxes_1: boxes_1 shape is [x, y, w, h]
        :param boxes_2: boxes_2 shape is [x, y, w, h]
        :return:
        """
        # transform [x, y, w, h] to [x_min, y_min, x_max, y_max]
        boxes_1 = tf.concat([boxes_1[..., :2] - boxes_1[..., 2:] * 0.5,
                             boxes_1[..., :2] + boxes_1[..., 2:] * 0.5], axis=-1)
        boxes_2 = tf.concat([boxes_2[..., :2] - boxes_2[..., 2:] * 0.5,
                             boxes_2[..., :2] + boxes_2[..., 2:] * 0.5], axis=-1)
        boxes_1 = tf.concat([tf.minimum(boxes_1[..., :2], boxes_1[..., 2:]),
                             tf.maximum(boxes_1[..., :2], boxes_1[..., 2:])], axis=-1)
        boxes_2 = tf.concat([tf.minimum(boxes_2[..., :2], boxes_2[..., 2:]),
                             tf.maximum(boxes_2[..., :2], boxes_2[..., 2:])], axis=-1)

        # calculate area of boxes_1 boxes_2
        boxes_1_area = (boxes_1[..., 2] - boxes_1[..., 0]) * (boxes_1[..., 3] - boxes_1[..., 1])
        boxes_2_area = (boxes_2[..., 2] - boxes_2[..., 0]) * (boxes_2[..., 3] - boxes_2[..., 1])

        # calculate the two corners of the intersection
        left_up = tf.maximum(boxes_1[..., :2], boxes_2[..., :2])
        right_down = tf.minimum(boxes_1[..., 2:], boxes_2[..., 2:])

        # calculate area of intersection
        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]

        # calculate union area
        union_area = boxes_1_area + boxes_2_area - inter_area

        # calculate iou
        iou = inter_area / union_area

        return iou

    def bbox_giou(self, boxes_1, boxes_2):
        """
        calculate regression loss using giou
        :param boxes_1: boxes_1 shape is [x, y, w, h]
        :param boxes_2: boxes_2 shape is [x, y, w, h]
        :return:
        """
        # transform [x, y, w, h] to [x_min, y_min, x_max, y_max]
        boxes_1 = tf.concat([boxes_1[..., :2] - boxes_1[..., 2:] * 0.5,
                             boxes_1[..., :2] + boxes_1[..., 2:] * 0.5], axis=-1)
        boxes_2 = tf.concat([boxes_2[..., :2] - boxes_2[..., 2:] * 0.5,
                             boxes_2[..., :2] + boxes_2[..., 2:] * 0.5], axis=-1)
        boxes_1 = tf.concat([tf.minimum(boxes_1[..., :2], boxes_1[..., 2:]),
                             tf.maximum(boxes_1[..., :2], boxes_1[..., 2:])], axis=-1)
        boxes_2 = tf.concat([tf.minimum(boxes_2[..., :2], boxes_2[..., 2:]),
                             tf.maximum(boxes_2[..., :2], boxes_2[..., 2:])], axis=-1)

        # calculate area of boxes_1 boxes_2
        boxes_1_area = (boxes_1[..., 2] - boxes_1[..., 0]) * (boxes_1[..., 3] - boxes_1[..., 1])
        boxes_2_area = (boxes_2[..., 2] - boxes_2[..., 0]) * (boxes_2[..., 3] - boxes_2[..., 1])

        # calculate the two corners of the intersection
        left_up = tf.maximum(boxes_1[..., :2], boxes_2[..., :2])
        right_down = tf.minimum(boxes_1[..., 2:], boxes_2[..., 2:])

        # calculate area of intersection
        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]

        # calculate union area
        union_area = boxes_1_area + boxes_2_area - inter_area

        # calculate iou
        iou = inter_area / union_area

        # calculate the upper left and lower right corners of the minimum closed convex surface
        enclose_left_up = tf.minimum(boxes_1[..., :2], boxes_2[..., :2])
        enclose_right_down = tf.maximum(boxes_1[..., 2:], boxes_2[..., 2:])

        # calculate width and height of the minimun closed convex surface
        enclose_wh = tf.maximum(enclose_right_down - enclose_left_up, 0.0)

        # calculate area of the minimun closed convex surface
        enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]

        # calculate the giou
        giou = iou - 1.0 * (enclose_area - union_area) / enclose_area

        return giou

    def bbox_diou(self, boxes_1, boxes_2):
        """
        calculate regression loss using diou
        :param boxes_1: boxes_1 shape is [x, y, w, h]
        :param boxes_2: boxes_2 shape is [x, y, w, h]
        :return:
        """
        # calculate center distance
        center_distance = tf.reduce_sum(tf.square(boxes_1[..., :2] - boxes_2[..., :2]), axis=-1)

        # transform [x, y, w, h] to [x_min, y_min, x_max, y_max]
        boxes_1 = tf.concat([boxes_1[..., :2] - boxes_1[..., 2:] * 0.5,
                             boxes_1[..., :2] + boxes_1[..., 2:] * 0.5], axis=-1)
        boxes_2 = tf.concat([boxes_2[..., :2] - boxes_2[..., 2:] * 0.5,
                             boxes_2[..., :2] + boxes_2[..., 2:] * 0.5], axis=-1)
        boxes_1 = tf.concat([tf.minimum(boxes_1[..., :2], boxes_1[..., 2:]),
                             tf.maximum(boxes_1[..., :2], boxes_1[..., 2:])], axis=-1)
        boxes_2 = tf.concat([tf.minimum(boxes_2[..., :2], boxes_2[..., 2:]),
                             tf.maximum(boxes_2[..., :2], boxes_2[..., 2:])], axis=-1)

        # calculate area of boxes_1 boxes_2
        boxes_1_area = (boxes_1[..., 2] - boxes_1[..., 0]) * (boxes_1[..., 3] - boxes_1[..., 1])
        boxes_2_area = (boxes_2[..., 2] - boxes_2[..., 0]) * (boxes_2[..., 3] - boxes_2[..., 1])

        # calculate the two corners of the intersection
        left_up = tf.maximum(boxes_1[..., :2], boxes_2[..., :2])
        right_down = tf.minimum(boxes_1[..., 2:], boxes_2[..., 2:])

        # calculate area of intersection
        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]

        # calculate union area
        union_area = boxes_1_area + boxes_2_area - inter_area

        # calculate iou
        iou = inter_area / union_area

        # calculate the upper left and lower right corners of the minimum closed convex surface
        enclose_left_up = tf.minimum(boxes_1[..., :2], boxes_2[..., :2])
        enclose_right_down = tf.maximum(boxes_1[..., 2:], boxes_2[..., 2:])

        # calculate width and height of the minimun closed convex surface
        enclose_wh = tf.maximum(enclose_right_down - enclose_left_up, 0.0)

        # calculate enclosed diagonal distance
        enclose_diagonal = tf.reduce_sum(tf.square(enclose_wh), axis=-1)

        # calculate diou
        diou = iou - 1.0 * center_distance / enclose_diagonal

        return diou

    def box_ciou(self, boxes_1, boxes_2):
        """
        calculate regression loss using ciou
        :param boxes_1: boxes_1 shape is [x, y, w, h]
        :param boxes_2: boxes_2 shape is [x, y, w, h]
        :return:
        """
        # calculate center distance
        center_distance = tf.reduce_sum(tf.square(boxes_1[..., :2] - boxes_2[..., :2]), axis=-1)

        v = 4 * tf.square(tf.math.atan2(boxes_1[..., 2], boxes_1[..., 3]) - tf.math.atan2(boxes_2[..., 2], boxes_2[..., 3])) / (math.pi * math.pi)

        # transform [x, y, w, h] to [x_min, y_min, x_max, y_max]
        boxes_1 = tf.concat([boxes_1[..., :2] - boxes_1[..., 2:] * 0.5,
                             boxes_1[..., :2] + boxes_1[..., 2:] * 0.5], axis=-1)
        boxes_2 = tf.concat([boxes_2[..., :2] - boxes_2[..., 2:] * 0.5,
                             boxes_2[..., :2] + boxes_2[..., 2:] * 0.5], axis=-1)
        boxes_1 = tf.concat([tf.minimum(boxes_1[..., :2], boxes_1[..., 2:]),
                             tf.maximum(boxes_1[..., :2], boxes_1[..., 2:])], axis=-1)
        boxes_2 = tf.concat([tf.minimum(boxes_2[..., :2], boxes_2[..., 2:]),
                             tf.maximum(boxes_2[..., :2], boxes_2[..., 2:])], axis=-1)

        # calculate area of boxes_1 boxes_2
        boxes_1_area = (boxes_1[..., 2] - boxes_1[..., 0]) * (boxes_1[..., 3] - boxes_1[..., 1])
        boxes_2_area = (boxes_2[..., 2] - boxes_2[..., 0]) * (boxes_2[..., 3] - boxes_2[..., 1])

        # calculate the two corners of the intersection
        left_up = tf.maximum(boxes_1[..., :2], boxes_2[..., :2])
        right_down = tf.minimum(boxes_1[..., 2:], boxes_2[..., 2:])

        # calculate area of intersection
        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]

        # calculate union area
        union_area = boxes_1_area + boxes_2_area - inter_area

        # calculate iou
        iou = inter_area / union_area

        # calculate the upper left and lower right corners of the minimum closed convex surface
        enclose_left_up = tf.minimum(boxes_1[..., :2], boxes_2[..., :2])
        enclose_right_down = tf.maximum(boxes_1[..., 2:], boxes_2[..., 2:])

        # calculate width and height of the minimun closed convex surface
        enclose_wh = tf.maximum(enclose_right_down - enclose_left_up, 0.0)

        # calculate enclosed diagonal distance
        enclose_diagonal = tf.reduce_sum(tf.square(enclose_wh), axis=-1)

        # calculate diou
        diou = iou - 1.0 * center_distance / enclose_diagonal

        # calculate param v and alpha to CIoU
        alpha = v / (1.0 - iou + v)

        # calculate ciou
        ciou = diou - alpha * v

        return ciou

    def focal(self, target, actual, alpha=0.25, gamma=2):
        focal_loss = tf.abs(alpha + target - 1) * tf.pow(tf.abs(target - actual), gamma)
        return focal_loss