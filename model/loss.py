# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2020/11/01
# @Author  : Oscar Chen
# @Email   : 530824679@qq.com
# @File    : loss.py
# Description :Yolo_v2 Loss损失函数.
# --------------------------------------

import math
import numpy as np
import tensorflow as tf
from cfg.config import model_params, solver_params

class Loss(object):
    def __init__(self, predicts, labels, scope='loss'):
        """
        :param predicts:网络的输出 anchor_num * (5 + class_num)
        :param labels:标签信息
        :param scope:命名loss
        """
        self.batch_size = solver_params['batch_size']
        self.image_height = model_params['image_height']
        self.image_width = model_params['image_width']
        self.anchor_per_scale = model_params['anchor_per_scale']
        self.class_num = len(model_params['classes'])
        self.iou_threshold = model_params['iou_threshold']
        self.class_scale = model_params['class_scale']
        self.object_scale = model_params['object_scale']
        self.noobject_scale = model_params['noobject_scale']
        self.coord_scale = model_params['coord_scale']

    def calc_loss(self, pred_sbbox, pred_mbbox, pred_lbbox, label_sbbox, label_mbbox, label_lbbox):
        with tf.name_scope('smaller_box_loss'):
            loss_sbbox = self.loss_layer(self.conv_sbbox, pred_sbbox, label_sbbox)

        with tf.name_scope('medium_box_loss'):
            loss_mbbox = self.loss_layer(self.conv_mbbox, pred_mbbox, label_mbbox)

        with tf.name_scope('bigger_box_loss'):
            loss_lbbox = self.loss_layer(self.conv_lbbox, pred_lbbox, label_lbbox)

        with tf.name_scope('ciou_loss'):
            ciou_loss = loss_sbbox[0] + loss_mbbox[0] + loss_lbbox[0]

        with tf.name_scope('conf_loss'):
            conf_loss = loss_sbbox[1] + loss_mbbox[1] + loss_lbbox[1]

        with tf.name_scope('prob_loss'):
            prob_loss = loss_sbbox[2] + loss_mbbox[2] + loss_lbbox[2]

        return ciou_loss, conf_loss, prob_loss

    def loss_layer(self, feature_map, pred, label, bboxes):
        conv_shape = tf.shape(feature_map)
        feature_map_height = conv_shape[1]
        feature_map_width = conv_shape[2]

        predicts = tf.reshape(feature_map, [-1, feature_map_height, feature_map_width, self.anchor_per_scale, (5 + self.class_num)])
        conv_conf = predicts[:, :, :, :, 4:5]
        conv_prob = predicts[:, :, :, :, 5:]

        pred_xywh = pred[:, :, :, :, 0:4]
        pred_conf = pred[:, :, :, :, 4:5]

        label_xywh = label[:, :, :, :, 0:4]
        respond_bbox = label[:, :, :, :, 4:5]
        label_prob = label[:, :, :, :, 5:]

        # coord loss label_wh normalzation 0-1
        bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / self.image_height / self.image_width
        ciou = tf.expand_dims(self.box_ciou(pred_xywh, label_xywh), axis=-1)
        ciou_loss = respond_bbox * bbox_loss_scale * (1 - ciou) * self.coord_scale

        # confidence loss
        iou = self.bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
        best_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)
        respond_bgd = (1.0 - respond_bbox) * tf.cast( best_iou < self.iou_threshold, tf.float32)

        # Focal loss分配了权重，注释掉原本的object_scale和noobject_scale
        conf_focal = self.focal(respond_bbox, pred_conf)
        object_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_conf)# * self.object_scale
        noobject_loss = respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_conf)# * self.noobject_scale
        conf_loss = conf_focal * (object_loss + noobject_loss)

        # prob loss
        prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_prob) * self.class_scale

        ciou_loss = tf.reduce_mean(tf.reduce_sum(ciou_loss, axis=[1, 2, 3, 4]))
        conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1, 2, 3, 4]))
        prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1, 2, 3, 4]))

        return ciou_loss, conf_loss, prob_loss

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
