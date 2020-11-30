# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2020/11/01
# @Author  : Oscar Chen
# @Email   : 530824679@qq.com
# @File    : dataset.py
# Description :preprocess data
# --------------------------------------

import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2
import os
import math
import numpy as np
import tensorflow as tf
from xml.etree import ElementTree as ET
from cfg.config import path_params, model_params, classes_map
from utils.process_utils import *

class Dataset(object):
    def __init__(self):
        self.data_path = path_params['data_path']
        self.image_height = model_params['image_height']
        self.image_width = model_params['image_width']
        self.iou_threshold = model_params['iou_threshold']
        self.strides = model_params['strides']
        self.anchors = model_params['anchors']
        self.anchor_per_scale = model_params['anchor_per_scale']
        self.class_num = len(model_params['classes'])
        self.max_bbox_per_scale = model_params['max_bbox_per_scale']
        self.feature_map_sizes = [np.array([self.image_height, self.image_width]) // stride for stride in self.strides]

    def load_image(self, image_num):
        image_path = os.path.join(self.data_path, 'JPEGImages', image_num + '.jpg')
        if not os.path.exists(image_path):
            raise KeyError("%s does not exist ... " %image_path)
        image = cv2.imread(image_path)

        self.h_ratio = 1.0 * self.image_height / image.shape[0]
        self.w_ratio = 1.0 * self.image_width / image.shape[1]

        image = cv2.resize(image, (self.image_height, self.image_width), interpolation=cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = image / 255.0 * 2 - 1

        return image

    def load_label(self, image_num):
        label_path = os.path.join(self.data_path, 'Annotations', image_num + '.xml')
        if not os.path.exists(label_path):
            raise KeyError("%s does not exist ... " %label_path)

        tree = ET.parse(label_path)
        root = tree.getroot()

        bboxes = []
        # 得到某个xml_file文件中所有的object
        objects = root.findall('object')
        for object in objects:
            bndbox = object.find('bndbox')
            xmin = bndbox.find('xmin').text
            ymin = bndbox.find('ymin').text
            xmax = bndbox.find('xmax').text
            ymax = bndbox.find('ymax').text

            # 将原始样本的标定转换为resize后的图片的标定,按照等比例转换的方式,从0开始索引
            x1 = max(min(float(xmin) * self.w_ratio, self.image_width - 1), 0)
            y1 = max(min(float(ymin) * self.h_ratio, self.image_height - 1), 0)
            x2 = max(min(float(xmax) * self.w_ratio, self.image_width - 1), 0)
            y2 = max(min(float(ymax) * self.h_ratio, self.image_height - 1), 0)

            # 将类别由字符串转换为对应的int数
            class_index = self.cls_type_to_id(object.find('name').text.lower().strip())

            box = [x1, y1, x2, y2, class_index]
            bboxes.append(box)

        return np.array(bboxes)

    def cls_type_to_id(self, data):
        type = data[1]
        if type not in classes_map.keys():
            print("class is %s", type)
            return -1
        return classes_map[type]

    def preprocess_true_boxes(self, labels, input_height, input_width, anchors, num_classes):
        """
        preprocess true boxes to train input format
        :param labels: numpy.ndarray of shape [num, 5]
                       shape[0]: the number of labels in each image.
                       shape[1]: x_min, y_min, x_max, y_max, class_index
        :param input_height: the shape of input image height
        :param input_width: the shape of input image width
        :param anchors: array, shape=[9, 2]
                        shape[0]: the number of anchors
                        shape[1]: width, height
        :param num_classes: the number of class
        :return: y_true shape is [feature_height, feature_width, per_anchor_num, 5 + num_classes]
        """
        input_shape = np.array([input_height, input_width], dtype=np.int32)
        num_layers = len(anchors) // 3
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        feature_map_sizes = [input_shape // 32, input_shape // 16, input_shape // 8]

        y_true_13 = np.zeros(shape=[feature_map_sizes[0][0], feature_map_sizes[0][1], 3, 5 + num_classes], dtype=np.float32)
        y_true_26 = np.zeros(shape=[feature_map_sizes[1][0], feature_map_sizes[1][1], 3, 5 + num_classes], dtype=np.float32)
        y_true_52 = np.zeros(shape=[feature_map_sizes[2][0], feature_map_sizes[2][1], 3, 5 + num_classes], dtype=np.float32)
        y_true = [y_true_13, y_true_26, y_true_52]

        # convert boxes from (min_x, min_y, max_x, max_y) to (x, y, w, h)
        boxes_xy = (labels[:, 0:2] + labels[:, 2:4]) / 2
        boxes_wh = labels[:, 2:4] - labels[:, 0:2]
        true_boxes = np.concatenate([boxes_xy, boxes_wh], axis=-1)

        # [N, 1, 2]
        valid_mask = boxes_wh[:, 0] > 0
        wh = boxes_wh[valid_mask]
        wh = np.expand_dims(wh, -2)

        boxes_max = wh / 2.
        boxes_min = - wh / 2.
        anchors_max = anchors / 2.
        anchors_min = - anchors / 2.

        # [N, 1, 2] & [9, 2] ==> [N, 9, 2]
        intersect_mins = np.maximum(boxes_min, anchors_min)
        intersect_maxs = np.minimum(boxes_max, anchors_max)
        # [N, 9, 2]
        intersect_wh = np.maximum(intersect_maxs - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[:, 0] * anchors[:, 1]
        # [N, 9]
        iou = intersect_area / (box_area + anchor_area - intersect_area + 1e-10)

        # Find best anchor for each true box [N]
        best_anchor = np.argmax(iou, axis=-1)

        for t, n in enumerate(best_anchor):
            for l in range(num_layers):
                if n not in anchor_mask[l]: continue
                i = np.floor(true_boxes[t, 0] / input_shape[0] * feature_map_sizes[l][0]).astype('int32')
                j = np.floor(true_boxes[t, 1] / input_shape[1] * feature_map_sizes[l][1]).astype('int32')
                k = anchor_mask[l].index(n)
                c = labels[t][4].astype('int32')
                # smooth labels
                onehot = np.zeros(self.class_num, dtype=np.float)
                onehot[c] = 1.0
                uniform_distribution = np.full(self.class_num, 1.0 / self.class_num)
                deta = 0.01
                smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution

                y_true[l][j, i, k, 0:4] = true_boxes[t, 0:4]
                y_true[l][j, i, k, 4] = 1
                y_true[l][j, i, k, 5:] = smooth_onehot

        return y_true_13, y_true_26, y_true_52