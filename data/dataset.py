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
from data.augmentation import *

class Dataset(object):
    def __init__(self):
        self.data_path = path_params['data_path']
        self.input_height = model_params['input_height']
        self.input_width = model_params['input_width']
        self.iou_threshold = model_params['iou_threshold']
        self.anchors = model_params['anchors']
        self.anchor_per_scale = model_params['anchor_per_scale']
        self.class_num = len(model_params['classes'])
        self.max_bbox_per_scale = model_params['max_bbox_per_scale']

    def load_image(self, image_num):
        image_path = os.path.join(self.data_path, 'JPEGImages', image_num + '.jpg')
        image_path_1 = os.path.join(self.data_path, 'JPEGImages', image_num + '.JPG')

        if os.path.exists(image_path):
            image = cv2.imread(image_path)
        elif os.path.exists(image_path_1):
            image = cv2.imread(image_path_1)
        else:
            raise KeyError("%s does not exist ... " %image_path)

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

            # 将类别由字符串转换为对应的int数
            class_index = self.cls_type_to_id(object.find('name').text.lower().strip())

            box = [xmin, ymin, xmax, ymax, class_index]
            bboxes.append(box)

        return bboxes

    def cls_type_to_id(self, data):
        type = data
        if type not in classes_map.keys():
            print("class is:{}".format(type))
            return -1
        return classes_map[type]

    def preprocess_data(self, image, boxes, input_height, input_width):
        image = np.array(image)

        image, boxes = random_horizontal_flip(image, boxes)
        image, labels = random_crop(image, boxes)
        image, boxes = random_translate(image, boxes)
        image = random_color_distort(image)

        image_rgb = cv2.cvtColor(np.copy(image), cv2.COLOR_BGR2RGB).astype(np.float32)
        image_rgb, labels = letterbox_resize(image_rgb, (input_height, input_width), np.copy(labels), interp=0)
        image_norm = image_rgb / 255.

        # labels 去除空标签
        valid = (np.sum(boxes, axis=-1) > 0).tolist()
        boxes = boxes[valid]

        y_true_13, y_true_26, y_true_52 = self.preprocess_true_boxes(boxes, input_height, input_width, self.anchors, self.class_num)

        return image_norm, y_true_13, y_true_26, y_true_52

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
        anchors = np.array(anchors)
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        feature_map_sizes = [input_shape // 32, input_shape // 16, input_shape // 8]

        # labels 去除空标签
        valid = (np.sum(labels, axis=-1) > 0).tolist()
        labels = labels[valid]

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

        ratio_dict = {1.: 8., 2.: 16., 3.: 32.}
        for n, idx in enumerate(best_anchor):
            # idx: 0,1,2 ==> 2; 3,4,5 ==> 1; 6,7,8 ==> 0
            feature_map_group = 2 - idx // 3
            # scale ratio: 0,1,2 ==> 8; 3,4,5 ==> 16; 6,7,8 ==> 32
            ratio = ratio_dict[np.ceil((idx + 1) / 3.)]

            i = int(np.floor(true_boxes[n, 0] / ratio))
            j = int(np.floor(true_boxes[n, 1] / ratio))
            k = anchor_mask[feature_map_group].index(idx)
            c = labels[n][4].astype('int32')
            #print(feature_map_group, '|', j, i, k, c)

            # smooth labels
            onehot = np.zeros(self.class_num, dtype=np.float)
            onehot[c] = 1.0
            uniform_distribution = np.full(self.class_num, 1.0 / self.class_num)
            deta = 0.01
            smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution

            y_true[feature_map_group][j, i, k, 0:4] = true_boxes[n, 0:4]
            y_true[feature_map_group][j, i, k, 4] = 1
            y_true[feature_map_group][j, i, k, 5:] = smooth_onehot

        return y_true_13, y_true_26, y_true_52
