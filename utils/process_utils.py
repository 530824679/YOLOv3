# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2020/11/01
# @Author  : Oscar Chen
# @Email   : 530824679@qq.com
# @File    : process_utils.py
# Description :function
# --------------------------------------
import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2
import random
import colorsys
import numpy as np
import tensorflow as tf

def total_sample(file_name):
    sample_nums = 0
    for record in tf.python_io.tf_record_iterator(file_name):
        sample_nums += 1
    return sample_nums

def calc_iou_wh(box1_wh, box2_wh):
    """
    param box1_wh (list, tuple): Width and height of a box
    param box2_wh (list, tuple): Width and height of a box
    return (float): iou
    """
    min_w = min(box1_wh[0], box2_wh[0])
    min_h = min(box1_wh[1], box2_wh[1])
    area_r1 = box1_wh[0] * box1_wh[1]
    area_r2 = box2_wh[0] * box2_wh[1]
    intersect = min_w * min_h
    union = area_r1 + area_r2 - intersect
    return intersect / union

def calculate_iou(box_1, box_2):
    """
    calculate iou
    :param box_1: (x0, y0, x1, y1)
    :param box_2: (x0, y0, x1, y1)
    :return: value of iou
    """
    bboxes1 = np.transpose(box_1)
    bboxes2 = np.transpose(box_2)

    # 计算两个box的交集：交集左上角的点取两个box的max，交集右下角的点取两个box的min
    int_ymin = np.maximum(bboxes1[0], bboxes2[0])
    int_xmin = np.maximum(bboxes1[1], bboxes2[1])
    int_ymax = np.minimum(bboxes1[2], bboxes2[2])
    int_xmax = np.minimum(bboxes1[3], bboxes2[3])

    # 计算两个box交集的wh：如果两个box没有交集，那么wh为0(按照计算方式wh为负数，跟0比较取最大值)
    int_h = np.maximum(int_ymax - int_ymin, 0.)
    int_w = np.maximum(int_xmax - int_xmin, 0.)

    # 交集面积
    intersection = int_h * int_w
    vol1 = (bboxes1[2] - bboxes1[0]) * (bboxes1[3] - bboxes1[1])  # bboxes1面积
    vol2 = (bboxes2[2] - bboxes2[0]) * (bboxes2[3] - bboxes2[1])  # bboxes2面积

    # iou=交集/并集
    iou = intersection / (vol1 + vol2 - intersection)

    return iou

def bboxes_sort(coords, scores, classes, top_k=100):
    index = np.argsort(-scores)
    classes = classes[index][:top_k]
    scores = scores[index][:top_k]
    coords = coords[index][:top_k]
    return coords, scores, classes

def bboxes_nms(classes, scores, bboxes, nms_threshold=0.5):
    keep_bboxes = np.ones(scores.shape, dtype=np.bool)
    for i in range(scores.size-1):
        if keep_bboxes[i]:
            # Computer overlap with bboxes which are following.
            overlap = calculate_iou(bboxes[i], bboxes[(i+1):])
            # Overlap threshold for keeping + checking part of the same class
            keep_overlap = np.logical_or(overlap < nms_threshold, classes[(i+1):] != classes[i])
            keep_bboxes[(i+1):] = np.logical_and(keep_bboxes[(i+1):], keep_overlap)

    idxes = np.where(keep_bboxes)
    return classes[idxes], scores[idxes], bboxes[idxes]

def non_maximum_suppression(bboxes, scores, classes, iou_threshold=0.45):
    """
    calculate the non-maximum suppression to eliminate the overlapped box
    :param classes: shape is [num, 1] classes
    :param scores: shape is [num, 1] scores
    :param bboxes: shape is [num, 4] (xmin, ymin, xmax, ymax)
    :param nms_threshold: iou threshold
    :return:
    """
    results = np.concatenate([bboxes, scores, classes], axis=-1)
    classes_in_img = list(set(results[:, 5]))
    best_results = []

    for cls in classes_in_img:
        cls_mask = (results[:, 5] == cls)
        cls_bboxes = results[cls_mask]

        while len(cls_bboxes) > 0:
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_result = cls_bboxes[max_ind]
            best_results.append(best_result)
            cls_bboxes = np.concatenate([cls_bboxes[:max_ind], cls_bboxes[max_ind + 1:]])
            overlap = calculate_iou(best_result[np.newaxis, :4], cls_bboxes[:, :4])

            weight = np.ones((len(overlap),), dtype=np.float32)
            iou_mask = overlap > iou_threshold
            weight[iou_mask] = 0.0

            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > 0.
            cls_bboxes = cls_bboxes[score_mask]

    return best_results


def soft_non_maximum_suppression(classes, scores, bboxes, sigma=0.3):
    """
    calculate the soft non-maximum suppression to eliminate the overlapped box
    :param classes: shape is [num, 1] classes
    :param scores: shape is [num, 1] scores
    :param bboxes: shape is [num, 4] (xmin, ymin, xmax, ymax)
    :param sigma: soft weight
    :return:
    """
    results = np.concatenate([bboxes, scores, classes], axis=-1)
    classes_in_img = list(set(results[:, 5]))
    best_results = []

    for cls in classes_in_img:
        cls_mask = (results[:, 5] == cls)
        cls_bboxes = results[cls_mask]

        while len(cls_bboxes) > 0:
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_result = cls_bboxes[max_ind]
            best_results.append(best_result)
            cls_bboxes = np.concatenate([cls_bboxes[:max_ind], cls_bboxes[max_ind + 1:]])
            overlap = calculate_iou(best_result[np.newaxis, :4], cls_bboxes[:, :4])

            weight = np.ones((len(overlap),), dtype=np.float32)
            weight = np.exp(-(1.0 * overlap ** 2 / sigma))

            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > 0.
            cls_bboxes = cls_bboxes[score_mask]

    return best_results

def postprocess(bboxes, obj_probs, class_probs, image_shape=(416,416), input_shape=(416, 416), threshold=0.5):
    # boxes shape——> [num, 4]
    bboxes = np.reshape(bboxes, [-1, 4])

    image_height, image_width = image_shape
    resize_ratio = min(input_shape[1] / image_width, input_shape[0] / image_height)

    dw = (input_shape[1] - resize_ratio * image_width) / 2
    dh = (input_shape[0] - resize_ratio * image_height) / 2

    bboxes[:, 0::2] = 1.0 * (bboxes[:, 0::2] - dw) / resize_ratio
    bboxes[:, 1::2] = 1.0 * (bboxes[:, 1::2] - dh) / resize_ratio
    bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, image_width)
    bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, image_height)

    bboxes = bboxes.astype(np.int32)

    # 置信度 * 类别条件概率 = 类别置信度scores
    obj_probs = np.reshape(obj_probs, [-1])
    class_probs = np.reshape(class_probs, [len(obj_probs), -1])
    class_max_index = np.argmax(class_probs, axis=1)
    class_probs = class_probs[np.arange(len(obj_probs)), class_max_index]
    scores = obj_probs * class_probs

    # 类别置信度scores > threshold的边界框bboxes留下
    keep_index = scores > threshold
    class_max_index = class_max_index[keep_index]
    scores = scores[keep_index]
    bboxes = bboxes[keep_index]

    # 排序取前400个
    class_max_index, scores, bboxes = bboxes_sort(class_max_index, scores, bboxes)

    # 计算nms
    class_max_index, scores, bboxes = bboxes_nms(class_max_index, scores, bboxes)

    return bboxes, scores, class_max_index

def preporcess(image, target_size, gt_boxes=None):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

    ih, iw = target_size
    h,  w, _ = image.shape

    scale = min(iw/w, ih/h)
    nw, nh  = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
    image_paded = image_paded / 255.

    if gt_boxes is None:
        return image_paded

    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_paded, gt_boxes

def visualization(im, bboxes, scores, cls_inds, labels, thr=0.02):
    # Generate colors for drawing bounding boxes.
    hsv_tuples = [(x / float(len(labels)), 1., 1.) for x in range(len(labels))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.

    # draw image
    imgcv = np.copy(im)
    h, w, _ = imgcv.shape
    for i, box in enumerate(bboxes):
        if scores[i] < thr:
            continue
        cls_indx = cls_inds[i]

        thick = int((h + w) / 300)
        cv2.rectangle(imgcv, (box[0], box[1]), (box[2], box[3]), colors[cls_indx], thick)
        mess = '%s: %.3f' % (labels[cls_indx], scores[i])
        if box[1] < 20:
            text_loc = (box[0] + 2, box[1] + 15)
        else:
            text_loc = (box[0], box[1] - 10)
        cv2.putText(imgcv, mess, text_loc, cv2.FONT_HERSHEY_SIMPLEX, 1e-3 * h, (255, 255, 255), thick // 3)
    cv2.imshow("test", imgcv)
    cv2.waitKey(0)