# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2020/11/01
# @Author  : Oscar Chen
# @Email   : 530824679@qq.com
# @File    : tfrecord.py
# Description :create and parse tfrecord
# --------------------------------------

import os
import numpy as np
import tensorflow as tf
from data.dataset import Dataset
from cfg.config import path_params, model_params, solver_params, classes_map
from data.augmentation import *

class TFRecord(object):
    def __init__(self):
        self.data_path = path_params['data_path']
        self.tfrecord_dir = path_params['tfrecord_dir']
        self.train_tfrecord_name = path_params['train_tfrecord_name']
        self.test_tfrecord_name = path_params['test_tfrecord_name']
        self.input_width = model_params['input_width']
        self.input_height = model_params['input_height']
        self.channels = model_params['channels']
        self.class_num = len(model_params['classes'])
        self.batch_size = solver_params['batch_size']
        self.dataset = Dataset()

    def _int64_feature(self, value):
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _float_feature(self, value):
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def _bytes_feature(self, value):
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def create_tfrecord(self):
        # 获取作为训练验证集的图片序列
        trainval_path = os.path.join(self.data_path, 'ImageSets', 'Main', 'trainval.txt')

        tf_file = os.path.join(self.tfrecord_dir, self.train_tfrecord_name)
        if os.path.exists(tf_file):
            os.remove(tf_file)

        writer = tf.python_io.TFRecordWriter(tf_file)
        with open(trainval_path, 'r') as read:
            lines = read.readlines()
            for line in lines:
                num = line[0:-1]
                image = self.dataset.load_image(num)
                boxes = self.dataset.load_label(num)

                if len(boxes) == 0:
                    continue

                image, boxes = letterbox_resize(image, np.array(boxes, dtype=np.float32), self.input_height, self.input_width)
                while boxes.shape[0] < 300:
                    boxes = np.append(boxes, [[0.0, 0.0, 0.0, 0.0, 0.0]], axis=0)
                boxes = np.array(boxes, dtype=np.float32)

                image_string = image.tobytes()
                boxes_string = boxes.tobytes()

                example = tf.train.Example(features=tf.train.Features(
                    feature={
                        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_string])),
                        'bbox': tf.train.Feature(bytes_list=tf.train.BytesList(value=[boxes_string])),
                    }))
                writer.write(example.SerializeToString())
        writer.close()
        print('Finish trainval.tfrecord Done')

    def parse_single_example(self, serialized_example):
        """
        :param file_name:待解析的tfrecord文件的名称
        :return: 从文件中解析出的单个样本的相关特征，image, label
        """
        features = tf.parse_single_example(
            serialized_example,
            features={
                'image': tf.FixedLenFeature([], tf.string),
                'bbox': tf.FixedLenFeature([], tf.string),
            })

        image = features['image']
        label = features['bbox']

        # 进行解码
        tf_image = tf.decode_raw(image, tf.uint8)
        tf_label = tf.decode_raw(label, tf.float32)

        # 转换为网络输入所要求的形状
        tf_image = tf.reshape(tf_image, [self.input_height, self.input_width, self.channels])
        tf_label = tf.reshape(tf_label, [300, 5])

        # preprocess
        tf_image, y_true_13, y_true_26, y_true_52 = tf.py_func(self.dataset.preprocess_data, inp=[tf_image, tf_label, self.input_height, self.input_width], Tout=[tf.float32, tf.float32, tf.float32, tf.float32])

        return tf_image, y_true_13, y_true_26, y_true_52

    def create_dataset(self, filenames, batch_size=1, is_shuffle=False):
        """
        :param filenames: record file names
        :param batch_size: batch size
        :param is_shuffle: whether shuffle
        :param n_repeats: number of repeats
        :return:
        """
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(self.parse_single_example, num_parallel_calls=4)
        dataset = dataset.repeat()
        dataset = dataset.prefetch(batch_size)
        if is_shuffle:
            dataset = dataset.shuffle(20*batch_size)
        dataset = dataset.batch(batch_size)

        return dataset

if __name__ == '__main__':
    tfrecord = TFRecord()
    tfrecord.create_tfrecord()

    # import cv2
    # import matplotlib.pyplot as plt
    # record_file = '../tfrecord/train.tfrecord'
    # data_train = tfrecord.create_dataset(record_file, batch_size=4, is_shuffle=False)
    # # data_train = tf.data.TFRecordDataset(record_file)
    # # data_train = data_train.map(tfrecord.parse_single_example)
    # iterator = data_train.make_one_shot_iterator()
    # batch_image, y_true_13, y_true_26, y_true_52 = iterator.get_next()
    #
    # with tf.Session() as sess:
    #     for i in range(20):
    #         try:
    #             images_, y_true_13_, y_true_26_, y_true_52_ = sess.run([batch_image, y_true_13, y_true_26, y_true_52])
    #
    #             # for images_i, y_true_13_i, y_true_26_i, y_true_52_i in zip(images_, y_true_13_, y_true_26_, y_true_52_):
    #
    #                 # boxes_ = boxes_[..., 0:4]
    #                 # valid = (np.sum(boxes_, axis=-1) > 0).tolist()
    #                 # print([int(idx) for idx in boxes_[:, 0][valid].tolist()])
    #                 # for box in boxes_[:, 0:4][valid].tolist():
    #                 #     cv2.rectangle(images_i, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
    #                 # cv2.imshow("image", images_i)
    #                 # cv2.waitKey(0)
    #             print(images_.shape, y_true_13_.shape)
    #         except tf.errors.OutOfRangeError:
    #             print("Done!!!")
    #             break
