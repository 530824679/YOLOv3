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

class TFRecord(object):
    def __init__(self):
        self.data_path = path_params['data_path']
        self.tfrecord_dir = path_params['tfrecord_dir']
        self.train_tfrecord_name = path_params['train_tfrecord_name']
        self.test_tfrecord_name = path_params['test_tfrecord_name']
        self.input_width = model_params['input_width']
        self.input_height = model_params['input_height']
        self.channels = model_params['channels']
        self.grid_height = model_params['grid_height']
        self.grid_width = model_params['grid_width']
        self.class_num = model_params['num_classes']
        self.batch_size = solver_params['batch_size']
        self.dataset = Dataset()

    # 数值形式的数据,首先转换为string,再转换为int形式进行保存
    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _float_feature(self, value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    # 数组形式的数据,首先转换为string,再转换为二进制形式进行保存
    def _bytes_feature(self, value):
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
                label = self.dataset.load_label(num)

                if len(label) == 0:
                    continue

                image_string = image.tobytes()
                label_string = label.tobytes()
                label_shape = label.shape

                example = tf.train.Example(features=tf.train.Features(
                    feature={
                        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_string])),
                        'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_string])),
                        'bbox_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=label_shape))
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
                'label': tf.FixedLenFeature([], tf.string),
                'label_shape': tf.FixedLenFeature(shape=(2,), dtype=tf.int64)
            })

        image = features['image']
        label = features['label']
        label_shape = features['label_shape']

        # 进行解码
        tf_image = tf.decode_raw(image, tf.float32)
        tf_label = tf.decode_raw(label, tf.float32)

        # 转换为网络输入所要求的形状
        tf_image = tf.reshape(tf_image, [self.input_height, self.input_width, self.channels])
        tf_label = tf.reshape(tf_label, label_shape)

        # preprocess
        tf_image = tf_image / 255
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
    #tfrecord.create_tfrecord()

    import cv2
    import utils.visualize as v

    record_file = './tfrecord/train.tfrecord'
    data_train = tfrecord.create_dataset(record_file, batch_size=2, is_shuffle=False, n_repeats=20)
    # data_train = tf.data.TFRecordDataset(record_file)
    # data_train = data_train.map(tfrecord.parse_single_example)
    iterator = data_train.make_one_shot_iterator()
    batch_image, y_true_13, y_true_26, y_true_52 = iterator.get_next()

    with tf.Session() as sess:
        for i in range(20):
            try:
                image, true_19, true_38, true_76 = sess.run([batch_image, y_true_13, y_true_26, y_true_52])

                # for boxes in label:
                #     v.draw_rotated_box(image, int(boxes[0]), int(boxes[1]), int(boxes[2]), int(boxes[3]), boxes[5],
                #                        (255, 0, 0))
                # cv2.imshow("image", image)
                # cv2.waitKey(0)
                print(np.shape(image))
            except tf.errors.OutOfRangeError:
                print("Done!!!")
                break
