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
        self.image_width = model_params['image_width']
        self.image_height = model_params['image_height']
        self.channels = model_params['channels']
        self.grid_height = model_params['grid_height']
        self.grid_width = model_params['grid_width']
        self.class_num = model_params['num_classes']
        self.batch_size = solver_params['batch_size']
        self.dataset = Dataset()

    # 数值形式的数据,首先转换为string,再转换为int形式进行保存
    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    # 数组形式的数据,首先转换为string,再转换为二进制形式进行保存
    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def create_tfrecord(self):
        # 获取作为训练验证集的图片序列
        trainval_path = os.path.join(self.data_path, 'ImageSets', 'Main', 'trainval.txt')

        tf_file = os.path.join(self.tfrecord_dir, self.train_tfrecord_name)
        if not os.path.exists(tf_file):
            # 循环写入每一帧点云转换的bev和标签到tfrecord文件
            writer = tf.python_io.TFRecordWriter(tf_file)
            with open(trainval_path, 'r') as read:
                lines = read.readlines()
                for line in lines:
                    num = line[0:-1]
                    image = self.dataset.load_image(num)
                    label = self.dataset.load_label(num)

                    image_string = image.tostring()
                    label_string = label.tostring()

                    example = tf.train.Example(features=tf.train.Features(
                        feature={
                            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_string])),
                            'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_string]))
                        }))
                    writer.write(example.SerializeToString())
            writer.close()
            print('Finish trainval.tfrecord Done')

    def parse_single_example(self, file_name):
        """
        :param file_name:待解析的tfrecord文件的名称
        :return: 从文件中解析出的单个样本的相关特征，image, label
        """
        tfrecord_file = os.path.join(self.tfrecord_dir, self.train_tfrecord_name)

        # 定义解析TFRecord文件操作
        reader = tf.TFRecordReader()

        # 创建样本文件名称队列
        filename_queue = tf.train.string_input_producer([tfrecord_file])

        # 解析单个样本文件
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
            serialized_example,
            features={
                'image': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.string)
            })


        image = features['image']
        label = features['label']

        return image, label

    def parse_batch_examples(self, file_name):
        """
        :param file_name:待解析的tfrecord文件的名称
        :return: 解析得到的batch_size个样本
        """
        batch_size = self.batch_size
        min_after_dequeue = 100
        num_threads = 8
        capacity = min_after_dequeue + 3 * batch_size


        image, label = self.parse_single_example(file_name)
        image_batch, label_batch = tf.train.shuffle_batch([image, label],
                                                          batch_size=batch_size,
                                                          num_threads=num_threads,
                                                          capacity=capacity,
                                                          min_after_dequeue=min_after_dequeue)

        # 进行解码
        image_batch = tf.decode_raw(image_batch, tf.float32)
        label_batch = tf.decode_raw(label_batch, tf.float32)

        # 转换为网络输入所要求的形状
        image_batch = tf.reshape(image_batch, [self.batch_size, self.image_height, self.image_width, self.channels])
        label_batch = tf.reshape(label_batch, [self.batch_size, self.grid_height, self.grid_width, 7 + self.class_num])

        return image_batch, label_batch

if __name__ == '__main__':
    tfrecord = TFRecord()
    tfrecord.create_tfrecord()

    # file = './tfrecord/train.tfrecord'
    # tfrecord = TFRecord()
    # batch_example, batch_label = tfrecord.parse_batch_examples(file)
    # with tf.Session() as sess:
    #
    #     init_op = tf.global_variables_initializer()
    #     sess.run(init_op)
    #
    #     coord = tf.train.Coordinator()
    #     threads = tf.train.start_queue_runners(coord=coord)
    #     for i in range(1):
    #         example, label = sess.run([batch_example, batch_label])
    #         print(label)
    #         print(label.astype(np.float32))
    #         box = label[0, ]
    #         # cv2.imshow('w', example[0, :, :, :])
    #         # cv2.waitKey(0)
    #         print(np.shape(example), np.shape(label))
    #     # cv2.imshow('img', example)
    #     # cv2.waitKey(0)
    #     # print(type(example))
    #     coord.request_stop()
    #     # coord.clear_stop()
    #     coord.join(threads)