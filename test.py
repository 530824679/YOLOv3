import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2
import tensorflow as tf
from utils.process_utils import *
from model.network import Network
from cfg.config import *

def predict_image():
    image_path = "/home/chenwei/HDD/Project/datasets/object_detection/VOCdevkit/VOC2007/JPEGImages/000066.jpg"
    image = cv2.imread(image_path)
    image_size = image.shape[:2]
    input_shape = [model_params['input_height'], model_params['input_width']]
    image_data = preporcess(image, input_shape)
    image_data = image_data[np.newaxis, ...]

    input = tf.placeholder(shape=[1, input_shape[0], input_shape[1], 3], dtype=tf.float32)

    model = Network(len(model_params['classes']), model_params['anchors'], is_train=False)
    with tf.variable_scope('yolov3'):
        logits = model.build_network(input)
        output = model.inference(logits)

    checkpoints = "/home/chenwei/HDD/Project/YOLOv3/weights/yolov3.ckpt"
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, checkpoints)
        bboxes, obj_probs, class_probs = sess.run(output, feed_dict={input: image_data})

    bboxes, scores, class_max_index = postprocess(bboxes, obj_probs, class_probs, image_shape=image_size, input_shape=input_shape)

    resize_ratio = min(input_shape[1] / image_size[1], input_shape[0] / image_size[0])
    dw = (input_shape[1] - resize_ratio * image_size[1]) / 2
    dh = (input_shape[0] - resize_ratio * image_size[0]) / 2
    bboxes[:, [0, 2]] = (bboxes[:, [0, 2]] - dw) / resize_ratio
    bboxes[:, [1, 3]] = (bboxes[:, [1, 3]] - dh) / resize_ratio

    img_detection = visualization(image, bboxes, scores, class_max_index, model_params["classes"])
    cv2.imshow("result", img_detection)
    cv2.waitKey(0)

if __name__ == "__main__":
    predict_image()