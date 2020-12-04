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
    image_path = "/home/chenwei/HDD/Project/datasets/object_detection/VOC2028/JPEGImages/000000.jpg"
    image = cv2.imread(image_path)

    input_shape = (416, 416)
    image_shape = image.shape[:2]
    image_normal = process(image, input_shape)

    input = tf.placeholder(tf.float32,[1, input_shape[0], input_shape[1], 3])

    network = Network(is_train=False)
    output = network.inference(input)

    checkpoints = "./checkpoints/model.ckpt-15000"
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, checkpoints)
        bboxes, obj_probs, class_probs = sess.run(output, feed_dict={input: image_normal})

    bboxes, scores, class_max_index = postprocess(bboxes, obj_probs, class_probs, image_shape=image_shape)

    image_resized = cv2.resize(image, (416, 416), interpolation=0)
    img_detection = visualization(image_resized, bboxes, scores, class_max_index, model_params["classes"])
    cv2.imshow("result", img_detection)
    cv2.waitKey(0)

if __name__ == "__main__":
    predict_image()