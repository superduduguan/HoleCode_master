import cv2
import os
import numpy as np
import json
import random
from model import Model
import matplotlib as plt
import tensorflow as tf
import time

INPUT_SIZE_W = 80
INPUT_SIZE_H = 80
IMG_SIZE_W = 100
IMG_SIZE_H = 100
POOLING_SCALE = 16

IMAGE_DIR = '/home/vision-02/Hole_Detection/Hole_Data/images/'
LABEL_DIR = '/home/vision-02/Hole_Detection/Hole_Data/labels_clean/'

THRESHOLD = 0.7
VISUALIZE = 0.6

from tensorflow.python.platform import gfile

def load_image(name, norm=True):
    img = cv2.imread(name)
    if img is None:
        print(name)
        exit()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if norm:
        img = img.astype(np.float32) / 255
    else:
        img = img.astype(np.float32)
    return img


sess = tf.Session()
with gfile.FastGFile(r'C:\Users\pc\Desktop\HoleCode\HolePosition\output_graph.pb', 'rb') as f:

    graph_def = tf.GraphDef()
    print('a')
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()

    tf.import_graph_def(graph_def, name='')

sess.run(tf.global_variables_initializer())
input_x = sess.graph.get_tensor_by_name('input/img_in:0')
op = sess.graph.get_tensor_by_name('HoleDetection/LocationResult:0')

img = load_image(r'C:\Users\pc\Desktop\HoleCode\v2.4.1\pos\a_8_24_1_43.bmp')
img = img[10:90, 10:90]
_input = np.expand_dims(img, 0)


ret = sess.run(op,  feed_dict={input_x: _input})

print(ret) # this is our output
