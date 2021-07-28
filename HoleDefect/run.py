"""
Set Configurations and Hyperparameters.
Launch the training process.
"""
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from datagen import DataGenerator
from model import Model
from datetime import datetime
now = datetime.now()
import cv2
import numpy as np




cur_path = os.path.abspath(__file__)
cur_dir = os.path.dirname(cur_path)
IMAGE_DIR = os.path.join(cur_dir, '..\\example\\dataset\\train\\')
TEXTDIR = os.path.join(cur_dir, 'txtdir')
LOGDIR = "logs/%d%02d%02d_%02d%02d/" % (
    now.year, now.month, now.day, now.hour, now.minute)

INPUT_SIZE_W = 48  # 1224
INPUT_SIZE_H = 48  # 512
POOLING_SCALE = 16

VAL_RATIO = 0.2

WEIGHT_DECAY = 4e-5
BASE_LR = 1e-2
BATCH_SIZE = 32
EPOCH = 30
EPOCH_SIZE = 2017//BATCH_SIZE  #only train
LR_DECAY = 0.9
LR_DECAY_FREQ = 2
GPU_MEMORY_FRACTION = 0.3  #1.0

FOLD_NUM = 5


# Prepare the dataset
for i in range(FOLD_NUM):
    print('--Preparing Dataset')
    dataset = DataGenerator(image_dir=IMAGE_DIR,
                            in_size_h=INPUT_SIZE_H,
                            in_size_w=INPUT_SIZE_W,
                            pool_scale=POOLING_SCALE,
                            val_ratio=VAL_RATIO,
                            val_group=i,
                            txtdir=TEXTDIR)
    dataset._create_train_table()
    dataset._randomize()
    dataset._create_sets()

    # Build the model and train
    print('--Initializing the model')


    model = Model(dataset=dataset,
                  logdir=LOGDIR + str(i) + '/',
                  in_size_h=INPUT_SIZE_H,
                  in_size_w=INPUT_SIZE_W,
                  pool_scale=POOLING_SCALE,
                  weight_decay=WEIGHT_DECAY,
                  base_lr=BASE_LR,
                  epoch=EPOCH,
                  epoch_size=EPOCH_SIZE,
                  lr_decay=LR_DECAY,
                  lr_decay_freq=LR_DECAY_FREQ,
                  batch_size=BATCH_SIZE,
                  gpu_memory_fraction=GPU_MEMORY_FRACTION)

    model.BuildModel()
    model.train()







import os
import sys
import tensorflow as tf
slim = tf.contrib.slim
from model import Model

nums = ['0', '1', '2', '3', '4']
for num in nums:
    MODEL_PATH = os.path.join(cur_dir, LOGDIR + num + '\\model.ckpt-' + str(EPOCH-1))  

    def freeze_mobilenet(meta_file):

        tf.reset_default_graph()
        # model = AttModel(training=False,w_summary=False)
        model = Model(training=False, w_summary=False)
        model.BuildModel()

        output_node_names = ['HoleDefect/Classfication/dense/BiasAdd']
        output_pb_name = 'HoleDefect' + num + '_' + str(EPOCH) + '.pb'
        rest_var = slim.get_variables_to_restore()

        with tf.Session() as sess:
            graph = tf.get_default_graph()
            input_graph_def = graph.as_graph_def()

            saver = tf.train.Saver(rest_var)
            saver.restore(sess, meta_file)
            output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names)
            tf.train.write_graph(output_graph_def, "./", output_pb_name, as_text=False)
    freeze_mobilenet(MODEL_PATH)
print('\nfinish transforming')