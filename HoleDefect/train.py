"""
Set Configurations and Hyperparameters.
Launch the training process.
"""
import os
from datagen import DataGenerator
from model import Model
from datetime import datetime
now = datetime.now()
import cv2
import numpy as np


# Configurations and hyper-params
IMAGE_DIR = 'C:\\Users\\pc\\Desktop\\HoleCode\\Normalized_Data\\train\\' #train
# LABEL_DIR = '/home/vision-02/Hole_Detection/Hole_Data/labels_clean/'
LOGDIR = "logs/%d%02d%02d_%02d%02d/" %(
    now.year, now.month, now.day, now.hour, now.minute)

INPUT_SIZE_W = 48 # 1224
INPUT_SIZE_H = 48 # 512
POOLING_SCALE = 16

VAL_RATIO = 0.2

WEIGHT_DECAY = 4e-5
BASE_LR = 1e-2
BATCH_SIZE = 256
EPOCH = 250
EPOCH_SIZE = 10421//BATCH_SIZE
LR_DECAY = 0.9
LR_DECAY_FREQ = 2
GPU_MEMORY_FRACTION = 0.3  #1.0

FOLD_NUM = 5


# Prepare the dataset
for i in range(FOLD_NUM):
    print ('--Preparing Dataset')
    dataset = DataGenerator(image_dir=IMAGE_DIR,
                            in_size_h=INPUT_SIZE_H,
                            in_size_w=INPUT_SIZE_W,
                            pool_scale=POOLING_SCALE,
                            val_ratio=VAL_RATIO,
                            val_group=i)
    dataset._create_train_table()
    dataset._randomize()
    dataset._create_sets()

    # Build the model and train
    print ('--Initializing the model')


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
