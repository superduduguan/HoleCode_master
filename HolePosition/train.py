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

os.environ['CUDA_VIDIBLE_DEVICES'] = '0'

# Configurations and hyper-params
IMAGE_DIR = '/home/vision-02/Hole_Detection/Hole_Data/images/'
LABEL_DIR = '/home/vision-02/Hole_Detection/Hole_Data/labels_clean/'
LOGDIR = "logs/%d%02d%02d_%02d%02d/" %(
    now.year, now.month, now.day, now.hour, now.minute)

INPUT_SIZE_W = 80 # 1224
INPUT_SIZE_H = 80 # 512
IMG_SIZE_W = 100
IMG_SIZE_H = 100 # 512
POOLING_SCALE = 16

VAL_RATIO = 0.2

WEIGHT_DECAY = 5e-4
BASE_LR = 1e-3
BATCH_SIZE = 32
EPOCH = 80
EPOCH_SIZE = 5274//BATCH_SIZE
LR_DECAY = 0.9
LR_DECAY_FREQ = 2
GPU_MEMORY_FRACTION = 0.3  #1.0


# Prepare the dataset
print ('--Preparing Dataset')
dataset = DataGenerator(image_dir=IMAGE_DIR,
                        label_dir=LABEL_DIR,
                        in_size_h=INPUT_SIZE_H,
                        in_size_w=INPUT_SIZE_W,
                        img_size_h=IMG_SIZE_H,
                        img_size_w=IMG_SIZE_W,
                        pool_scale=POOLING_SCALE,
                        val_ratio=VAL_RATIO)
dataset._create_train_table()
dataset._randomize()
dataset._create_sets()

# Build the model and train
print ('--Initializing the model')


model = Model(dataset=dataset,
              logdir=LOGDIR,
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
