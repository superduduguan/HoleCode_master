"""
Prepare the dataset and generate batch data for training process.
"""

import numpy as np
import os
import cv2
import random
import json
from albumentations import HorizontalFlip, CenterCrop, ShiftScaleRotate, GaussNoise, \
    MedianBlur, RandomBrightnessContrast, Compose, VerticalFlip, Resize


class DataGenerator(object):
    
    def __init__(self,
                 image_dir=None,
                 label_dir=None,
                 in_size_h=80,
                 in_size_w=80,
                 img_size_h=100,
                 img_size_w=100,
                 pool_scale=16,
                 val_ratio=0.2):
        
        self.image_dir = image_dir
        self.label_dir = label_dir
        
        self.in_size_h = in_size_h
        self.in_size_w = in_size_w
        self.img_size_h = img_size_h
        self.img_size_w = img_size_w
        self.pool_scale = pool_scale
        
        self.val_ratio = val_ratio

   
    # ======= Initialize Generator ======== 
    def _create_train_table(self):
        """ 
        Scan the dataset and create table of samples
        """
        self.train_table = []
        self.label_dict = {}
        # Add samples to the whole dataset
        for filename in os.listdir(self.label_dir):
            samplename = filename[:-5]
            labels = json.load(open(self.label_dir + filename, 'r'))
            CenterPoints = labels['center']
            Radius = labels['radius']
            img_path = labels['path']
            classific = labels['class']
            try:
                image = self.load_image(self.image_dir + img_path, norm=False)
                self.label_dict[samplename] = {'image': image, 'centP': CenterPoints,
                                               'r': Radius, 'class': classific}
                self.train_table.append(samplename)
            except:
                pass
               
            
    def _randomize(self):
        """ 
        Randomize the set
        """
        random.shuffle(self.train_table)
    
    
    def _create_sets(self):
        """ 
        Select samples to construct training and validation set
        """
        num_sample = len(self.train_table)
        num_val = int(num_sample * self.val_ratio)
        self.val_set = self.train_table[:num_val]
        self.train_set = self.train_table[num_val:]
        with open('train.txt', 'w') as f:
            for i in range(len(self.train_set)):
                f.write(self.train_set[i] + '\n')
        with open('test.txt', 'w') as f:
            for i in range(len(self.val_set)):
                f.write(self.val_set[i] + '\n')
        print('--Training set: ', len(self.train_set), ' samples.')
        print('--Validation set: ', len(self.val_set), ' samples.')
    
    
    # ======= Batch Generator ========
    def _batch_generator(self, batch_size=16, normalize=True, sample_set='train'):
        """ 
        Generate batch-wise data
        """
        data_set = []
        if sample_set == 'train':
            data_set = self.train_set
        elif sample_set == 'val':
            data_set = self.val_set
            
        # Record the traverse of dataset
        sample_idx = -1
        augumation = self.Image_Keypoint_Augumation()
        while True:
            # Construct batch data containers:
            train_img = np.zeros((batch_size, self.in_size_h, self.in_size_w, 3), dtype=np.float32)
            train_gt = np.zeros((batch_size, 3), dtype=np.float32)
            
            i = 0
            while i < batch_size:
                sample_idx = (sample_idx + 1) % len(data_set)
                if sample_idx == 0 and sample_set == 'train':
                    random.shuffle(data_set)
                sample = data_set[sample_idx]
                # Load original image and mask according to sample
                img = self.label_dict[sample]['image']
                CenterPoint = self.label_dict[sample]['centP']
                Radius = self.label_dict[sample]['r']
                label = self.label_dict[sample]['class']
                # Crop a patch from original image and mask as inputs

                aug = augumation(image=img, keypoints=[(CenterPoint[0], CenterPoint[1], 0, 1)])
                train_img[i] = aug['image']
                train_gt[i][0], train_gt[i][1], train_gt[i][2] = \
                    aug['keypoints'][0][0], aug['keypoints'][0][1], aug['keypoints'][0][-1] * Radius * 2

                i += 1
            # print(train_gt[0], train_img.shape, np.max(train_img))
            # img = (train_img[0]).astype(np.uint8)
            # cv2.circle(img, (int(train_gt[0][0]), int(train_gt[0][1])), radius=int(train_gt[0][-1]), thickness=2, color=(0, 255, 255))
            # cv2.imshow('2', img)
            # cv2.waitKey(0)
            train_gt = train_gt / 80.0
            train_img = train_img / 255.

            yield train_img, train_gt
            

    def Image_Keypoint_Augumation(self):
        return Compose([
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            ShiftScaleRotate(p=1, scale_limit=0.25, shift_limit=0.1, rotate_limit=90),
            GaussNoise(p=0.3),
            RandomBrightnessContrast(p=1, brightness_limit=0.2, contrast_limit=0.2),
            MedianBlur(blur_limit=3, p=0.2),
            Resize(p=1, height=80, width=80)
        ], p=1)

    def load_image(self, name, norm=False):
        """
        Load input image, perform normalization if needed.
        """
        # img_f = os.path.join(self.image_dir, name + '.jpg')
        img = cv2.imread(name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = cv2.resize(img, (self.img_size_w, self.img_size_h))
        # if norm:
        #     img = img.astype(np.float32) / 255
        # else:
        #     img = img.astype(np.float32)
        return img
        

    # def load_gt(self, joint, position, norm=True):
    #     """
    #     Load groundtruth map for labeled samples.
    #     """
    #     num_joints = len(joint)
    #     hm = np.zeros((self.img_size_h, self.img_size_w, 2), dtype=np.float32)
    #
    #     # Generate groundtruth segmentation map
    #     for i in range(num_joints):
    #         if joint[i] == '0':
    #             #hm[int(position[i][1]), int(position[i][0]), 0] = 255
    #             temp = cv2.cvtColor(hm[:, :, 0], cv2.COLOR_GRAY2BGR)
    #             cv2.circle(temp, (int(position[i][0]), int(position[i][1])),
    #                 self.mask_r, (255, 255, 255), -1)
    #             hm[:, :, 0] = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
    #         else:
    #             joint_cat = int(joint[i]) - 1
    #             temp = cv2.cvtColor(hm[:, :, joint_cat], cv2.COLOR_GRAY2BGR)
    #             num_positions = len(position[i])
    #             for j in range(num_positions // 2 - 1):
    #                 cv2.line(temp, (int(position[i][2 * j]), int(position[i][2 * j + 1])),
    #                               (int(position[i][2 * j + 2]), int(position[i][2 * j + 3])),
    #                                (255, 255, 255), self.mask_r*2)
    #             hm[:, :, joint_cat] = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
    #
    #     # Normalize segmentation map
    #     if norm:
    #         for cat in range(2):
    #             if np.max(hm[:, :, cat]) > 1e-10:
    #                 hm[:, :, cat] = hm[:, :, cat] / np.max(hm[:, :, cat])
    #
    #     return hm
