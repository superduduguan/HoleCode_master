"""
Prepare the dataset and generate batch data for training process.
"""

import numpy as np
import os
import cv2
import random
import json
from albumentations import HorizontalFlip, ShiftScaleRotate, GaussNoise, \
    MedianBlur, RandomBrightnessContrast, Compose, VerticalFlip


class DataGenerator(object):
    
    def __init__(self,
                 image_dir=None,
                 label_dir=None,
                 in_size_h=48,
                 in_size_w=48,
                 pool_scale=16,
                 val_ratio=0.2,
                 val_group=0):
        
        self.image_dir = image_dir
        self.label_dir = label_dir
        
        self.in_size_h = in_size_h
        self.in_size_w = in_size_w
        self.pool_scale = pool_scale
        self.val_group = val_group
        
        self.val_ratio = val_ratio

        self.color = [[1., 0., 0.], [1., 1., 0.], [0., 0., 1.]]
   
    # ======= Initialize Generator ======== 
    def _create_train_table(self):
        """ 
        Scan the dataset and create table of samples
        """
        self.train_table = []
        self.label_dict = {}
        # Add samples to the whole dataset
        for filename in os.listdir(self.image_dir):
            samplename = filename[:-3]
            img_path = os.path.join(self.image_dir, filename)
            cat = filename.split('!')[1]
            # FP = 0
            if cat in ['pos']:
                classific = 2
            elif cat == 'neg':
                classific = 0
                if filename.split('!')[0] != 'v1.1':
                    classific = 1
            else:
                raise ValueError('Unspecified class: {}'.format(cat))
                
            image = self.load_image(img_path)
            image = cv2.resize(image, (self.in_size_w, self.in_size_h))
                
            self.label_dict[samplename] = {'image': image, 'class': classific}
            self.train_table.append(samplename)
               
            
    def _randomize(self):
        """ 
        Randomize the set
        """
        random.shuffle(self.train_table)
    
    
    def _create_sets(self):
        """ 
        Select samples to construct training and validation set
        """

        # num_sample = len(self.train_table)
        # num_val = int(num_sample * self.val_ratio)
        # self.val_set = self.train_table[:num_val]
        # self.train_set = self.train_table[num_val:]
        self.train_set, self.val_set = [], []
        pos_val_num = 0
        with open(str(self.val_group) + '/train.txt', 'r') as f:
            # for i in range(len(self.train_set)):
            #     f.write(self.train_set[i] + '\n')
            while True:
                lines = f.readline().strip()
                if not lines:
                    break
                self.train_set.append(lines)
        with open(str(self.val_group) + '/test.txt', 'r') as f:
            # for i in range(len(self.val_set)):
            while True:
                lines = f.readline().strip()
                if not lines:
                    break
                self.val_set.append(lines)
                if lines.split('!')[1] == 'pos':
                    pos_val_num += 1
            #     f.write(self.val_set[i] + '\n')
        print(str(self.val_group), '--Training set: ', len(self.train_set), ' samples.')
        print(str(self.val_group), '--Validation set: ', len(self.val_set), ' samples.')
        print(str(self.val_group), '--Validation pos: ', pos_val_num, ' samples.')
    
    
    # ======= Batch Generator ========
    def _batch_generator(self, batch_size=64, normalize=True, sample_set='train'):
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
        augmentation = self.Image_Augmentation()
        while True:
            # Construct batch data containers:
            train_img = np.zeros((batch_size, self.in_size_h, self.in_size_w, 3), dtype=np.float32)
            train_gt = np.zeros((batch_size, ), dtype=np.float32)
            class_gt = np.zeros((batch_size, ), dtype=np.float32)
            total_color = np.zeros(shape=[batch_size, 3], dtype=np.uint8)

            # while i < batch_size:
            for i in range(batch_size):
                sample_idx = (sample_idx + 1) % len(data_set)
                if sample_idx == 0 and sample_set == 'train':
                    random.shuffle(data_set)
                sample = data_set[sample_idx]
                # Load original image and mask according to sample
                img = self.label_dict[sample]['image']
                label = self.label_dict[sample]['class']
                # Crop a patch from original image and mask as inputs
                if normalize:
                    aug = augmentation(image=img)
                    aug_img = aug['image'] / 255.

                    aug_img -= np.mean(aug_img)
                    aug_img -= np.min(aug_img)
                    train_img[i] = aug_img / (np.max(aug_img) + 1e-6)
                else:
                    aug_img = img / 255.
                    aug_img -= np.mean(aug_img)
                    aug_img -= np.min(aug_img)
                    train_img[i] = aug_img / (np.max(aug_img) + 1e-6)
                train_gt[i] = label
                total_color[i] = self.color[label]
                class_gt[i] = 1 if label > 2-1e-3 else 0
                # print(np.min(train_img[i]), np.max(train_img[i]), sample)
                # cv2.imshow('0', img)
                # cv2.imshow('1', (train_img[i]*255).astype(np.uint8))
                # cv2.waitKey(0)

            yield train_img, train_gt, total_color, class_gt
            
    
    # def crop(self, image, mask, norm=True):
    #     """
    #     Crop a patch from original image and mask, perform normalization if needed.
    #     """
    #     label = np.zeros((2), dtype=np.float32)
    #     # Randomly determine the size of cropped patch
    #     crop_scale = np.random.uniform(self.min_crop, 1.1)
    #     crop_scale = min(crop_scale, 1)
    #     crop_size_h = int(self.in_size_h * crop_scale)
    #     crop_size_w = int(self.in_size_w * crop_scale)
    #     # Randomly select a position
    #     h_start = np.random.randint(self.in_size_h - crop_size_h + 1)
    #     w_start = np.random.randint(self.in_size_w - crop_size_w + 1)
    #     cropped_image = image[h_start:(h_start + crop_size_h - 1),
    #                           w_start:(w_start + crop_size_w - 1)]
    #     cropped_mask = mask[h_start:(h_start + crop_size_h - 1),
    #                         w_start:(w_start + crop_size_w - 1)]
    #     # Resize and Normalize
    #     cropped_image = cv2.resize(cropped_image.astype(np.uint8),
    #         (self.in_size_w, self.in_size_h)).astype(np.float32)
    #     cropped_mask = cv2.resize(cropped_mask.astype(np.uint8),
    #         (self.in_size_w//self.pool_scale, self.in_size_h//self.pool_scale)).astype(np.float32)
    #     for cat in range(2):
    #         if np.max(cropped_mask[:, :, cat]) > 1e-10:
    #             label[cat] = 1
    #             if norm:
    #                 cropped_mask[:, :, cat] = cropped_mask[:, :, cat] / np.max(cropped_mask[:, :, cat])
    #     if norm:
    #         cropped_image = cropped_image / 255
    #
    #     return cropped_image, cropped_mask, label

    def Image_Augmentation(self):
        return Compose([
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            ShiftScaleRotate(p=1, scale_limit=0.05, shift_limit=0.08, rotate_limit=90),
            GaussNoise(p=0.3),
            RandomBrightnessContrast(p=1, brightness_limit=0.2, contrast_limit=0.2),
            MedianBlur(blur_limit=3, p=0.2),

        ], p=1)

    def load_image(self, name, norm=True):
        """
        Load input image, perform normalization if needed.
        """
        # img_f = os.path.join(self.image_dir, name + '.jpg')
        img = cv2.imread(name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # if norm:
        #     img = img.astype(np.float32) / 255
        # else:
        #     img = img.astype(np.float32)
        return img    
