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
                 val_group=0, 
                 txtdir='C:\\Users\\pc\\Desktop\\ResineHole-dataset\\5-folds\\'):
        self.txtdir = txtdir
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
        self.train_table = []
        self.label_dict = {}
        # Add samples to the whole dataset
        for filename in os.listdir(self.image_dir):
            samplename = filename
            img_path = os.path.join(self.image_dir, filename)
            image = self.load_image(img_path)
            cat = int(img_path.split('\\')[-1].split('!')[0])

            self.label_dict[samplename] = {'image': image, 'class': cat}
            self.train_table.append(samplename)


               
            
    def _randomize(self):
        random.shuffle(self.train_table)
    
    
    def _create_sets(self):
        self.train_set, self.val_set = [], []

        with open(self.txtdir + '//' + str(self.val_group) + '/train.txt', 'r') as f:
            while True:
                lines = f.readline().strip()
                if not lines:
                    break
                self.train_set.append(lines)


        with open(self.txtdir + '//' + str(self.val_group) + '/val.txt', 'r') as f:
            # for i in range(len(self.val_set)):
            while True:
                lines = f.readline().strip()
                if not lines:
                    break
                self.val_set.append(lines)
        random.shuffle(self.train_set)
        random.shuffle(self.val_set)

        dic = {}
        cat = [str(i) for i in range(10)]
        cnt = [0 for i in range(10)]
        dic.update(list(zip(cat, cnt)))
        

        print(str(self.val_group), '--Training set: ', len(self.train_set), ' samples.')
        for i in self.train_set:
            dic[i.split('\\')[-1].split('!')[0]] += 1
        print('train:', dic)

        dic.update(list(zip(cat, cnt)))
        print(str(self.val_group), '--Validation set: ', len(self.val_set), ' samples.')
        for i in self.val_set:
            dic[i.split('\\')[-1].split('!')[0]] += 1
        print('train:', dic)
        


    
    
    # ======= Batch Generator ========
    def _batch_generator(self, batch_size=64, normalize=True, sample_set='train'):
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

            for i in range(batch_size):
                sample_idx = (sample_idx + 1) % len(data_set)
                if sample_idx == 0 and sample_set == 'train':
                    random.shuffle(data_set)
                sample = data_set[sample_idx].split('\\')[-1]
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

            yield train_img, train_gt
            
    

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
        
        img = cv2.imread(name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img    
