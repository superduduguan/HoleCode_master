import cv2
import numpy as np
import shutil
import os
import json
from tqdm import tqdm
from albumentations import \
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90, CenterCrop, \
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,\
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,\
    IAASharpen, IAAEmboss, RandomBrightnessContrast, RandomBrightness, Flip, OneOf, Compose, VerticalFlip

# def Image_Keypoint_Augumation():
#     return Compose([
#         HorizontalFlip(p=0.5),
#         VerticalFlip(p=0.5),
#         ShiftScaleRotate(p=1, scale_limit=0.25, shift_limit=0.1, rotate_limit=90),
#         GaussNoise(p=0.3),
#         RandomBrightnessContrast(p=1, brightness_limit=0.2, contrast_limit=0.2),
#         MedianBlur(blur_limit=3, p=0.2),
#         CenterCrop(p=1, height=80, width=80)
#     ], p=1)

# anno_path = '/home/vision-02/Hole_Detection/Hole_Data/labels/'
# img_path  = '/home/vision-02/Hole_Detection/Hole_Data/images/'
# anno_shuffle_path = '/home/vision-02/Hole_Detection/Hole_Data/labels_clean/'
# anno = os.listdir(anno_path)
# min_r, max_r = 999, -1
# min_x, min_y, max_x, max_y = 999, 999, -1, -1
# num_neg=0
# for i in tqdm(range(len(anno))):
#     anno_temp = anno[i].split('!')[3:]
#     anno_tmp = '/'.join(anno_temp)
#     img_tmp = img_path + anno_tmp[:-4] + 'bmp'
#     img = cv2.imread(img_tmp)
#
#
#     f = open(anno_path + anno[i], 'r')
#     js = json.load(f)
#     xmin, ymin, xmax, ymax = -1, -1, -1, -1
#     bnbbox = js['outputs']['object']
#     num = 0
#     if len(bnbbox) > 1:
#         for j in range(len(bnbbox)):
#             bnbbox_tmp = bnbbox[j]['bndbox']
#             # print(bnbbox_tmp)
#             xmin, ymin, xmax, ymax = bnbbox_tmp['xmin'], bnbbox_tmp['ymin'], bnbbox_tmp['xmax'], bnbbox_tmp['ymax']
#             if xmin > 100 or ymin > 100 or xmax > 100 or ymax > 100:
#                 xmin, ymin, xmax, ymax = -1, -1, -1, -1
#                 continue
#             num += 1
#     elif len(bnbbox) == 1:
#         bnbbox_tmp = bnbbox[0]['bndbox']
#         # print(bnbbox_tmp)
#         xmin, ymin, xmax, ymax = bnbbox_tmp['xmin'], bnbbox_tmp['ymin'], bnbbox_tmp['xmax'], bnbbox_tmp['ymax']
#         if xmin > 100 or ymin > 100 or xmax > 100 or ymax > 100:
#             xmin, ymin, xmax, ymax = -1, -1, -1, -1
#             num -= 1
#         num += 1
#
#     xcenter, ycenter = (xmin + xmax) / 2, (ymin + ymax) / 2
#     radius = (xmax - xmin) / 4 + (ymax - ymin) / 4
#     # aug = HorizontalFlip(p=0.5)
#     # img_aug = aug(image=img, keypoints=[(xcenter, ycenter, 0, 1)])
#     # img2, kp2 = img_aug['image'], img_aug['keypoints']
#     # print((xcenter, ycenter, 0, 1), kp2[0])
#
#     # aug = Image_Keypoint_Augumation()
#     # img_aug = aug(image=img, keypoints=[(xcenter, ycenter, 0, 1)])
#     # img2, kp2 = img_aug['image'], img_aug['keypoints']
#     # print(kp2)
#     # aug = ShiftScaleRotate(p=1,scale_limit=0.25, shift_limit=0.1)
#     # img_IAAPerspective = aug(image=img, keypoints=[(xcenter, ycenter, 0, 1)])
#     # img3, kp3 = img_IAAPerspective['image'], img_IAAPerspective['keypoints']
#     # print(kp3[0])
#     # img2 = np.array(img2)
#
#     # cv2.circle(img, (int(xcenter), int(ycenter)), radius=int(radius), thickness=2, color=(0, 0, 255))
#     # cv2.imshow("img", img)
#     # kp2 = kp2[0]
#     # cv2.circle(img2, (int(kp2[0]), int(kp2[1])), radius=int(radius * kp2[-1]), thickness=2, color=(0, 255, 255))
#     # cv2.imshow('2', img2)
#     # print(radius, kp2)
#     # radius = radius * kp2[-1]
#     # if kp2[0]-radius < 0 or kp2[0] + radius > 80 or kp2[1]-radius < 0 or kp2[1] + radius > 80:
#     #     num_neg += 1
#     #     print(num_neg)
#     # cv2.circle(img2, (int(kp2[0]), int(kp2[1])), radius=int(radius), thickness=2, color=(0, 255, 255))
#     # cv2.imshow('2', img2)
#     # cv2.waitKey(0)
#
#
#     if num == 1:
#         xc, yc = (xmin + xmax) / 2, (ymin + ymax) / 2
#         radius = (xmax-xmin)/4 + (ymax-ymin)/4
#         if radius > max_r:
#             max_r = radius
#         if radius < min_r:
#             min_r = radius
#         min_x, min_y, max_x, max_y = min(min_x, xmin), min(min_y, ymin), max(max_x, xmax), max(max_y, ymax)
#         new_dict = {'path': anno_tmp[:-4] + 'bmp',
#                     'class': 1,
#                     'bndbox': [xmin, ymin, xmax, ymax],
#                     'center': [xc, yc],
#                     'radius': radius}
#         # with open(anno_shuffle_path + '!'.join(anno_temp), "w") as f:
#         #     json.dump(new_dict, f)
#     elif num == 0:
#         print(i)
#         new_dict = {'path': anno_tmp[:-4] + 'bmp',
#                     'class': 0}
#         # with open(anno_shuffle_path + '!'.join(anno_temp), "w") as f:
#         #     json.dump(new_dict, f)
#     else:
#         print(num, anno[i])
# print(min_r, max_r, min_x, min_y, max_x, max_y)

context_r = 0.5
img_size_w, img_size_h = 100, 100
in_size_w, in_size_h = 48, 48

def Gt_ImageSample(gt_anno, imgs, output_path):
    anns = os.listdir(gt_anno)
    for ann in anns:
        if ann.split('!')[1] not in ['pos', 'neg']:
            if ann.split('!')[1] not in ['pos_cate', 'ignore']:
                print(ann)
            continue
        ann_path = ann.replace('!', '/')
        img_path = os.path.join(imgs, ann_path)[:-4] + 'bmp'
        img = cv2.imread(img_path)
        cv2.imshow('1', img)
        labels = json.load(open(gt_anno + ann, 'r'))
        center = labels['center']
        radius = labels['radius'] * (1 + context_r)
        xmin, xmax, ymin, ymax = int(center[0]-radius+0.5), \
                                 int(center[0]+radius+0.5), \
                                 int(center[1]-radius+0.5), \
                                 int(center[1]+radius+0.5)
        xmin, ymin = max(xmin, 0), max(ymin, 0)
        xmax, ymax = min(xmax, img_size_w - 1), min(ymax, img_size_h - 1)
        img = img[ymin:ymax + 1, xmin:xmax + 1]
        img = cv2.resize(img, (in_size_w, in_size_h), cv2.INTER_LINEAR)
        cv2.imwrite(output_path + ann[:-4] + 'bmp', img)


def Pred_ImageSample(pred_anno, imgs, output_path):
    anns = os.listdir(pred_anno)
    for ann in anns:
        if ann.split('!')[1] not in ['pos', 'neg']:
            if ann.split('!')[1] not in ['pos_cate', 'ignore']:
                print(ann)
            continue
        ann_path = ann.replace('!', '/')
        img_path = os.path.join(imgs, ann_path)[:-4] + 'bmp'
        img = cv2.imread(img_path)
        # cv2.imshow('1', img)
        labels = json.load(open(pred_anno + ann, 'r'))
        center = labels['center']
        radius = float(labels['radius']) * (1 + context_r)
        xmin, xmax, ymin, ymax = int(center[0] - radius + 0.5), \
                                 int(center[0] + radius + 0.5), \
                                 int(center[1] - radius + 0.5), \
                                 int(center[1] + radius + 0.5)
        xmin, ymin = max(xmin, 0), max(ymin, 0)
        xmax, ymax = min(xmax, img_size_w - 1), min(ymax, img_size_h - 1)
        img = img[ymin:ymax + 1, xmin:xmax + 1]
        img = cv2.resize(img, (in_size_w, in_size_h), cv2.INTER_LINEAR)
        cv2.imwrite(output_path + ann[:-4] + 'bmp', img)
        # cv2.imshow('2', img)
        # cv2.waitKey(0)

def table_create(input_path, output_path, neg_table, pos_table):
    for img in os.listdir(input_path):
        labels = img.split('!')
        if len(labels) < 2:
            continue
        if labels[1] == 'pos':
            if labels[0] in pos_table:
                shutil.move(os.path.join(input_path, img),
                            os.path.join(output_path, img))
        elif labels[1] == 'neg':
            if labels[0] in neg_table:
                shutil.move(os.path.join(input_path, img),
                            os.path.join(output_path, img))

if __name__ == '__main__':
    # v1.1 --- v1.3
    gt_anno = '/home/vision-02/Hole_Detection/Hole_Data/labels_clean/'
    img_gt = '/home/vision-02/Hole_Detection/Hole_Data/images/'
    # v1.4 --- v1.7
    pred_anno = '/home/vision-02/Hole_Detection/Hole_Data/JieJunHoleAOI-dataset/results1/'
    img_pred = '/home/vision-02/Hole_Detection/Hole_Data/JieJunHoleAOI-dataset/'
    output_path = '/home/vision-02/Hole_Detection/Hole_Data/Normalized_Data/'

    # Gt_ImageSample(gt_anno, img_gt, output_path)
    # Pred_ImageSample(pred_anno, img_pred, output_path)
    table_create('/home/vision-02/Hole_Detection/Hole_Data/Normalized_Data/',
                 '/home/vision-02/Hole_Detection/Hole_Data/Normalized_Data/train/',
                 neg_table=['v1.1', 'v1.2', 'v1.3', 'v1.4'],
                 pos_table=['v1.1', 'v1.2'])
