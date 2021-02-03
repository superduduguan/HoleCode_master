
import os
os.environ['CUDA_VIDIBLE_DEVICES'] = '0'
import cv2
import numpy as np
import json
import random
from model import Model
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import math
from tqdm import tqdm
import tensorflow as tf
import time

INPUT_SIZE_W = 48
INPUT_SIZE_H = 48
POOLING_SCALE = 16

IMAGE_DIR = '/home/vision-02/Hole_Detection/Hole_Data/Normalized_Data/train/'
# IMGAE_TEST_DIR = '/home/vision-02/Hole_Detection/Hole_Data/Normalized_Data/test/'
#IMGAE_TEST_DIR = '/home/vision-02/Hole_Detection/Hole_Data/Normalized_Data_v89/'
IMGAE_TEST_DIR = '/home/vision-02/Hole_Detection/Hole_Data/Data20201207/Normalized_Data_v2/'
# LABEL_DIR = '/home/vision-02/Hole_Detection/Hole_Data/labels_clean/'
MODEL_PATH = '/home/vision-02/Hole_Detection/Hole_DefectCenterv2/logs/20201110_2131/'
FOLD_NUM = 5
IGNORE_GROUP = []
POS_DECISION = 2

from tensorflow.python.platform import gfile
 
sess = tf.Session()
with gfile.FastGFile('/home/vision-02/Hole_Detection/Hole_DefectCenterv2/HoleDefect0.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
 
sess.run(tf.global_variables_initializer())

input_x = sess.graph.get_tensor_by_name('input/img_in:0')
op = sess.graph.get_tensor_by_name('HoleDefect/Classfication/dense/BiasAdd:0')


class DataEvaluator(object):

    def __init__(self,
                 image_dir=None,
                 label_dir=None,
                 in_size_h=48,
                 in_size_w=48):

        self.image_dir = image_dir
        self.label_dir = label_dir

        self.in_size_h = in_size_h
        self.in_size_w = in_size_w
        self.color = [[1., 0., 0.], [1., 1., 0.], [0., 0., 1.]]

    def load_image(self, name, norm=True):
        """
        Load input image, perform normalization if needed.
        """
        # img_f = os.path.join(self.image_dir, name + '.jpg')
        img = cv2.imread(name)
        if img is None:
            print(name)
            exit()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if norm:
            img = img.astype(np.float32) / 255.
            img -= np.mean(img)
            img -= np.min(img)
            img = img / (np.max(img) + 1e-6)
        else:
            img = img.astype(np.float32)
        return img

    def _create_test_table(self):
        """
        Create table of test samples and load groundtruth
        """
        self.test_table = []
        self.label_dict = {}
        self.train_table = []
        self.center = np.zeros([1, 64])
        i = 0
       

        for sample in os.listdir(IMAGE_DIR):
            labels = sample.split('!')[1]
            if labels == 'pos':
                classific = 2
            elif labels == 'neg':
                classific = 0
            else:
                raise ValueError(sample)
            image = self.load_image(IMAGE_DIR + sample)
            image = cv2.resize(image, (self.in_size_w, self.in_size_h))
            self.label_dict[sample] = {'image': image, 'class': classific}  # 裁减的区域
            self.train_table.append(sample)

        for sample in os.listdir(IMGAE_TEST_DIR):
            labels = sample.split('!')[1]
            if labels == 'pos':
                classific = 2
            elif labels == 'neg':
                classific = 0
            else:
                raise ValueError(sample)
            image = self.load_image(IMGAE_TEST_DIR + sample)
            image = cv2.resize(image, (self.in_size_w, self.in_size_h))
            self.label_dict[sample] = {'image': image, 'class': classific}  # 裁减的区域
            self.test_table.append(sample)

        print('--Test set: ', len(self.test_table), ' samples.')

    def eval(self, model=None):
        """
        Perform evaluation on test set with specified model and save results
        to a .json file.
        """
        self.test_pred = {}
        ave_t = 0.
        cnt = -20
        for sample in tqdm(self.test_table):
            img = self.label_dict[sample]['image']
            # img = cv2.resize(img.astype(np.uint8), (self.in_size_w, self.in_size_h)).astype(np.float32)
            _input = np.expand_dims(img, 0)
            #_input = _input.repeat(64, 0)
            for i in range(FOLD_NUM):
                if i in IGNORE_GROUP:
                    continue
                model_tmp = model[i]
                t1 = time.time()
                pred, label = model_tmp.sess.run([model_tmp.predmap, model_tmp.pred], feed_dict={model_tmp.img: _input})
                #label = model_tmp.sess.run([model_tmp.pred], feed_dict={model_tmp.img: _input})
                t2 = time.time()
                if cnt > 0:
                    ave_t += t2 - t1
                cnt += 1
                self.test_pred[sample + str(i)] = {'pred': pred, 'label': label}
        print(ave_t, cnt)
        ave_t /= cnt / 5
        ave_t *= 1000
        print(ave_t)#, ave_t/64.)

    def eval_validation(self, model=None):
        self.train_pred = {}
        for sample in tqdm(self.train_table):
            img = self.label_dict[sample]['image']
            # img = cv2.resize(img.astype(np.uint8), (self.in_size_w, self.in_size_h)).astype(np.float32)
            _input = np.expand_dims(img, 0)
            for i in range(FOLD_NUM):
                if i in IGNORE_GROUP:
                    continue
                model_tmp = model[i]
                pred, label = model_tmp.sess.run([model_tmp.predmap, model_tmp.pred],
                                                 feed_dict={model_tmp.img: _input})
                # ret = sess.run(op, feed_dict={input_x: _input})
                # print(ret, label)
                self.train_pred[sample + str(i)] = {'pred': pred, 'label': label}




    def visualize(self):
        # if sample not in self.test_table:
        #     sample = random.choice(self.test_table)
        # image = (self.label_dict[sample]['image'] * 255).astype(np.uint8)
        total_embedding = np.zeros(shape=[4 * len(self.test_pred), 64], dtype=np.float)
        # total_color = np.zeros(shape=[len(self.test_pred), 3], dtype=np.uint8)
        for i, sample in enumerate(self.test_table):
            total_embedding[i] = self.test_pred[sample]['pred'][0]
            # total_embedding[4 * i + 1] = self.test_pred[sample + '1']['pred'][0]
            # total_embedding[4 * i + 2] = self.test_pred[sample + '2']['pred'][0]
            # total_embedding[4 * i + 3] = self.test_pred[sample + '3']['pred'][0]
            # total_color[i] = self.color[self.label_dict[sample]['class']]
        Isomap_data = total_embedding
        # Isomap_data = TSNE(perplexity=25).fit_transform(total_embedding)
        # Isomap_data = Isomap(total_embedding)
        # figure = plt.figure(figsize=(15, 8))
        # plt.suptitle('ISOMAP AND MDS COMPARE TO ORIGINAL DATA')
        # plt.title('ISOMAP')
        # print(Isomap_data.shape, Isomap_data)
        # plt.scatter(Isomap_data[:,0],Isomap_data[:,1],c=total_color)
        # plt.show()
        # plt.pause(2)
        # plt.close()


    def acc(self, center, threshold):

        def sigmoid(x):
            return 1 / (1 + math.exp(-x))

        tp, fp, tn, fn = 0, 0, 0, 0
        Tsample, Fsample = 0, 0
        # self.error_list = []
        for sample in self.test_table:
            predcls = []
            gtcls = 1 if self.label_dict[sample]['class'] > 1 + 1e-3 else 0
            # predcls[0] = sigmoid(self.test_pred[sample]['label'])
            for i in range(FOLD_NUM):
                if i in IGNORE_GROUP:
                    continue
                predcls.append(sigmoid(self.test_pred[sample + str(i)]['label']))
                # print(predcls[i])

            if gtcls == 1:
                Tsample += 1
                num = 0
                for i in range(FOLD_NUM - len(IGNORE_GROUP)):
                    if predcls[i] > threshold:
                        num += 1
                if num >= POS_DECISION:
                    tp += 1
                else:
                    # cv2.imshow('wrong', self.label_dict[sample]['image'])
                    # cv2.waitKey(0)
                    # cv2.imwrite('wrong' + str(fn) + '.bmp', (255 * self.label_dict[sample]['image']).astype(np.uint8))
                    # print(sample)
                    fn += 1
                # if sample not in self.error_list:
                #     self.error_list.append(sample)
            else:
                num = 0
                Fsample += 1
                for i in range(FOLD_NUM - len(IGNORE_GROUP)):
                    if predcls[i] > threshold:
                        num += 1
                if num >= POS_DECISION:
                    # cv2.imwrite('FP/' + sample, (255 * self.label_dict[sample]['image']).astype(np.uint8))
                    fp += 1
                # if sample not in self.error_list:
                #     self.error_list.append(sample)
                else:
                    tn += 1
        if threshold == 0.01:
            print('True Sample: %4d, False Sample: %4d' % (Tsample, Fsample))
        return tp, fp, tn, fn

    def acc_train(self, center, threshold):

        def sigmoid(x):
            return 1 / (1 + math.exp(-x))

        tp, fp, tn, fn = 0, 0, 0, 0
        # self.error_list = []
        for sample in self.train_table:
            predcls = []
            gtcls = 1 if self.label_dict[sample]['class'] > 1 + 1e-3 else 0
            for i in range(FOLD_NUM):
                if i in IGNORE_GROUP:
                    continue
                predcls.append(sigmoid(self.train_pred[sample + str(i)]['label']))
                # print(predcls[i])

            if gtcls == 1:
                num = 0
                for i in range(FOLD_NUM - len(IGNORE_GROUP)):
                    if predcls[i] > threshold:
                        num += 1
                if num >= POS_DECISION:
                    tp += 1
                else:
                    if threshold < 0.329:
                        cv2.imwrite(sample, (255 * evaluator.label_dict[sample]['image']).astype(np.uint8))
                    fn += 1
            else:
                num = 0
                for i in range(FOLD_NUM - len(IGNORE_GROUP)):
                    if predcls[i] > threshold:
                        num += 1
                if num >= POS_DECISION:
                    fp += 1
                else:
                    tn += 1
        return tp, fp, tn, fn

    def acc_validation(self, threshold, m):

        def sigmoid(x):
            return 1 / (1 + math.exp(-x))

        tp, fp, tn, fn = 0, 0, 0, 0
        # self.error_list = []
        test_table_tmp = []
        with open(str(m) + '/train.txt') as f:
            while True:
                lines = f.readline().strip()
                if not lines:
                    break
                test_table_tmp.append(lines + 'bmp')

        for sample in test_table_tmp:
            # predcls = [0] * 4
            gtcls = 1 if self.label_dict[sample]['class'] > 1 + 1e-3 else 0
            predcls = sigmoid(self.train_pred[sample + str(m)]['label'])

            if gtcls == 1:
                if predcls > threshold:
                    tp += 1
                else:
                    fn += 1
            else:
                if predcls > threshold:
                    fp += 1
                else:
                    tn += 1
        return tp, fp, tn, fn

    def mAP(self, interpolate=True):
        pr_scale = np.linspace(0, 1, 1001)
        # pr_scale = [0.007]
        precisions = []
        recalls = []
        center = self.center
        for thresh in tqdm(pr_scale):
            tp, fp, tn, fn = self.acc(center, thresh)
            if (tp + fn) == 0:
                recalls.append(0)
            else:
                recalls.append(tp / (tp + fn))
            if (tn + fp) == 0:
                precisions.append(0)
            else:
                precisions.append(tn / (tn + fp))

        if interpolate:
            interpolate_precisions = []
            last_max = 0
            for precision in precisions:
                interpolate_precisions.append(max(precision, last_max))
                last_max = max(interpolate_precisions)
            precisions = interpolate_precisions

        # Compute mean average precision
        pre_recall = 0
        ap = 0
        for precision, recall in zip(precisions[::-1], recalls[::-1]):
            ap += precision * (recall - pre_recall)
            pre_recall = recall
        scatter_precision, scatter_recall = [], []
        for i in range(1, 1000):
            if recalls[i] > 0.985:
                scatter_precision.append(precisions[i])
                scatter_recall.append(recalls[i])
                print("Precisions: %.3f, Recall: %.3f in Threshold %.4f" % (precisions[i], recalls[i], pr_scale[i]))
            else:
                break
        print('mAP of positive: %f' % (ap))
        # fig, ax = plt.subplots(figsize=(11, 9))
        plt.scatter(scatter_precision[481], scatter_recall[481])
        plt.plot(precisions, recalls)
        plt.xlabel('Precisions')
        plt.ylabel('Recall')

        # plt.show()

    def mAP_train(self, interpolate=True):
        pr_scale = np.linspace(0, 1, 1001)
        # pr_scale = [0.007]
        precisions = []
        recalls = []
        center = self.center
        for thresh in tqdm(pr_scale):
            tp, fp, tn, fn = self.acc_train(center, thresh)
            if (tp + fn) == 0:
                recalls.append(0)
            else:
                recalls.append(tp / (tp + fn))
            if (tn + fp) == 0:
                precisions.append(0)
            else:
                precisions.append(tn / (tn + fp))

        if interpolate:
            interpolate_precisions = []
            last_max = 0
            for precision in precisions:
                interpolate_precisions.append(max(precision, last_max))
                last_max = max(interpolate_precisions)
            precisions = interpolate_precisions

        # Compute mean average precision
        pre_recall = 0
        ap = 0
        for precision, recall in zip(precisions[::-1], recalls[::-1]):
            ap += precision * (recall - pre_recall)
            pre_recall = recall
        scatter_precision, scatter_recall = [], []
        for i in range(1, 1000):
            if recalls[i] > 0.985:
                scatter_precision.append(precisions[i])
                scatter_recall.append(recalls[i])
                print("Precisions: %.3f, Recall: %.3f in Threshold %.4f" % (precisions[i], recalls[i], pr_scale[i]))
            else:
                break
        print('mAP of positive: %f' % (ap))
        # fig, ax = plt.subplots(figsize=(11, 9))
        plt.scatter(scatter_precision[286], scatter_recall[286])
        plt.plot(precisions, recalls)
        plt.xlabel('Precisions')
        plt.ylabel('Recall')

        # plt.show()

    def mAP_validation(self, interpolate=True):
        pr_scale = np.linspace(0, 1, 1001)
        fig, ax = plt.subplots(figsize=(11, 9))
        for m in range(FOLD_NUM):
            precisions = []
            recalls = []
            center = self.center
            for thresh in pr_scale:
                tp, fp, tn, fn = self.acc_validation(thresh, m)
                if (tp + fn) == 0:
                    recalls.append(0)
                else:
                    recalls.append(tp / (tp + fn))
                if (tn + fp) == 0:
                    precisions.append(0)
                else:
                    precisions.append(tn / (tn + fp))

            if interpolate:
                interpolate_precisions = []
                last_max = 0
                for precision in precisions:
                    interpolate_precisions.append(max(precision, last_max))
                    last_max = max(interpolate_precisions)
                precisions = interpolate_precisions

            # Compute mean average precision
            pre_recall = 0
            ap = 0
            for precision, recall in zip(precisions[::-1], recalls[::-1]):
                ap += precision * (recall - pre_recall)
                pre_recall = recall
            for i in range(1, 1000):
                if recalls[i] > 0.98:
                    print("Validation of model %2d: Precisions: %.3f, Recall: %.3f in Threshold %.4f"
                          % (m, precisions[i], recalls[i], pr_scale[i]))
                else:
                    break
            print('mAP of positive: %f' % (ap))

            # plt.scatter([precisions[1], precisions[5], precisions[8]], [recalls[1], recalls[5], recalls[8]])
            plt.plot(precisions, recalls, c=[1.0, 0.19 * m, 1.0 - 0.19 * m])
            plt.legend(["0","1","2","3","4"])
            # plt.xlabel('Precisions')
            # plt.ylabel('Recall')
        # plt.show()

model = []
for i in range(FOLD_NUM):
    model_tmp = Model(in_size_w=INPUT_SIZE_W,
                  in_size_h=INPUT_SIZE_H,
                  pool_scale=POOLING_SCALE,
                  training=False)
    model_tmp.BuildModel()
    model_tmp.restore_sess(MODEL_PATH + str(i) + '/model.ckpt-249')
    model.append(model_tmp)


# Construct the evaluator
evaluator = DataEvaluator(image_dir=IMAGE_DIR,
                          in_size_w=INPUT_SIZE_W,
                          in_size_h=INPUT_SIZE_H)

evaluator._create_test_table()

#evaluator.eval_validation(model)
#evaluator.mAP_validation(interpolate=False)
#
evaluator.eval(model)
evaluator.mAP(interpolate=False)
# evaluator.mAP_train(interpolate=False)

# evaluator.visualize()
# print(evaluator.accuracy())
# while True:
#     evaluator.visualize()
# evaluator.output_result('/home/vision-02/Hole_Detection/Hole_Data/JieJunHoleAOI-dataset')
plt.show()
