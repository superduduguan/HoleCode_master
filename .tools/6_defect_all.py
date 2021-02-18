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
EPOCH = 50
EPOCH_SIZE = 8209 //BATCH_SIZE # 10421
LR_DECAY = 0.9
LR_DECAY_FREQ = 2
GPU_MEMORY_FRACTION = 0.2  #1.0

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

print('\nfinish trainning')
import os
import sys
import tensorflow as tf
slim = tf.contrib.slim
from model import Model

nums = ['0', '1', '2', '3', '4']
for num in nums:
    MODEL_PATH = 'C:\\Users\\pc\\Desktop\\HoleCode_master\\HoleDefect\\' + LOGDIR + num + r'\model.ckpt-' + str(EPOCH-1)

    def freeze_mobilenet(meta_file):

        tf.reset_default_graph()
        # model = AttModel(training=False,w_summary=False)
        model = Model(training=False,w_summary=False)
        model.BuildModel()

        output_node_names = ['HoleDefect/Classfication/dense/BiasAdd']
        # output_node_names = ['AnomalyDetection/ClassResult']

        output_pb_name = 'HoleDefect' + num + '.pb'

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

# encoding: utf-8

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import numpy as np
import tensorflow as tf
import tensorflow.gfile as gfile
import cv2
from tqdm import tqdm

class PbModel:
    """用于载入模型并维护独立的session

       对于存在多个模型的情况，可以创建多个对象，每个对象将会持有一个tensorflow的Session

       对于不同的模型，需要实现各自的predict函数，用于获取输入输出节点，并运行计算图并获得结果
    """

    def __init__(self, model_file):
        # 从pb文件载入模型
        self.graph = tf.Graph()
        self.graph_def = tf.GraphDef()
        with gfile.FastGFile(model_file, 'rb') as f:
            self.graph_def.ParseFromString(f.read())
        with self.graph.as_default():
            tf.import_graph_def(self.graph_def, name='')
        # 设置显存的使用方式为动态申请
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph, config=config)

    def predict(self, images: list):
        pass


class LocPbModel(PbModel):
    """ 用于管理塞孔定位网络
    """

    def predict(self, data):
        # 获取输入和输出节点
        output_node = self.sess.graph.get_tensor_by_name('%s:0' % 'HoleDetection/LocationResult')
        input_x = self.sess.graph.get_tensor_by_name('%s:0' % 'input/img_in')
        # 请求模型
        feed = {input_x: data}
        out = self.sess.run(output_node, feed)
        return out


class ClaPbModel(PbModel):
    """用于管理塞孔分类网络
    """
    def predict(self, data):
        def sigmoid(x):  ########
            return 1 / (1 + np.exp(-x))
        # 获取输入和输出节点
        output_node = self.sess.graph.get_tensor_by_name('%s:0' % 'HoleDefect/Classfication/dense/BiasAdd')
        input_x = self.sess.graph.get_tensor_by_name('%s:0' % 'input/img_in')
        # 请求模型
        feed = {input_x: data}
        out = self.sess.run(output_node, feed)
        return sigmoid(out)  ########


def load_img(path):  ########
    """ Get img for Hole Position"""
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (80, 80)) / 255.
    return img
    
def Preprocess4Defect(img):  ########
    """
    img = (img - mean(img) - min(img)) / max(img)
    Preprocessing img for better classification performance.
    """
    img -= np.mean(img)
    img -= np.min(img)
    img = img / (np.max(img) + 1e-6)
    return img

def get_all_path(input_dir):
    all_paths = []
    for rootdir, subdirs, filenames in os.walk(input_dir):
        if len(filenames) > 0:
            for filename in filenames:
                all_paths.append(os.path.join(rootdir, filename))
    return all_paths

def mainx(img_path):
    # 获取当前文件所在目录
    cur_path = os.path.abspath(__file__)
    cur_dir = os.path.dirname(cur_path)
    # 载入定位网络
    loc_model_path = '../model/position/3and2x2.pb'
    loc_model = LocPbModel(loc_model_path)
    # 创建分类网络
    NUM_CLA_MODEL = 5
    cla_models = []
    for i in range(NUM_CLA_MODEL):
        cla_model_path = 'HoleDefect{}.pb'.format(i)
        cla_model_path = os.path.join(cur_dir, cla_model_path)
        cla_models.append(ClaPbModel(cla_model_path))

    
    # 运行定位网络
    ORI_IMG_SIZE = 80
    img = np.zeros((1, ORI_IMG_SIZE, ORI_IMG_SIZE, 3), np.float)
    img[0] = load_img(img_path)  ########


    loc_result = loc_model.predict(img)
    # print('loc_result: {}'.format(loc_result))
    
    # 获取塞孔图像并统一大小
    SIZE_RATIO = 0.9  ########
    lux = int((loc_result[0][0] - SIZE_RATIO * loc_result[0][2]) * ORI_IMG_SIZE)
    luy = int((loc_result[0][1] - SIZE_RATIO * loc_result[0][2]) * ORI_IMG_SIZE)
    rdx = int((loc_result[0][0] + SIZE_RATIO * loc_result[0][2]) * ORI_IMG_SIZE)
    rdy = int((loc_result[0][1] + SIZE_RATIO * loc_result[0][2]) * ORI_IMG_SIZE) 

    luy, lux = max(0, luy), max(0, lux)
    rdx, rdy = min(rdx, 80), min(rdy, 80)
    length = rdx - lux
    height = rdy - luy
    minor = length - height
    hole_img = img[0][luy:rdy, lux:rdx, :]

    # print('\n', hole_img.shape)
    if minor > 0:
        # print('1')
        hole_img = cv2.copyMakeBorder(hole_img, minor // 2, minor - minor // 2, 0, 0, cv2.BORDER_REPLICATE)
    elif minor < 0:
        # print('2', (-minor // 2), (-minor) - (-minor // 2))
        hole_img = cv2.copyMakeBorder(hole_img, 0, 0, (-minor // 2), (-minor) - (-minor // 2), cv2.BORDER_REPLICATE)
    # print(hole_img.shape)
    hole_img = np.expand_dims(hole_img, axis=0)

    # print('\n')
    #



    #输出图hole_img

    NORMAL_IMG_SIZE = 48
    #
    # np.resize != cv2.resize. Here produce a great error while testing.
    normal_hole = np.resize(hole_img, (1, NORMAL_IMG_SIZE, NORMAL_IMG_SIZE, 3))
    normal_hole[0] = cv2.resize(hole_img[0], (NORMAL_IMG_SIZE, NORMAL_IMG_SIZE))  ########
    normal_hole = Preprocess4Defect(normal_hole)  ########
    
    # 运行分类网络0
    all_score = []


    for i in range(NUM_CLA_MODEL):
        cla_result = cla_models[i].predict(normal_hole)
        # print('cla_result_{}:{}'.format(i, cla_result))
        all_score.append(cla_result[0][0])



    return hole_img, all_score
    # 投票选出最终结果
    # if defect_count >= 2:
    #     print(defect_count, 'final:defect')
    # else:
    #     print(defect_count, 'final:normal')

if __name__ == '__main__':

    dir = r'C:\Users\pc\Desktop\HoleCode\Normalized_Data\test'
    all_paths = get_all_path(dir)

    ALLSCORE = []
    GT = []
    for path in tqdm(all_paths):
        # try:
        hole_img, score = mainx(path)
        ALLSCORE.append(score)
        gt = path.split('\\')[-1].split('!')[1]
        GT.append(gt)

    for thr in [0.1, 0.2, 0.25, 0.3, 0.4, 0.43, 0.45, 0.48, 0.51]:
        wrong_neg = 0
        wrong_pos = 0
        for i in range(len(GT)):
            defect_count = 0

            gt = GT[i]
            scores = ALLSCORE[i]
            # print(scores, gt)
            for j in range(len(scores)):
                score = scores[j]
                if score > thr:
                    defect_count += 1  # #################################
            if gt == 'neg':
                if defect_count >= 2:
                    wrong_neg += 1
                    # filepath = "C:\\Users\\pc\\Desktop\\HoleCode\\test_result\\" + str(defect_count) + '_' + str(path.split('\\')[-1])
                    # cv2.imwrite(filepath, hole_img[0] * 255)

            if gt == 'pos':
                if defect_count < 2:
                    wrong_pos += 1
                    # print(defect_count, score)
                    # filepath = "C:\\Users\\pc\\Desktop\\HoleCode\\test_result\\" + str(defect_count) + '_' + str(path.split('\\')[-1])
                    # cv2.imwrite(filepath, hole_img[0] * 255)

        print('\nDEFECT_TH', thr)
        print('虚警:', wrong_neg, int(wrong_neg / 582 * 1000) / 10.0, '%') # 728
        print('漏检', wrong_pos, int(wrong_pos / 57 * 1000) / 10.0, '%') # 72

