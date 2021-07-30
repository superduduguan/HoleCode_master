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
from tqdm import tqdm
import argparse
from shutil import copyfile


parser = argparse.ArgumentParser('argument for training')
parser.add_argument('--ep',
                    type=int,
                    default=100,  
                    help='epochs')
parser.add_argument('--bs',
                    type=int,
                    default=32,  
                    help='batch_size')
parser.add_argument('--l2',
                    type=bool,
                    default=True,
                    help='l2_regulation')
args = parser.parse_args()
EPOCH = args.ep
BATCH_SIZE = args.bs
l2 = args.l2

cur_path = os.path.abspath(__file__)
cur_dir = os.path.dirname(cur_path)
IMAGE_DIR = os.path.join(cur_dir, '..\\example\\dataset\\train\\')
TEXTDIR = os.path.join(cur_dir, 'txtdir')
LOGDIR = "logs/%d%02d%02d_%02d%02d%02d/" % (
    now.year, now.month, now.day, now.hour, now.minute, now.second)

INPUT_SIZE_W = 48  # 1224
INPUT_SIZE_H = 48  # 512
POOLING_SCALE = 16

VAL_RATIO = 0.2

WEIGHT_DECAY = 4e-5
BASE_LR = 1e-2
VAL_BATCH_SIZE = 100

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
                  val_batch_size=VAL_BATCH_SIZE,
                  gpu_memory_fraction=GPU_MEMORY_FRACTION,
                  l2=l2)

    model.BuildModel()
    model.train()







import os
import sys
import tensorflow as tf
slim = tf.contrib.slim
from model import Model

nums = [str(i) for i in range(5)]
for num in nums:
    MODEL_PATH = os.path.join(cur_dir, LOGDIR + num + '\\model.ckpt-' + str(EPOCH-1))

    tf.reset_default_graph()

    model = Model(training=False, w_summary=False)
    model.BuildModel()

    output_node_names = ['HoleDefect/Classfication/dense/BiasAdd']
    output_pb_name = 'HoleDefect' + num + '_' + str(EPOCH) + "_%d%02d%02d_%02d%02d%02d.pb" % (now.year, now.month, now.day, now.hour, now.minute, now.second)
    rest_var = slim.get_variables_to_restore()

    with tf.Session() as sess:
        graph = tf.get_default_graph()
        input_graph_def = graph.as_graph_def()

        saver = tf.train.Saver(rest_var)
        saver.restore(sess, MODEL_PATH)
        output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names)
        tf.train.write_graph(output_graph_def, os.path.join(cur_dir, 'model/'), output_pb_name, as_text=False)

print('\nfinish transforming')

import tensorflow.gfile as gfile


def grade_mode(list):
    list_set = set(list)  # 取list的集合，去除重复元素
    frequency_dict = {}
    for i in list_set:  # 遍历每一个list的元素，得到该元素何其对应的个数.count(i)
        frequency_dict[i] = list.count(i)  # 创建dict; new_dict[key]=value
    grade_mode = []
    for key, value in frequency_dict.items():  # 遍历dict的key and value。key:value
        if value == max(frequency_dict.values()):
            grade_mode.append(key)
    return grade_mode

def myargmax(lst):
    return max(range(len(lst)), key=lst.__getitem__)

class PbModel:
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



class ClaPbModel(PbModel):
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

def Preprocess4Defect(img):  ########
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

def defect(normal_hole):
    # 获取当前文件所在目录
    cur_path = os.path.abspath(__file__)
    cur_dir = os.path.dirname(cur_path)

    # 创建分类网络
    NUM_CLA_MODEL = 5
    cla_models = []
    for i in range(NUM_CLA_MODEL):
        cla_model_path = os.path.join(cur_dir, 'model/HoleDefect' + str(i) + '_' + str(EPOCH) + "_%d%02d%02d_%02d%02d%02d.pb" % (now.year, now.month, now.day, now.hour, now.minute, now.second))
        cla_models.append(ClaPbModel(cla_model_path))

    # 运行分类网络
    all_score = []
    for i in range(NUM_CLA_MODEL):
        cla_result = cla_models[i].predict(normal_hole)
        all_score.append(cla_result)

    return all_score


if __name__ == '__main__':

    print('testing on test dataset')

    # 获取当前文件所在目录
    cur_path = os.path.abspath(__file__)
    cur_dir = os.path.dirname(cur_path)
    normdir = os.path.join(cur_dir, "../example/dataset/test")
    all_paths = get_all_path(normdir)

    # defect
    ALLSCORE = []
    GT = []
    PATH = []
    IMG = []
    all = len(all_paths)
    for path in tqdm(all_paths):
        PATH.append(path)
        img = cv2.imread(path)
        img = np.expand_dims(img, axis=0)
        img = img.astype('float64')
        normal_hole = Preprocess4Defect(img)
        IMG.append(normal_hole)
        gt = path.split('\\')[-1].split('!')[0]
        GT.append(gt)

    IMG = np.array(IMG)
    IMG = IMG.squeeze(axis=1)
    ALLSCORE = defect(IMG)
    scores = np.array(ALLSCORE)
    scores = np.transpose(scores, (1, 0, 2))
    samples = [list(i) for i in scores]  
    

    predicts = []
    for sample in samples:  
        scores = [list(j) for j in sample]  
        SC = []
        for score in scores:  
            SC.append(myargmax(score))
        mode = grade_mode(SC)[-1]
        predicts.append(str(mode))

    correct = 0
    dic_all = {}
    dic_right = {}
    cats = [str(i) for i in range(10)]
    cnt = [0 for i in range(10)]
    dic_right.update(list(zip(cats, cnt)))
    dic_all.update(list(zip(cats, cnt)))
    time_stamp = "%d%02d%02d_%02d%02d%02d" % (now.year, now.month, now.day, now.hour, now.minute, now.second)
    os.mkdir(os.path.join(cur_dir, '../example/error/' + time_stamp))

    for x in range(len(GT)):
        dic_all[GT[x]] += 1
        if predicts[x] == GT[x]:
            correct += 1
            dic_right[GT[x]] += 1
        else:
            print(int(GT[x])+1, 'to', int(predicts[x])+1)
            folder_name = str(int(GT[x])+1) + 'to' + str(int(predicts[x])+1)
            folder_path = os.path.join(cur_dir, '../example/error/' + time_stamp, folder_name)
            if not os.path.exists(folder_path):
                os.mkdir(folder_path)
            copyfile(PATH[x], os.path.join(folder_path, PATH[x].split('\\')[-1]))

    for cat, all in dic_all.items():
        dic_all[cat] = dic_right[cat] / all * 100
    print('acc:', correct / len(GT))
    print(dic_all)
    print('filename:', output_pb_name)
    print(args)
