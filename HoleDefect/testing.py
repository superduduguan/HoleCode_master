import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np

import tensorflow.gfile as gfile
import cv2
import pickle
from tqdm import tqdm
from shutil import copyfile
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def grade_mode(list):
    '''
    计算众数

    参数：
        list：列表类型，待分析数据

    返回值：
        grade_mode: 列表类型，待分析数据的众数

    '''
    # TODO
    # 定义计算众数的函数
    # grade_mode返回为一个列表，可记录一个或者多个众数

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

def load_img1(path):
    """ Get img for Hole Position"""
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (80, 80)) / 255.
    return img
    

def load_img(path):
    """ Get img for Hole Position"""
    img = cv2.imread(path)
    img = img / 255.
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
    """
    get paths of all files in input_dir
    """
    all_paths = []
    for rootdir, subdirs, filenames in os.walk(input_dir):
        if len(filenames) > 0:
            for filename in filenames:
                all_paths.append(os.path.join(rootdir, filename))
    return all_paths

def defect(normal_hole):
    """
    get the score
    """
    # 获取当前文件所在目录
    cur_path = os.path.abspath(__file__)
    cur_dir = os.path.dirname(cur_path)

    # 创建分类网络
    NUM_CLA_MODEL = 5
    cla_models = []
    for i in range(NUM_CLA_MODEL):
        cla_model_path = os.path.join(cur_dir, 'HoleDefect' + str(i) + '_30.pb')
        cla_models.append(ClaPbModel(cla_model_path))


    # 运行分类网络0
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
    for x in range(len(GT)):
        dic_all[GT[x]] += 1
        if predicts[x] == GT[x]:
            correct += 1
            dic_right[GT[x]] += 1
    for cat, all in dic_all.items():
        dic_all[cat] = dic_right[cat] / all * 100
    print('acc:', correct / len(GT))
    print(dic_all)
   








