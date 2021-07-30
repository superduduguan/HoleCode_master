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
        cla_model_path = os.path.join(cur_dir, 'HoleDefect' + str(i) + '_90.pb')
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
    for x in range(len(GT)):
        dic_all[GT[x]] += 1
        if predicts[x] == GT[x]:
            correct += 1
            dic_right[GT[x]] += 1
    for cat, all in dic_all.items():
        dic_all[cat] = dic_right[cat] / all * 100
    print('acc:', correct / len(GT))
    print(dic_all)
   








