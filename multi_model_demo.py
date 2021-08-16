#encoding: utf-8
 
'''
用于提供一个完整的调用方式，加载所有模型，输入一种样本图像，输出概率数值或者判断标签。

'''

# encoding: utf-8
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import tensorflow.gfile as gfile
import cv2
import pickle
from tqdm import tqdm
from shutil import copyfile
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
 
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
        cla_model_path = os.path.join(cur_dir, 'model/HoleDefect/HoleDefect' + str(i) + '_450_128_False_20210814_070121.pb')
        cla_models.append(ClaPbModel(cla_model_path))




    # 运行分类网络0
    all_score = []
    for i in range(NUM_CLA_MODEL):
        cla_result = cla_models[i].predict(normal_hole)

        all_score.append(cla_result)

    return all_score





if __name__ == '__main__':

    # 获取当前文件所在目录
    cur_path = os.path.abspath(__file__)
    cur_dir = os.path.dirname(cur_path)
    SAMPLE_DIR = os.path.join(cur_dir, 'example/origin')

    pb_path = os.path.join(cur_dir, 'model\HolePosition\HolePosition.pb')
    paths = get_all_path(SAMPLE_DIR)
    
    sess = tf.Session()
    with gfile.FastGFile(pb_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')

    for path in tqdm(paths):
        if path.split('.')[-1] != 'bmp':
            continue
        try:
            src = path
            img = load_img1(src)
            sess.run(tf.global_variables_initializer())
            input_x = sess.graph.get_tensor_by_name('input/img_in:0')
            op = sess.graph.get_tensor_by_name('HoleDetection/LocationResult:0')
            _input = np.expand_dims(img, 0)
            a = sess.run(op,  feed_dict={input_x: _input})[0]
            RATIO = 1.05
            lux = int((a[0] - RATIO * a[2]) * 80)
            luy = int((a[1] - RATIO * a[2]) * 80)
            rdx = int((a[0] + RATIO * a[2]) * 80)
            rdy = int((a[1] + RATIO * a[2]) * 80)
            luy, lux = max(0, luy), max(0, lux)
            rdx, rdy = min(rdx, 80), min(rdy, 80)
            hole_img = img[luy:rdy, lux:rdx, :]

            length = rdx - lux
            height = rdy - luy
            minor = length - height
            if minor > 0:
                hole_img = cv2.copyMakeBorder(hole_img, minor // 2, minor - minor // 2, 0, 0, cv2.BORDER_REPLICATE)
            elif minor < 0:
                hole_img = cv2.copyMakeBorder(hole_img, 0, 0, (-minor // 2), (-minor) - (-minor // 2), cv2.BORDER_REPLICATE)

            alpha = 8
            beta = 1.3
            center = np.array([hole_img.shape[0]//2, hole_img.shape[1]//2])
            for i in range(hole_img.shape[0]):
                for j in range(hole_img.shape[1]):
                    for k in range(hole_img.shape[2]):
                        new_dist = max(0.3 * hole_img.shape[0] + 2, np.linalg.norm(center-np.array([i, j]))) - 0.3 * hole_img.shape[0] - 2
                        x = hole_img[i][j][k]
                        hole_img[i][j][k] = hole_img[i][j][k] / (1 + (new_dist / alpha) ** beta)
                        
            hole_img = cv2.resize(hole_img, (48, 48))
            name = path.split('\\')[-1]
            if height < 25:
                print(path, ' is too small...Pay attention!')
                tdir = os.path.join(cur_dir, 'example/toosmall', path.split('\\')[4] + '!' + name)
                cv2.imwrite(tdir, hole_img * 255)
                continue
            
            destdir = os.path.join(cur_dir, 'example/normalized', path.split('\\')[4] + '!' + name)
            cv2.imwrite(destdir, hole_img * 255)
        except Exception as e:
            print(path, e)
    print('normalization finished')
    normdir = os.path.join(cur_dir, 'example/normalized')
    all_paths = get_all_path(normdir)

    # defect
    ALLSCORE = []
    GT = []
    PATH = []
    IMG = []
    for path in tqdm(all_paths):
        PATH.append(path)
        img = cv2.imread(path)

        img = np.expand_dims(img, axis=0)
        img = img.astype('float64')
        normal_hole = Preprocess4Defect(img)
        IMG.append(normal_hole)
        gt = path.split('\\')[-1].split('!')[-2]
        GT.append(gt)

    IMG = np.array(IMG)
    IMG = IMG.squeeze(axis=1)
    ALLSCORE = defect(IMG)
    scores = list(np.array(ALLSCORE).squeeze(axis=-1).T)
    ALLSCORE = [list(i) for i in scores]
    print(ALLSCORE)
