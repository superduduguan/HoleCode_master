#encoding: utf-8
 
'''
用于提供一个完整的调用方式，加载所有模型，输入一种样本图像，输出概率数值或者判断标签。

'''

# encoding: utf-8
import os
import numpy as np
import tensorflow as tf
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
        cla_model_path = os.path.join(cur_dir, 'model/HoleDefect/HoleDefect' + str(i) + '.pb')
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
    SAMPLE_DIR = r'D:\ResineHole-dataset\subset_0001\test'

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
            RATIO = 0.9
            lux = int((a[0] - RATIO * a[2]) * 80)
            luy = int((a[1] - RATIO * a[2]) * 80)
            rdx = int((a[0] + RATIO * a[2]) * 80)
            rdy = int((a[1] + RATIO * a[2]) * 80)

            luy, lux = max(0, luy), max(0, lux)
            rdx, rdy = min(rdx, 80), min(rdy, 80)
            length = rdx - lux
            height = rdy - luy
            minor = length - height
            hole_img = img[luy:rdy, lux:rdx, :]



            if minor > 0:
                hole_img = cv2.copyMakeBorder(hole_img, minor // 2, minor - minor // 2, 0, 0, cv2.BORDER_REPLICATE)
            elif minor < 0:
                hole_img = cv2.copyMakeBorder(hole_img, 0, 0, (-minor // 2), (-minor) - (-minor // 2), cv2.BORDER_REPLICATE)

            name = path.split('\\')[-1]
            destdir = os.path.join(cur_dir, 'example/normalized', path.split('\\')[4] + '!' + name)
            hole_img = cv2.resize(hole_img, (48, 48))

            if height < 25:
                print(path, ' is too small...Pay attention!')

                tdir = os.path.join(cur_dir, 'example/toosmall', path.split('\\')[4] + '!' + name)
                cv2.imwrite(tdir, hole_img * 255)
                continue
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
        # print(img.shape)
        # print(img)
        normal_hole = Preprocess4Defect(img)
        # print(normal_hole.shape)
        IMG.append(normal_hole)
        gt = path.split('\\')[-1].split('!')[0]
        GT.append(gt)


    IMG = np.array(IMG)

    IMG = IMG.squeeze(axis=1)

    ALLSCORE = defect(IMG)

    scores = list(np.array(ALLSCORE).squeeze(axis=-1).T)
    AS = [list(i) for i in scores]

    with open(os.path.join(cur_dir, 'example/result/ALLSCORE.data'), 'wb') as f:
        pickle.dump(AS, f)
    with open(os.path.join(cur_dir, 'example/result/GT.data'), 'wb') as ff:
        pickle.dump(GT, ff)
    with open(os.path.join(cur_dir, 'example/result/PATH.data'), 'wb') as fff:
        pickle.dump(PATH, fff)


    posc = 0
    negc = 0

    with open(os.path.join(cur_dir, 'example/result/ALLSCORE.data'), 'rb') as f:
        ALLSCORE = pickle.load(f)
    with open(os.path.join(cur_dir, 'example/result/GT.data'), 'rb') as ff:
        GT = pickle.load(ff)
    with open(os.path.join(cur_dir, 'example/result/PATH.data'), 'rb') as fff:
        PATH = pickle.load(fff)

    for gt in GT:
        if gt == 'pos':
            posc += 1
        if gt == 'neg':
            negc += 1
    print('total_neg:', negc)
    print('total_pos:', posc)

    for thr in [0.1, 0.12, 0.16, 0.19, 0.23]:
        wrong_neg = 0
        wrong_pos = 0
        for i in range(len(GT)):
            defect_count = 0

            gt = GT[i]
            scores = ALLSCORE[i]
            path = PATH[i]

            for j in range(len(scores)):
                score = scores[j]
                if score > thr:
                    defect_count += 1

            if gt == 'neg':
                if defect_count >= 2:
                    wrong_neg += 1
                    if thr == 0.12:
                        # copyfile(path, os.path.join(r'C:\Users\pc\Desktop\v0.0.2\example\error\wrong_neg', path.split('\\')[-1]))
            if gt == 'pos':
                if defect_count < 2:
                    wrong_pos += 1
                    if thr == 0.12:
                        # copyfile(path, os.path.join(r'C:\Users\pc\Desktop\v0.0.2\example\error\wrong_pos', path.split('\\')[-1]))

        print('\nDEFECT_Thr:', thr)
        print('虚警:', wrong_neg, wrong_neg / negc)
        print('漏检', wrong_pos, wrong_pos / posc) 
