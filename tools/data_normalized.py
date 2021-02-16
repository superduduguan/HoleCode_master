# encoding: utf-8
import os
import numpy as np
import tensorflow as tf
import tensorflow.gfile as gfile
import cv2
from shutil import copyfile
from tqdm import tqdm
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
SRC_DIR = r'C:\Users\pc\Desktop\HoleCode\all'
DEST_DIR = r'C:\Users\pc\Desktop\HoleCode\all_norm'


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



def load_img(path):  
    """ Get img for Hole Position"""
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (80, 80)) / 255.
    return img
    



def get_all_path(input_dir):
    all_paths = []
    for rootdir, subdirs, filenames in os.walk(input_dir):
        if len(filenames) > 0:
            for filename in filenames:
                all_paths.append(os.path.join(rootdir, filename))
    return all_paths


def main(img_path):
    # 获取当前文件所在目录
    cur_path = os.path.abspath(__file__)
    cur_dir = os.path.dirname(cur_path)
    # 载入定位网络
    loc_model_path = os.path.join(cur_dir, 'model/position/twice.pb')
    loc_model = LocPbModel(loc_model_path)


    
    # 运行定位网络
    ORI_IMG_SIZE = 80
    img = np.zeros((1, ORI_IMG_SIZE, ORI_IMG_SIZE, 3), np.float)
    img[0] = load_img(img_path) 


    loc_result = loc_model.predict(img)
    # print('loc_result: {}'.format(loc_result))
    
    # 获取塞孔图像并统一大小
    SIZE_RATIO = 0.9  
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
    hole_img = cv2.resize(hole_img, (48, 48))

if __name__ == '__main__':
    pb_path = r'C:\Users\pc\Desktop\HoleCode_master\model\position\twice.pb'
    paths = get_all_path(SRC_DIR)
    
    sess = tf.Session()
    with gfile.FastGFile(pb_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')

    for path in paths:
        try:
            src = path
            img = load_img(src)
            sess.run(tf.global_variables_initializer())
            input_x = sess.graph.get_tensor_by_name('input/img_in:0')
            op = sess.graph.get_tensor_by_name('HoleDetection/LocationResult:0')
            _input = np.expand_dims(img, 0)
            a = sess.run(op,  feed_dict={input_x: _input})[0]
            RATIO = 0.9 #0.75
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
            # print('\n', hole_img.shape)
            if minor > 0:
                # print('1')
                hole_img = cv2.copyMakeBorder(hole_img, minor // 2, minor - minor // 2, 0, 0, cv2.BORDER_REPLICATE)
            elif minor < 0:
                # print('2', (-minor // 2), (-minor) - (-minor // 2))
                hole_img = cv2.copyMakeBorder(hole_img, 0, 0, (-minor // 2), (-minor) - (-minor // 2), cv2.BORDER_REPLICATE)


            path = path.split('\\')
            name = path[6] + '!' + path[7] + '!' + path[-1]
            dest = DEST_DIR + '\\' + name
            hole_img = cv2.resize(hole_img, (48, 48))
            cv2.imwrite(dest, hole_img * 255)
        except:
            print(path)
     