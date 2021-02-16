import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


pb_path = r'C:\Users\pc\Desktop\HoleCode_master\HolePosition\3and2x2.pb'
input_dir = r'C:\Users\pc\Desktop\HoleCode\rev.2'

def get_all_path(input_dir):
    all_paths = []
    for rootdir, subdirs, filenames in os.walk(input_dir):
        if len(filenames) > 0:
            for filename in filenames:
                all_paths.append(os.path.join(rootdir, filename))
    return all_paths


def load_image(name, norm=True):
    img = cv2.imread(name)
    if img is None:
        print(name)
        exit()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if norm:
        img = img.astype(np.float32) / 255
    else:
        img = img.astype(np.float32)
    return img

def main():
    all_paths = get_all_path(input_dir)


    sess = tf.Session()
    with gfile.FastGFile(pb_path, 'rb') as f:

        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()

        tf.import_graph_def(graph_def, name='')
    for image_path in all_paths:
        sess.run(tf.global_variables_initializer())
        input_x = sess.graph.get_tensor_by_name('input/img_in:0')
        op = sess.graph.get_tensor_by_name('HoleDetection/LocationResult:0')

        # image_path = r'C:\Users\pc\Desktop\HoleCode\v2.4.1\neg\b_19_11_2_8.bmp'
        # img = load_image(image_path)
        # a = [52.5/80, 50/80, 14.25/40]


        img = load_image(image_path)
        img = cv2.resize(img, (80, 80))
        # cv2.imshow('1', img)
        # cv2.waitKey(5000)
        _input = np.expand_dims(img, 0)

        a = sess.run(op,  feed_dict={input_x: _input})[0]
        # print(a)

        #

        RATIO = 0.9 #0.75
        lux = int((a[0] - RATIO * a[2]) * 80)
        luy = int((a[1] - RATIO * a[2]) * 80)
        rdx = int((a[0] + RATIO * a[2]) * 80)
        rdy = int((a[1] + RATIO * a[2]) * 80)
        # print(luy, rdy, lux, rdx)
        lux, luy = max(0, lux), max(0, luy)
        rdx, rdy = min(80, rdx), min(80, rdy)
        sub = '[' + str(lux) + '_' + str(luy) + '_' + str(rdx) + '_' + str(rdy) + ']'
        img = img[luy:rdy, lux:rdx, :]
        # cv2.imshow('1', img)
        # cv2.waitKey(5000)
        # print(luy, rdy, lux, rdx)
        # quit()


        try:
            filepath = "C:\\Users\\pc\\Desktop\\HoleCode\\test_result2\\" + str(image_path.split('\\')[-1])
            cv2.imwrite(filepath, img * 255)
        except:
            print(image_path)

if __name__ == '__main__':
    main()