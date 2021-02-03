import os
import sys
import tensorflow as tf
slim = tf.contrib.slim
from model import Model

MODEL_PATH = '/home/vision-02/Hole_Detection/Hole_Position/logs/20201023_1342/model.ckpt-79'

def freeze_mobilenet(meta_file):

    tf.reset_default_graph()
    # model = AttModel(training=False,w_summary=False)
    model = Model(training=False,w_summary=False)
    model.BuildModel()
    
    output_node_names = ['HoleDetection/LocationResult']
    # output_node_names = ['AnomalyDetection/ClassResult']

    output_pb_name = 'HolePosition.pb'

    rest_var = slim.get_variables_to_restore()

    with tf.Session() as sess:
        graph = tf.get_default_graph()
        input_graph_def = graph.as_graph_def()

        saver = tf.train.Saver(rest_var)
        saver.restore(sess, meta_file)
        output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names)
        tf.train.write_graph(output_graph_def, "./", output_pb_name, as_text=False)


freeze_mobilenet(MODEL_PATH)


