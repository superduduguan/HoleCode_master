"""
The Model Architecture.
"""

import tensorflow as tf
from baseModel import BaseModel


class Model(BaseModel):

    def __init__(self, *args, **kwargs):
        super(Model, self).__init__(*args, **kwargs)
        
        
    def net(self, image, name='HoleDetection'):
        """
        The deep CNN architecture.
        """
        with tf.variable_scope(name):
            # Downscale 2
            net = self._conv_bn_relu(image, 16, 3, 2, 'SAME', 'Conv1_1')
            net = self._conv_bn_relu(net, 16, 3, 1, 'SAME', 'Conv1_2')
            #net = self._conv_bn_relu(net, 32, 3, 1, 'SAME', 'Conv1_3')
            # Downscale 4
            net = self._conv_bn_relu(net, 32, 3, 2, 'SAME', 'Conv2_1')
            net = self._conv_bn_relu(net, 32, 3, 1, 'SAME', 'Conv2_2')
            #net = self._conv_bn_relu(net, 64, 3, 1, 'SAME', 'Conv2_3')
            # Downscale 8
            net = self._conv_bn_relu(net, 64, 3, 2, 'SAME', 'Conv3_1')
            net = self._conv_bn_relu(net, 64, 3, 1, 'SAME', 'Conv3_2')
            #net = self._conv_bn_relu(net, 64, 3, 1, 'SAME', 'Conv3_3')
            #net = self._conv_bn_relu(net, 64, 3, 1, 'SAME', 'Conv3_4')
            # Downscale 16
            net = self._conv_bn_relu(net, 64, 3, 2, 'SAME', 'Conv4_1')
            net = self._conv_bn_relu(net, 64, 3, 1, 'SAME', 'Conv4_2')

            # Regression
            location = self._fc(net, 3, name='Location') # x, y, r
            # score = self._fc(net, 1, name='Classfication')

            
        # if self.training:
        #     return location#, score
        # else:
        #     return location#, tf.nn.sigmoid(score, name='HoleDetection/ClassResult')
        return tf.nn.sigmoid(location, name='HoleDetection/LocationResult')
