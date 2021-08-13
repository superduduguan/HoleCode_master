"""
The Model Architecture.
"""

import tensorflow as tf
from baseModel import BaseModel


class Model(BaseModel):

    def __init__(self, *args, **kwargs):
        super(Model, self).__init__(*args, **kwargs)
        
        
    def net(self, image, name='HoleDefect'):
        """
        The deep CNN architecture.
        """
        with tf.variable_scope(name):
            # Downscale 2
            net = self._conv_bn_relu(image, 16, 3, 2, 'SAME', 'Conv1_1')
            net = self._conv_bn_relu(net, 16, 3, 1, 'SAME', 'Conv1_2')

            # Downscale 4
            net = self._conv_bn_relu(net, 32, 3, 2, 'SAME', 'Conv2_1')
            net = self._conv_bn_relu(net, 32, 3, 1, 'SAME', 'Conv2_2')

            # Downscale 8
            net = self._conv_bn_relu(net, 64, 3, 2, 'SAME', 'Conv3_1')
            net = self._conv_bn_relu(net, 64, 3, 1, 'SAME', 'Conv3_2')

            # Downscale 16
            net = self._conv_bn_relu(net, 64, 3, 2, 'SAME', 'Conv4_1')
            net = self._conv_bn_relu(net, 64, 3, 1, 'SAME', 'Conv4_2')

            # Downscale 32 ??
            if self.xl:
                net = self._conv_bn_relu(net, 64, 3, 2, 'SAME', 'Conv5_1')
                net = self._conv_bn_relu(net, 64, 3, 1, 'SAME', 'Conv5_2')


            # Regression
            embedding = self._fc_bn_relu(net, 128, name='fc')
            embedding = self._fc(embedding, 64, name='Embedding')  # x, y, r

            score = self._bn_relu(embedding, name='embedding_bn_relu')
            score = self._fc_bn_relu(score, 32, name='Classfication_1')

            score = self._fc(score, 10, name='Classfication')

        return embedding, score
