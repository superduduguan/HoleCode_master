"""
The base class of the Model.
"""

import warnings

warnings.filterwarnings("ignore")
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import cv2
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm



class BaseModel(object):
    def __init__(
            self,
            dataset=None,
            logdir='logs',
            in_size_h=48,
            in_size_w=48,
            pool_scale=16,
            weight_decay=0.0001,
            base_lr=0.001,
            epoch=50,
            epoch_size=1000,
            lr_decay=0.95,
            lr_decay_freq=4,
            batch_size=8,
            val_batch_size=100,
            gpu_memory_fraction=1.0,  # 1.0
            training=True,
            w_summary=True,
            alpha=0.6,
            beta=2,
            announce=False, 
            l2=True, 
            xl=False):

        # Configure GPU
        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=gpu_memory_fraction)
        config = tf.ConfigProto(log_device_placement=False,
                                allow_soft_placement=True,
                                gpu_options=gpu_options)
        tf.reset_default_graph()
        self.sess = tf.Session(config=config)
        self.announce = announce
        self.xl = xl
        # Set log dir
        self.log_dir = logdir
        self.writer = tf.summary.FileWriter(self.log_dir)

        # Net args
        self.dataset = dataset
        self.in_size_w = in_size_w
        self.in_size_h = in_size_h
        self.alpha = alpha
        self.beta = beta
        # print(self.in_size_h, self.in_size_w)
        self.pool_scale = pool_scale

        self.training = training
        self.weight_decay = weight_decay
        self.base_lr = base_lr
        self.epoch = epoch
        self.epoch_size = epoch_size
        self.lr_decay = lr_decay
        self.lr_decay_freq = lr_decay_freq
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size

        #   step learning rate policy
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(self.base_lr,
                                                        self.global_step,
                                                        self.lr_decay_freq *
                                                        self.epoch_size * 5,
                                                        self.lr_decay,
                                                        staircase=True)

        #   Inside Variable
        self.train_step = None
        self.loss = None
        self.w_summary = w_summary
        self.l2 = l2

        self.img = None
        self.gt = None

        self.summ_scalar_list = []
        self.summ_scalar_list_TRAIN = []
        # self.summ_accuracy_list = []
        self.summ_axis_list = []
        self.summ_size_list = []
        self.summ_image_list = []
        self.summ_histogram_list = []

    def __build_ph(self):
        """ 
        Build Placeholder in tensorflow session
        """
        # Input RGB image
        self.img = tf.placeholder(
            tf.float32,
            shape=[None, self.in_size_h, self.in_size_w, 3],
            name='img_in')
        self.gt = tf.placeholder(tf.float32, shape=[
            None,
        ], name='label')
        self.class_gt = tf.placeholder(tf.float32,
                                       shape=[
                                           None,
                                       ],
                                       name='class')


    def __build_train_op(self):
        """ 
        Build loss and optimizer
        """
        # Select part of parameters to train

        # Loss
        with tf.name_scope('loss'):
            self.mygt = tf.cast(tf.one_hot(
                tf.cast(self.class_gt, dtype=tf.uint8), 10),
                                dtype=tf.float32)  #classnum
            # print(self.mygt)
            # print(self.pred)
            self.class_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=self.mygt,
                                                        logits=self.pred))

            trainable_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES)
            self.l2_regular = self.weight_decay * tf.reduce_sum(
                [tf.nn.l2_loss(var) for var in trainable_vars],
                name='L2_Regularization')

            self.loss = self.class_loss + self.l2_regular
            if self.l2:
                self.loss += self.l2_regular

            self.summ_scalar_list.append(
                tf.summary.scalar("total_loss_val", self.loss, family='val'))
            self.summ_scalar_list.append(
                tf.summary.scalar("l2_regularization_val", self.l2_regular, family='val'))
            self.summ_scalar_list_TRAIN.append(tf.summary.scalar("total_loss_train", self.loss, family='train'))
            self.summ_scalar_list_TRAIN.append(tf.summary.scalar("l2_regularization_train", self.l2_regular, family='train'))
    

        # Optimizer
        with tf.name_scope('optimizer'):
            with tf.control_dependencies(
                    tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
                self.train_step = self.optimizer.minimize(
                    self.loss,
                    var_list=trainable_vars,
                    global_step=self.global_step)
        

    def BuildModel(self):
        """ 
        Build model in tensorflow session
        """
        # Input
        with tf.name_scope('input'):
            self.__build_ph()
        assert self.img != None and self.gt != None

        self.predmap, self.pred = self.net(self.img)  ############ 查一下score
        self.look = self.pred
        # self.pred = tf.reshape(self.pred, [-1])

        if self.training:
            # train op
            with tf.name_scope('train'):
                self.__build_train_op()

        # initialize all variables
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        if self.training:
            # merge all summary
            self.summ_scalar = tf.summary.merge(self.summ_scalar_list)
            self.summ_scalar_TRAINING = tf.summary.merge(self.summ_scalar_list_TRAIN)
        self.writer.add_graph(self.sess.graph)
        

    def restore_sess(self, model=None):
        """ 
        Restore session from ckpt format file
        """

        if model is not None:
            if self.training:
                variables_to_restore = tf.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES,
                    'HoleDetection/LocationResult')
                part_saver = tf.train.Saver(variables_to_restore)
                part_saver.restore(self.sess, model)
            else:
                self.saver.restore(self.sess, model)
            print(" -- Sess Restored")
        else:
            print("Please input proper model path to restore!")
            raise ValueError

    # train
    def train(self):
        """ 
        Training Process
        """
        _epoch_count = 0
        _iter_count = 0

        # Construct batch-data generator
        self.train_generator = self.dataset._batch_generator(
            self.batch_size, normalize=True, sample_set='train')
        self.val_generator = self.dataset._batch_generator(self.val_batch_size,
                                                           normalize=True,
                                                           sample_set='val')

        # Training process
        for n in tqdm(range(self.epoch)):
            #self.dataset._randomize()
            for m in range(self.epoch_size):  # m: one batch
                _train_batch = next(self.train_generator)

                self.sess.run(self.train_step,
                              feed_dict={
                                  self.img: _train_batch[0],
                                  self.gt: _train_batch[1],
                                  self.class_gt: _train_batch[1]
                              })

                if _iter_count % 20 == 0:
                    _test_batch = next(self.val_generator)
                    # Print training loss
                    predmap, loss, class_loss = \
                        self.sess.run([self.predmap, self.loss, self.class_loss],  # , self.embedding_loss],
                                  feed_dict={self.img: _train_batch[0],
                                             self.gt: _train_batch[1],
                                             self.class_gt: _train_batch[1]})
                    if self.announce:
                        print('--Epoch: ', _epoch_count, ' iter: ', _iter_count,
                            ' train_loss: ', [loss, class_loss])

                    # Record validation results
                    self.writer.add_summary(
                        self.sess.run(self.summ_scalar_TRAINING,
                                      feed_dict={self.img: _train_batch[0],
                                             self.gt: _train_batch[1],
                                             self.class_gt: _train_batch[1]}), _iter_count)

                    self.writer.add_summary(
                        self.sess.run(self.summ_scalar,
                                      feed_dict={
                                          self.img: _test_batch[0],
                                          self.gt: _test_batch[1],
                                          self.class_gt: _test_batch[1]
                                      }), _iter_count)

                    del _test_batch

                # print("iter: ", _iter_count)
                _iter_count += 1
                self.writer.flush()
                del _train_batch

            _epoch_count += 1
            # Save model every epoch
            if self.log_dir is not None:
                self.saver.save(self.sess,
                                os.path.join(self.log_dir, "model.ckpt"), n)

    # ======= Net Component ========

    def _conv(self,
              inputs,
              filters,
              kernel_size=1,
              strides=1,
              pad='VALID',
              name='conv'):
        """ 
        Spatial Convolution 2D
        """
        with tf.variable_scope(name):
            kernel = tf.Variable(
                tf.contrib.layers.xavier_initializer(uniform=False)([
                    kernel_size, kernel_size,
                    inputs.get_shape().as_list()[3], filters
                ]),
                name='weights')
            bias = tf.Variable(tf.zeros([filters]), name='bias')

            conv = tf.nn.conv2d(inputs,
                                kernel, [1, strides, strides, 1],
                                padding=pad,
                                data_format='NHWC')
            conv_bias = tf.nn.bias_add(conv, bias)
            if self.w_summary:
                with tf.device('/cpu:0'):
                    self.summ_histogram_list.append(
                        tf.summary.histogram(name + 'weights',
                                             kernel,
                                             collections=['weight']))
                    self.summ_histogram_list.append(
                        tf.summary.histogram(name + 'bias',
                                             bias,
                                             collections=['bias']))
            return conv_bias

    def _conv_bn_relu(self,
                      inputs,
                      filters,
                      kernel_size=1,
                      strides=1,
                      pad='VALID',
                      name='conv_bn_relu'):
        """ 
        Spatial Convolution 2D + BatchNormalization + ReLU Activation
        """
        with tf.variable_scope(name):
            kernel = tf.Variable(
                tf.contrib.layers.xavier_initializer(uniform=False)([
                    kernel_size, kernel_size,
                    inputs.get_shape().as_list()[3], filters
                ]),
                name='weights')
            bias = tf.Variable(tf.zeros([filters]), name='bias')

            conv = tf.nn.conv2d(inputs,
                                kernel, [1, strides, strides, 1],
                                padding=pad,
                                data_format='NHWC')
            conv_bias = tf.nn.bias_add(conv, bias)
            norm = tf.contrib.layers.batch_norm(conv_bias,
                                                0.9,
                                                epsilon=1e-5,
                                                activation_fn=tf.nn.relu,
                                                is_training=self.training)
            if self.w_summary:
                with tf.device('/cpu:0'):
                    self.summ_histogram_list.append(
                        tf.summary.histogram(name + 'weights',
                                             kernel,
                                             collections=['weight']))
                    self.summ_histogram_list.append(
                        tf.summary.histogram(name + 'bias',
                                             bias,
                                             collections=['bias']))
            return norm

    def _fc(self, inputs, filters, name='fc'):
        inputs_shape = inputs.get_shape()
        # print(inputs_shape)
        if len(inputs_shape) > 2:
            inputs = tf.reshape(inputs, [
                -1, inputs_shape[1].value * inputs_shape[2].value *
                inputs_shape[3].value
            ],
                                name='flatten')
        with tf.variable_scope(name):
            # kernel = tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False)([n_in, filters]), name='weights')
            # bias = tf.Variable(tf.zeros([filters]), name='bias')
            fc = tf.layers.dense(
                inputs,
                units=filters,
            )

            return fc

    def _fc_bn_relu(self, inputs, filters, name='fc_bn_relu'):
        inputs_shape = inputs.get_shape()
        # print(inputs_shape)
        if len(inputs_shape) > 2:
            inputs = tf.reshape(inputs, [
                -1, inputs_shape[1].value * inputs_shape[2].value *
                inputs_shape[3].value
            ],
                                name='flatten')
        with tf.variable_scope(name):
            fc = tf.layers.dense(
                inputs,
                units=filters,
            )
            norm = tf.contrib.layers.batch_norm(fc,
                                                0.9,
                                                epsilon=1e-5,
                                                activation_fn=tf.nn.relu,
                                                is_training=self.training)
            return norm

    def _bn_relu(self, inputs, name='bn_relu'):
        with tf.variable_scope(name):
            norm = tf.contrib.layers.batch_norm(inputs,
                                                0.9,
                                                epsilon=1e-5,
                                                activation_fn=tf.nn.relu,
                                                is_training=self.training)
            return norm
