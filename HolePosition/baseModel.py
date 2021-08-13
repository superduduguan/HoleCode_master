"""
The base class of the Model.
"""

import os
import tensorflow as tf
import cv2
import numpy as np





class BaseModel(object):
    
    def __init__(self,
                 dataset=None,
                 logdir='logs',
                 in_size_h=352,
                 in_size_w=608,
                 pool_scale=8,
                 weight_decay=0.0001,
                 base_lr=0.001,
                 epoch=50,
                 epoch_size=1000,
                 lr_decay=0.95,
                 lr_decay_freq=4,
                 batch_size=8,
                 gpu_memory_fraction=0.1,
                 train_vars='segm',
                 training=True,
                 w_summary=True):
        
        # Configure GPU
        gpu_options = tf.GPUOptions(allow_growth = True )
        config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True, gpu_options=gpu_options)
        tf.reset_default_graph()
        self.sess = tf.Session(config=config)
        
        # Set log dir
        self.log_dir = logdir
        self.writer = tf.summary.FileWriter(self.log_dir)
        
        # Net args        
        self.dataset = dataset
        self.in_size_w = in_size_w
        self.in_size_h = in_size_h
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
        self.train_vars = train_vars
        
        #   step learning rate policy
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(self.base_lr,
            self.global_step, self.lr_decay_freq*self.epoch_size, self.lr_decay,
            staircase=True)
            
        #   Inside Variable
        self.train_step = None
        self.loss = None
        self.w_summary = w_summary

        self.img = None
        # self.gtmap = None
        self.gt = None

        self.summ_scalar_list = []
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
        self.img = tf.placeholder(tf.float32, 
            shape=[None, self.in_size_h, self.in_size_w, 3], name='img_in')
        self.gt = tf.placeholder(tf.float32, shape=[None, 3], name='gt_location')
        # self.gt = tf.placeholder(tf.float32, shape=[None, 1], name='gt_label')
        print ("--Placeholder Built")
    
    
    def __build_train_op(self):
        """ 
        Build loss and optimizer
        """
        # Select part of parameters to train
        # assert self.train_vars in ['segm', 'class'], 'Unrecognized trainable parameters!'
        
        # Loss
        with tf.name_scope('loss'):
            # self.segm_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
            #    labels=self.gtmap, logits=self.predmap, name='Segm_Loss'))
            # self.segm_loss = tf.nn.l2_loss(self.gtmap - tf.nn.sigmoid(self.predmap))


            self.locate_loss = tf.reduce_mean(tf.abs(self.predmap - self.gt, name='Locate_loss'))
            # self.class_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
            #    labels=self.gt, logits=self.pred, name='Class_Loss'))
            
            # if self.train_vars == 'segm':
            trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            self.l2_regular = self.weight_decay * tf.reduce_sum(
                    [tf.nn.l2_loss(var) for var in trainable_vars], name='L2_Regularization')
            # self.loss = self.locate_loss + self.class_loss + self.l2_regular
            self.loss = self.locate_loss + self.l2_regular
            # elif self.train_vars == 'class':
            #     trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'AnomalyDetection/Classification')
            #     self.l2_regular = self.weight_decay * tf.reduce_sum(
            #         [tf.nn.l2_loss(var) for var in trainable_vars], name='L2_Regularization')
            #     self.loss = self.class_loss
                
            self.summ_scalar_list.append(tf.summary.scalar("total lossx", self.loss))
            self.summ_scalar_list.append(tf.summary.scalar("l2_regularization", self.l2_regular))
            self.summ_scalar_list.append(tf.summary.scalar("lr", self.learning_rate))
        print ("--Loss & Scalar_summary Built")
        
        # Optimizer
        with tf.name_scope('optimizer'):
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):               
                # self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
                self.train_step = self.optimizer.minimize(self.loss / self.batch_size, var_list=trainable_vars,
                    global_step=self.global_step)
        print ("--Optimizer Built")
            

    def _build_AveragePixelError(self):
        """
        Computes accuracy tensor
        """
        # score_threshold = 0.7
        with tf.name_scope('APE'):
            ape = tf.abs(self.predmap - self.gt)
            ape_center = tf.reduce_mean(ape[:, :2])
            ape_size   = tf.reduce_mean(ape[:, 2])
            self.summ_axis_list.append(tf.summary.scalar('CenterPointError', ape_center))
            self.summ_size_list.append(tf.summary.scalar('RadiusError', ape_size))
        print ("--AveragePixelError_summary Built")

        
    def BuildModel(self):
        """ 
        Build model in tensorflow session
        """
        # Input
        with tf.name_scope('input'):
            self.__build_ph()
        assert self.img != None and self.gt != None

        self.predmap = self.net(self.img)
        print(self.img)
        print('8\n', self.predmap)

        if self.training:
            # train op
            with tf.name_scope('train'):
                self.__build_train_op()
            # with tf.name_scope('monitor'):
            #     self.__build_monitor()
            with tf.name_scope('AveragePixelError'):
                self._build_AveragePixelError()
                
        # initialize all variables
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        if self.training:
            # merge all summary
            # self.summ_image = tf.summary.merge(self.summ_image_list)
            self.summ_scalar = tf.summary.merge(self.summ_scalar_list)
            self.summ_APE_CenterPoint = tf.summary.merge(self.summ_axis_list)
            self.summ_APE_Radius = tf.summary.merge(self.summ_size_list)
            # self.summ_histogram = tf.summary.merge(self.summ_histogram_list)
        self.writer.add_graph(self.sess.graph)
        print(" -- Model Built")
        
        
    def restore_sess(self, model=None):
        """ 
        Restore session from ckpt format file
        """

        if model is not None:
            if self.training:
                variables_to_restore = tf.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES, 'HoleDetection/LocationResult')
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
        self.train_generator = self.dataset._batch_generator(self.batch_size, normalize=True, sample_set='train')
        self.val_generator = self.dataset._batch_generator(self.batch_size, normalize=True, sample_set='val')
        
        # Training process
        for n in range(self.epoch):
            for m in range(self.epoch_size):
                _train_batch = next(self.train_generator)

                # print(self.train_step)

                self.sess.run(self.train_step,
                              feed_dict={self.img: _train_batch[0],
                                         self.gt: _train_batch[1]})
                
                if _iter_count % 20 == 0:
                    _test_batch = next(self.val_generator)
                    # Print training loss
                    print ('--Epoch: ', _epoch_count, ' iter: ', _iter_count, ' train_loss: ',
                           self.sess.run(self.loss,
                                         feed_dict={self.img: _train_batch[0],
                                                    self.gt: _train_batch[1]}))
                    
                    # Record validation results
                                                 
                    self.writer.add_summary(
                        self.sess.run(self.summ_scalar,
                                      feed_dict={self.img: _test_batch[0],
                                                 self.gt: _test_batch[1]}), _iter_count)
                    self.writer.add_summary(
                        self.sess.run(self.summ_APE_CenterPoint,
                                      feed_dict={self.img: _test_batch[0],
                                                 self.gt: _test_batch[1]}), _iter_count)
                    self.writer.add_summary(
                        self.sess.run(self.summ_APE_Radius,
                                      feed_dict={self.img: _test_batch[0],
                                                 self.gt: _test_batch[1]}), _iter_count)

                    del _test_batch
                                                     
                # print("iter: ", _iter_count)
                _iter_count += 1
                self.writer.flush()
                del _train_batch



            _epoch_count += 1
            # Save model every epoch
            if self.log_dir is not None:
                self.saver.save(self.sess, os.path.join(self.log_dir, "model.ckpt"), n)
    
    
    # ======= Net Component ========
    
    def _conv(self, inputs, filters, kernel_size=1, strides=1, pad='VALID', name='conv'):
        """ 
        Spatial Convolution 2D
        """
        with tf.variable_scope(name):
            kernel = tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False)([kernel_size, kernel_size, inputs.get_shape().as_list()[3], filters]), name='weights')
            bias = tf.Variable(tf.zeros([filters]), name='bias')

            conv = tf.nn.conv2d(inputs, kernel, [1, strides, strides, 1], padding=pad, data_format='NHWC')
            conv_bias = tf.nn.bias_add(conv, bias)
            if self.w_summary:
                with tf.device('/cpu:0'):
                    self.summ_histogram_list.append(tf.summary.histogram(name+'weights', kernel, collections=['weight']))
                    self.summ_histogram_list.append(tf.summary.histogram(name+'bias', bias, collections=['bias']))
            return conv_bias


    def _conv_bn_relu(self, inputs, filters, kernel_size=1, strides=1, pad='VALID', name='conv_bn_relu'):
        """ 
        Spatial Convolution 2D + BatchNormalization + ReLU Activation
        """
        with tf.variable_scope(name):
            kernel = tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False)([kernel_size,kernel_size, inputs.get_shape().as_list()[3], filters]), name='weights')
            bias = tf.Variable(tf.zeros([filters]), name='bias')

            conv = tf.nn.conv2d(inputs, kernel, [1, strides, strides, 1], padding=pad, data_format='NHWC')
            conv_bias = tf.nn.bias_add(conv, bias)
            norm = tf.contrib.layers.batch_norm(conv_bias, 0.9, epsilon=1e-5, activation_fn=tf.nn.relu, is_training=self.training)
            if self.w_summary:
                with tf.device('/cpu:0'):
                    self.summ_histogram_list.append(tf.summary.histogram(name+'weights', kernel, collections=['weight']))
                    self.summ_histogram_list.append(tf.summary.histogram(name+'bias', bias, collections=['bias']))
            return norm

    def _fc(self, inputs, filters, name='fc_relu'):
        inputs_shape = inputs.get_shape()
        # print(inputs_shape)
        inputs = tf.reshape(inputs, [-1, inputs_shape[1].value * inputs_shape[2].value * inputs_shape[3].value], name='flatten')
        with tf.variable_scope(name):
            # kernel = tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False)([n_in, filters]), name='weights')
            # bias = tf.Variable(tf.zeros([filters]), name='bias')
            fc = tf.layers.dense(inputs, units=filters)

            return fc

