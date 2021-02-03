"""
The base class of the Model.
"""

import os
import tensorflow as tf
import cv2
import numpy as np
from matplotlib import pyplot as plt

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
                 gpu_memory_fraction=0.2,   # 1.0
                 train_vars='segm',
                 training=True,
                 w_summary=True,
                 alpha=0.6,
                 beta=2):
        
        # Configure GPU
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
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
        self.train_vars = train_vars
        
        #   step learning rate policy
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(self.base_lr,
            self.global_step, self.lr_decay_freq*self.epoch_size*5, self.lr_decay,
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
        self.gt = tf.placeholder(tf.float32, shape=[None,], name='label')
        self.class_gt = tf.placeholder(tf.float32, shape=[None,], name='class')
        self.center = tf.get_variable(dtype=tf.float32, shape=[2,], name='center',
                                      initializer=tf.constant_initializer(0),
                                      trainable=False)
        self.center_label = 0

        print ("--Placeholder Built")
    
    
    def __build_train_op(self):
        """ 
        Build loss and optimizer
        """
        # Select part of parameters to train

        # Loss
        with tf.name_scope('loss'):
            choose_neg_gt = self.gt
            choose_neg = self.predmap
            choose_neg = tf.gather_nd(choose_neg, tf.where(choose_neg_gt < 1e-6))
            neg_shape = choose_neg.get_shape()
            if self.center_label:
                self.center = self.center + self.alpha * (tf.reduce_sum(choose_neg, 0) / (1. + neg_shape[0]) - self.center)
            else:
                self.center = tf.reduce_mean(choose_neg, 0)
                self.center_label = 1

            # self.neg_loss = tf.pow(self.predmap - self.center, 2) # name='Neg_loss'
            # self.fp_loss = tf.pow((self.predmap - self.center), 2) / self.beta # , name='FP_loss'

            self.neg_loss = tf.sqrt(tf.reduce_sum(tf.square(self.predmap - self.center), 1)) - 0.5
            self.fp_loss = tf.sqrt(tf.reduce_sum(tf.square(self.predmap - self.center), 1)) - 2
            euclidean = tf.sqrt(tf.reduce_sum(tf.square(self.predmap - self.center), 1))
            self.pos_loss = 12 - euclidean
            # self.pos_loss = tf.pow((0.1 + self.beta * self.beta) / (0.1 + euclidean), 1) # , name='Pos_loss'

            # choose_fp_gt = self.fp
            self.neg_loss = tf.gather_nd(self.neg_loss, tf.where(choose_neg_gt < 1e-6))
            zero = tf.zeros_like(self.neg_loss)
            self.neg_loss = tf.where(self.neg_loss > 0, x=self.neg_loss, y=zero)
            self.neg_loss = tf.pow(self.neg_loss, 2)

            self.fp_loss = tf.gather_nd(self.fp_loss, tf.where(choose_neg_gt < 1 + 1e-6))
            choose_fp_gt2 = tf.gather_nd(choose_neg_gt, tf.where(choose_neg_gt < 1 + 1e-6))
            self.fp_loss = tf.gather_nd(self.fp_loss, tf.where(choose_fp_gt2 > 1 - 1e-6))
            zero = tf.zeros_like(self.fp_loss)
            self.fp_loss = tf.where(self.fp_loss > 0, x=self.fp_loss, y=zero)
            self.fp_loss = tf.pow(self.fp_loss, 2)

            self.pos_loss = tf.gather_nd(self.pos_loss, tf.where(choose_neg_gt > 2 - 1e-6))
            zero = tf.zeros_like(self.pos_loss)
            self.pos_loss = tf.where(self.pos_loss > 0, x=self.pos_loss, y=zero)
            self.pos_loss = tf.pow(self.pos_loss, 2)

            # self.class_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.class_gt, logits=self.pred, name='Class_Loss'))
            self.class_loss = tf.reduce_mean(
                self.sigmoid_focal_crossentropy(labels=self.class_gt, logits=self.pred, alpha=0.75))

            self.neg_loss = tf.reduce_mean(self.neg_loss)
            self.pos_loss = tf.reduce_mean(self.pos_loss)
            self.fp_loss  = tf.reduce_mean(self.fp_loss)

            
            # if self.train_vars == 'segm':
            trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            self.l2_regular = self.weight_decay * tf.reduce_sum(
                    [tf.nn.l2_loss(var) for var in trainable_vars], name='L2_Regularization')

            self.loss = self.neg_loss + self.fp_loss + self.pos_loss + 10 * self.class_loss
                
            self.summ_scalar_list.append(tf.summary.scalar("total loss", self.loss))
            self.summ_scalar_list.append(tf.summary.scalar("l2_regularization", self.l2_regular))
            self.summ_scalar_list.append(tf.summary.scalar("lr", self.learning_rate))
        print ("--Loss & Scalar_summary Built")
        
        # Optimizer
        with tf.name_scope('optimizer'):
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):               
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
                self.train_step = self.optimizer.minimize(self.loss, var_list=trainable_vars,
                    global_step=self.global_step)
        print ("--Optimizer Built")


        
    def BuildModel(self):
        """ 
        Build model in tensorflow session
        """
        # Input
        with tf.name_scope('input'):
            self.__build_ph()
        assert self.img != None and self.gt != None

        self.predmap, self.pred = self.net(self.img) #, self.pred
        print(self.pred)
        self.pred = tf.reshape(self.pred, [-1])
        print(self.img)

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
            # self.dataset._randomize()
            for m in range(self.epoch_size):
                _train_batch = next(self.train_generator)

                self.sess.run(self.train_step,
                              feed_dict={self.img: _train_batch[0],
                                         self.gt: _train_batch[1],
                                         self.class_gt: _train_batch[3]})
                
                if _iter_count % 20 == 0:
                    _test_batch = next(self.val_generator)
                    # Print training loss
                    center, predmap, loss, pos_loss, neg_loss, fp_loss, class_loss = \
                        self.sess.run([self.center, self.predmap, self.loss, self.pos_loss, self.neg_loss, self.fp_loss, self.class_loss],  # , self.embedding_loss],
                                  feed_dict={self.img: _train_batch[0],
                                             self.gt: _train_batch[1],
                                             self.class_gt: _train_batch[3]})
                    print ('--Epoch: ', _epoch_count, ' iter: ', _iter_count,
                           ' train_loss: ',[loss, pos_loss, neg_loss, fp_loss, class_loss])

                    # Record validation results
                                                 
                    self.writer.add_summary(
                        self.sess.run(self.summ_scalar,
                                      feed_dict={self.img: _test_batch[0],
                                                 self.gt: _test_batch[1],
                                                 self.class_gt: _test_batch[3]}), _iter_count)

                    del _test_batch
                                                     
                # print("iter: ", _iter_count)
                _iter_count += 1
                self.writer.flush()
                del _train_batch



            _epoch_count += 1
            # Save model every epoch
            if self.log_dir is not None:
                self.saver.save(self.sess, os.path.join(self.log_dir, "model.ckpt"), n)
        # print(center)
    
    
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

    def _fc(self, inputs, filters, name='fc'):
        inputs_shape = inputs.get_shape()
        # print(inputs_shape)
        if len(inputs_shape) > 2:
            inputs = tf.reshape(inputs, [-1, inputs_shape[1].value * inputs_shape[2].value * inputs_shape[3].value], name='flatten')
        with tf.variable_scope(name):
            # kernel = tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False)([n_in, filters]), name='weights')
            # bias = tf.Variable(tf.zeros([filters]), name='bias')
            fc = tf.layers.dense(inputs, units=filters, )

            return fc

    def _fc_bn_relu(self, inputs, filters, name='fc_bn_relu'):
        inputs_shape = inputs.get_shape()
        # print(inputs_shape)
        if len(inputs_shape) > 2:
            inputs = tf.reshape(inputs, [-1, inputs_shape[1].value * inputs_shape[2].value * inputs_shape[3].value], name='flatten')
        with tf.variable_scope(name):
            fc = tf.layers.dense(inputs, units=filters, )
            norm = tf.contrib.layers.batch_norm(fc, 0.9, epsilon=1e-5, activation_fn=tf.nn.relu,
                                                is_training=self.training)
            return norm

    def _bn_relu(self, inputs, name='bn_relu'):
        with tf.variable_scope(name):
            norm = tf.contrib.layers.batch_norm(inputs, 0.9, epsilon=1e-5, activation_fn=tf.nn.relu,
                                                is_training=self.training)
            return norm

    def sigmoid_focal_crossentropy(self,
        labels,
        logits,
        alpha = 0.75,
        gamma = 2.0,
        from_logits: bool = True,
    ):
        if gamma and gamma < 0:
            raise ValueError("Value of gamma should be greater than or equal to zero")

        logits = tf.convert_to_tensor(logits, dtype=tf.float32)
        labels = tf.convert_to_tensor(labels, dtype=logits.dtype)

        # Get the cross_entropy for each entry
        ce = tf.reshape(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels, name='Class_Loss'), [-1])

        # If logits are provided then convert the predictions into probabilities
        if from_logits:
            pred_prob = tf.sigmoid(logits)
        else:
            pred_prob = logits

        p_t = (labels * pred_prob) + ((1 - labels) * (1 - pred_prob))
        alpha_factor = 1.0
        modulating_factor = 1.0

        if alpha:
            alpha = tf.convert_to_tensor(alpha, dtype=tf.float32)
            alpha_factor = labels * alpha + (1 - labels) * (1 - alpha)

        if gamma:
            gamma = tf.convert_to_tensor(gamma, dtype=tf.float32)
            modulating_factor = tf.pow((1.0 - p_t), gamma)

        # compute the final loss and return
        return tf.reduce_sum(alpha_factor * modulating_factor * ce, axis=-1)
