import os

import numpy as np
import tensorflow as tf

import nn_util


class Discriminator(object):

    def __init__(self, train_phase, input_images,
                 image_height, image_width, image_depth):
        input_images = tf.reshape(
            input_images, [-1, image_height, image_width, image_depth])
        logits = self._build(train_phase, input_images)
        loss_neg = tf.nn.relu(logits) + tf.log(1 + tf.exp(-tf.abs(logits)))
        loss_pos = loss_neg - logits
        
        self._train_phase = train_phase
        self._input_images = input_images
        self._image_height = image_height
        self._image_width = image_width
        self._image_height = image_height
        self._image_depth = image_depth
        self._logits = logits
        # loss as if input_images all have positive labels.
        self._loss_pos = tf.reduce_mean(loss_pos)
        # loss as if input_images all have negative labels.
        self._loss_neg = tf.reduce_mean(loss_neg)
        
    def _build(self, train_phase, input):
        x = input
        x = nn_util.flatten(x)
        x = nn_util.linear(x, 1, "fc_")
        return x


class Generator(object):

    def __init__(self, train_phase, input_zs, z_depth, image_depth):
        input_zs = tf.reshape(input_zs, [-1, 1, 1, z_depth])
        images = self._build(train_phase, input_zs, image_depth)
        
        self._train_phase = train_phase
        self._input_zs = input_zs
        self._z_depth = z_depth
        self._image_depth = image_depth
        self._images = images

    def _build(self, train_phase, input, image_depth):
        return input


class Model(object):

    def __init__(self, images_batch_size, zs_batch_size,
                 image_height, image_width, image_depth, z_depth,
                 discriminator_class, generator_class):
        train_phase = tf.placeholder(shape=[], dtype=tf.bool)
        input_images = tf.placeholder(
            shape=[images_batch_size, image_height, image_width, image_depth],
            dtype=tf.float32)
        input_zs = tf.placeholder(shape=[zs_batch_size, z_depth],
                                  dtype=tf.float32)
        params_tracker = nn_util.ParamsTracker()
        with tf.variable_scope("gan"):
            with tf.variable_scope("G"):
                m_G = generator_class(
                    train_phase, input_zs, z_depth, image_depth)
                m_G_params = params_tracker.get_params()
            with tf.variable_scope("D"):
                m_Ddata = discriminator_class(
                    train_phase, input_images, image_depth)
                m_D_params = params_tracker.get_params()
                tf.get_variable_scope().reuse_variables()
                m_Dg = discriminator_class(
                    train_phase, m_G._images, image_depth)

        batch_size_total = images_batch_size + zs_batch_size
        loss_D = (m_Ddata._loss_pos * images_batch_size +
                  m_Dg._loss_neg * zs_batch_size) / float(batch_size_total)
        loss_G = m_Dg._loss_pos

        lr = tf.placeholder(shape=[], dtype=tf.float32)
        optimizer = tf.train.AdamOptimizer(lr, beta1=0.5)
        train_op_D = optimizer.minimize(loss=loss_D, var_list=m_D_params)
        train_op_G = optimizer.minimize(loss=loss_G, var_list=m_G_params)

        saver = tf.train.Saver(tf.all_variables())
        
        self._images_batch_size = images_batch_size
        self._zs_batch_size = zs_batch_size
        self._image_height = image_height
        self._image_width = image_width
        self._image_depth = image_depth
        self._z_depth = z_depth
        self._train_phase = train_phase
        self._input_images = input_images
        self._input_zs = input_zs
        self._m_Ddata = m_Ddata
        self._m_Dg = m_Dg
        self._m_G = m_G
        self._m_D_params = m_D_params
        self._m_G_params = m_G_params
        self._loss_D = loss_D
        self._loss_G = loss_G
        self._lr = lr
        self._train_op_D = train_op_D
        self._train_op_G = train_op_G
        self._saver = saver

    def _feed_images(self, images):
        return np.reshape(images, self._input_images.get_shape())

    def _feed_zs(self, zs):
        shape = self._input_zs.get_shape()
        if zs is None:
            return np.random.uniform(0.0, 1.0, shape)
        return np.reshape(zs, shape)
    
    def train_step_D(self, sess, images, zs, lr):
        feed_dict = {
            self._train_phase: True,
            self._input_images: self._feed_images(images),
            self._input_zs: self._feed_zs(zs),
            self._lr: lr
        }
        _, loss_D = sess.run([self._train_op_D, self._loss_D],
                             feed_dict=feed_dict)
        return loss_D

    def train_step_G(self, sess, zs, lr):
        feed_dict = {
            self._train_phase: True,
            self._input_zs: self._feed_zs(zs),
            self._lr: lr
        }
        _, loss_G = sess.run([self._train_op_G, self._loss_G],
                             feed_dict=feed_dict)
        return loss_G
    
    def test_step_D(self, sess, images, zs):
        feed_dict = {
            self._train_phase: False,
            self._input_images: self._feed_images(images),
            self._input_zs: self._feed_zs(zs)
        }
        return sess.run(self._loss_D, feed_dict=feed_dict)

    def test_step_G(self, sess, zs):
        feed_dict = {
            self._train_phase: False,
            self._input_zs: self._feed_zs(zs)
        }
        return sess.run(self._loss_G, feed_dict=feed_dict)
    
    def save(self, sess, model_dir, global_step):
        self._saver.save(sess, os.path.join(model_dir, "sav"),
                         global_step=global_step)
    
    def run_generator(self, sess, zs):
        zs = self._feed_zs(zs)
        feed_dict = {
            self._train_phase: False,
            self._input_zs: zs
        }
        return zs, sess.run(self._m_G._images, feed_dict=feed_dict)
