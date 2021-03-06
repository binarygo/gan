import os

import numpy as np
import tensorflow as tf

import nn_util


class Discriminator(object):

    def __init__(self, train_phase, targets, input_images,
                 image_height, image_width, image_depth):
        targets = tf.reshape(targets, [-1, 1])
        input_images = tf.reshape(
            input_images, [-1, image_height, image_width, image_depth])
        logits = self._build_logits(train_phase, input_images)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits, targets)
        loss = tf.reduce_mean(loss)

        self._train_phase = train_phase
        self._targets = targets
        self._input_images = input_images
        self._image_height = image_height
        self._image_width = image_width
        self._image_depth = image_depth
        self._logits = logits
        self._loss = loss

    def _build_logits(self, train_phase, input_images):
        x = input_images
        x = nn_util.flatten(x)
        x = nn_util.linear(x, 1, "fc_")
        return x


class Generator(object):

    def __init__(self, train_phase, input_zs, z_depth):
        input_zs = tf.reshape(input_zs, [-1, 1, 1, z_depth])
        images = self._build_images(train_phase, input_zs)

        self._train_phase = train_phase
        self._input_zs = input_zs
        self._z_depth = z_depth
        self._images = images
        
    def _build_images(self, train_phase, input_zs):
        return input_zs


class Model(object):

    def __init__(self, images_batch_size, zs_batch_size,
                 image_height, image_width, image_depth, z_depth,
                 discriminator_factory, generator_factory):
        global_step = tf.Variable(0, trainable=False)
        inc_global_step = tf.assign(global_step, global_step + 1)
        train_phase = tf.placeholder(shape=[], dtype=tf.bool)
        input_images = tf.placeholder(
            shape=[images_batch_size, image_height, image_width, image_depth],
            dtype=tf.float32)
        # scale input images from [0, 1] to [-1, 1]
        scaled_input_images = 2.0 * input_images - 1.0
        input_zs = tf.placeholder(shape=[zs_batch_size, z_depth],
                                  dtype=tf.float32)

        params_tracker = nn_util.ParamsTracker()
        with tf.variable_scope("gan"):
            t_false = tf.constant(False)
            with tf.variable_scope("G"):
                m_G = generator_factory(train_phase, input_zs)
                tf.get_variable_scope().reuse_variables()
                m_Gtest = generator_factory(t_false, input_zs)
                m_G_params = params_tracker.get_params()
            with tf.variable_scope("D"):
                m_Ddata = discriminator_factory(
                    train_phase,
                    nn_util.make_pos_neg_targets(images_batch_size, 0),
                    scaled_input_images)
                tf.get_variable_scope().reuse_variables()
                m_Dg_neg = discriminator_factory(
                    train_phase,
                    nn_util.make_pos_neg_targets(0, zs_batch_size),
                    m_G._images)
                m_Dg_pos = discriminator_factory(
                    train_phase,
                    nn_util.make_pos_neg_targets(zs_batch_size, 0),
                    m_G._images)
                m_D_params = params_tracker.get_params()

        total_batch_size = images_batch_size + zs_batch_size
        loss_D = (m_Ddata._loss * images_batch_size +
                  m_Dg_neg._loss * zs_batch_size) / float(total_batch_size)
        loss_G = m_Dg_pos._loss

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
        self._global_step = global_step
        self._inc_global_step = inc_global_step
        self._train_phase = train_phase
        self._input_images = input_images
        self._input_zs = input_zs
        self._m_G = m_G
        self._m_Gtest = m_Gtest
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
            return np.random.uniform(-1.0, 1.0, shape)
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

    def global_step(self, sess):
        return sess.run(self._global_step)
    
    def inc_global_step(self, sess):
        return sess.run(self._inc_global_step)
    
    def save(self, sess, model_dir):
        self._saver.save(sess, os.path.join(model_dir, "sav"),
                         global_step=self._global_step)

    def restore(self, sess, model_path):
        self._saver.restore(sess, model_path)
        
    def run_generator(self, sess, zs):
        zs = self._feed_zs(zs)
        feed_dict = {
            self._train_phase: False,
            self._input_zs: zs
        }
        xs = sess.run(self._m_Gtest._images, feed_dict=feed_dict)
        return zs, (xs + 1.0) / 2.0
