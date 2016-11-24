import os

import numpy as np
import tensorflow as tf

import nn_util
import gan_model_base


_IMAGE_SIZE = 28


class Discriminator(gan_model_base.Discriminator):

    def __init__(self, train_phase, input_images, image_depth=1):
        super(Discriminator, self).__init__(
            train_phase, input_images,
            _IMAGE_SIZE, _IMAGE_SIZE, image_depth)
        
    def _build(self, train_phase, input):
        x = input
        # x = nn_util.batch_norm(train_phase, x, label="bn_input_")
        
        x = nn_util.conv2d(x, 5, 32, 2, "SAME", label="conv1_")
        x = nn_util.batch_norm(train_phase, x, label="bn_conv1_")
        x = nn_util.leaky_relu(x, alpha=0.2)
        
        x = nn_util.conv2d(x, 5, 64, 2, "SAME", label="conv2_")
        x = nn_util.batch_norm(train_phase, x, label="bn_conv2_")
        x = nn_util.leaky_relu(x, alpha=0.2)

        x = nn_util.conv2d(x, 5, 128, 1, "VALID", label="conv3_")
        x = nn_util.batch_norm(train_phase, x, label="bn_conv3_")
        x = nn_util.leaky_relu(x, alpha=0.2)

        x = nn_util.flatten(x)
        x = nn_util.linear(x, 1, "fc_")
        return x
    

class Generator(gan_model_base.Generator):

    def __init__(self, train_phase, input_zs, z_depth=100, image_depth=1):
        super(Generator, self).__init__(
            train_phase, input_zs, z_depth, image_depth)
        
    def _build(self, train_phase, input, image_depth):
        x = input
        x = nn_util.batch_norm(train_phase, x, label="bn_input_")
        
        x = nn_util.deconv2d(x, 3, 128, 1, "VALID", label="deconv1_")
        x = nn_util.batch_norm(train_phase, x, label="bn_deconv1_")
        x = tf.nn.relu(x)

        x = nn_util.deconv2d(x, 5, 64, 1, "VALID", label="deconv2_")
        x = nn_util.batch_norm(train_phase, x, label="bn_deconv2_")
        x = tf.nn.relu(x)

        x = nn_util.deconv2d(x, 5, 32, 2, "SAME", label="deconv3_")
        x = nn_util.batch_norm(train_phase, x, label="bn_deconv3_")
        x = tf.nn.relu(x)

        x = nn_util.deconv2d(x, 5, image_depth, 2, "SAME", label="deconv4_")
        # x = nn_util.batch_norm(train_phase, x, label="bn_deconv4_")
        x = tf.nn.sigmoid(x)
        return x


class Model(gan_model_base.Model):

    def __init__(self, images_batch_size, zs_batch_size,
                 image_depth=1, z_depth=100):
        super(Model, self).__init__(
            images_batch_size, zs_batch_size,
            _IMAGE_SIZE, _IMAGE_SIZE, image_depth, z_depth,
            Discriminator, Generator)