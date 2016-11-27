import os

import numpy as np
import tensorflow as tf

import nn_util
import gan_model_image


_IMAGE_SIZE = 28


class DiscriminatorFactory(gan_model_image.DiscriminatorFactory):

    def __init__(self, image_depth=1):
        super(DiscriminatorFactory, self).__init__()
        self._image_depth = image_depth
        
    def _build_logits(self, train_phase, input_images):
        x = input_images
        x = tf.reshape(x, [-1, _IMAGE_SIZE, _IMAGE_SIZE, self._image_depth])

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
    

class GeneratorFactory(gan_model_image.GeneratorFactory):

    def __init__(self, z_depth=100, image_depth=1):
        super(GeneratorFactory, self).__init__()
        self._z_depth = z_depth
        self._image_depth = image_depth
        
    def _build_images(self, train_phase, input_zs):
        x = input_zs
        x = tf.reshape(x, [-1, 1, 1, self._z_depth])

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

        x = nn_util.deconv2d(
            x, 5, self._image_depth, 2, "SAME", label="deconv4_")
        # x = nn_util.batch_norm(train_phase, x, label="bn_deconv4_")
        x = tf.nn.sigmoid(x)
        return x


class Model(gan_model_image.Model):

    def __init__(self, images_batch_size, zs_batch_size,
                 image_depth=1, z_depth=100):
        d_factory = DiscriminatorFactory(image_depth)
        g_factory = GeneratorFactory(z_depth, image_depth)
        super(Model, self).__init__(
            images_batch_size, zs_batch_size,
            _IMAGE_SIZE, _IMAGE_SIZE, image_depth, z_depth,
            d_factory, g_factory)
