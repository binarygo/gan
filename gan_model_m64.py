import os

import numpy as np
import tensorflow as tf

import nn_util
import gan_model_image


_IMAGE_SIZE = 64


class Discriminator(gan_model_image.Discriminator):

    def __init__(self, train_phase, input_images, image_depth=3):
        super(Discriminator, self).__init__(
            train_phase, input_images, _IMAGE_SIZE, _IMAGE_SIZE, image_depth)

    def _build_logits(self, train_phase, input_images):
        x = input_images
        
        # x = nn_util.batch_norm(train_phase, x, label="bn_input_")
        
        x = nn_util.conv2d(x, 5, 128, 2, "SAME", label="conv1_")
        x = nn_util.batch_norm(train_phase, x, label="bn_conv1_")
        x = nn_util.leaky_relu(x, alpha=0.2)
        
        x = nn_util.conv2d(x, 5, 256, 2, "SAME", label="conv2_")
        x = nn_util.batch_norm(train_phase, x, label="bn_conv2_")
        x = nn_util.leaky_relu(x, alpha=0.2)

        x = nn_util.conv2d(x, 5, 512, 2, "SAME", label="conv3_")
        x = nn_util.batch_norm(train_phase, x, label="bn_conv3_")
        x = nn_util.leaky_relu(x, alpha=0.2)

        x = nn_util.conv2d(x, 5, 1024, 2, "SAME", label="conv4_")
        x = nn_util.batch_norm(train_phase, x, label="bn_conv4_")
        x = nn_util.leaky_relu(x, alpha=0.2)        
        
        x = nn_util.flatten(x)
        x = nn_util.linear(x, 1, "fc_")
        return x
    

class Generator(gan_model_image.Generator):

    def __init__(self, train_phase, input_zs, z_depth=100, image_depth=3):
        self._image_depth = image_depth
        super(Generator, self).__init__(train_phase, input_zs, z_depth)

    def _build_images(self, train_phase, input_zs):
        x = input_zs
        
        x = nn_util.batch_norm(train_phase, x, label="bn_input_")
        
        x = nn_util.deconv2d(x, 4, 1024, 1, "VALID", label="deconv1_")
        x = nn_util.batch_norm(train_phase, x, label="bn_deconv1_")
        x = tf.nn.relu(x)

        x = nn_util.deconv2d(x, 5, 512, 2, "SAME", label="deconv2_")
        x = nn_util.batch_norm(train_phase, x, label="bn_deconv2_")
        x = tf.nn.relu(x)

        x = nn_util.deconv2d(x, 5, 256, 2, "SAME", label="deconv3_")
        x = nn_util.batch_norm(train_phase, x, label="bn_deconv3_")
        x = tf.nn.relu(x)

        x = nn_util.deconv2d(x, 5, 128, 2, "SAME", label="deconv4_")
        x = nn_util.batch_norm(train_phase, x, label="bn_deconv4_")
        x = tf.nn.relu(x)
        
        x = nn_util.deconv2d(
            x, 5, self._image_depth, 2, "SAME", label="deconv5_")
        # x = nn_util.batch_norm(train_phase, x, label="bn_deconv5_")
        x = tf.nn.sigmoid(x)
        return x


class Model(gan_model_image.Model):

    def __init__(self, images_batch_size, zs_batch_size,
                 image_depth=3, z_depth=100):
        def d_factory(train_phase, input_images):
            return Discriminator(train_phase, input_images, image_depth)

        def g_factory(train_phase, input_zs):
            return Generator(train_phase, input_zs, z_depth, image_depth)

        super(Model, self).__init__(
            images_batch_size, zs_batch_size,
            _IMAGE_SIZE, _IMAGE_SIZE, image_depth, z_depth,
            d_factory, g_factory)
