import os
import sys

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


def gen_data(gan_model_factory, model_path, batch_size, zs):
    if zs is None:
        assert batch_size > 0
    elif batch_size is None:
        batch_size = zs.shape[0]
    else:
        assert zs.shape[0] == batch_size
    
    with tf.Graph().as_default():
        with tf.variable_scope("gan"):
            m = gan_model_factory(images_batch_size=batch_size,
                                  zs_batch_size=batch_size)

        with tf.Session() as sess:
            m._saver.restore(sess, model_path)
            return m.run_generator(sess, zs)


def train(data_mgr, gan_model_factory, model_dir,
          batch_size, lr_D=0.0002, lr_G=0.0002, init_stddev=0.02,
          total_num_steps=100000, dump_steps=100,
          dump_callback=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    valid_data = data_mgr.valid_batch()

    with tf.Graph().as_default():
        common_init=tf.truncated_normal_initializer(
            mean=0.0, stddev=init_stddev, dtype=tf.float32)
        with tf.variable_scope("gan", initializer=common_init):
            m = gan_model_factory(images_batch_size=batch_size,
                                zs_batch_size=batch_size)
    
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            for i in range(total_num_steps + 1):
                for step_D in range(1):
                    train_data = data_mgr.train_batch()
                    train_loss_D = m.train_step_D(sess, train_data, None, lr_D)
                for step_G in range(1):
                    train_loss_G = m.train_step_G(sess, None, lr_G)
                valid_loss_D = m.test_step_D(sess, valid_data, None)
                valid_loss_G = m.test_step_G(sess, None)
                if i % dump_steps == 0:
                    m.save(sess, model_dir, i)
                    print ("step = {:d}\n"
                           "train_loss_D = {:.6f}, "
                           "train_loss_G = {:.6f}\n"
                           "valid_loss_D = {:.6f}, "
                           "valid_loss_G = {:.6f}").format(
                               i, train_loss_D, train_loss_G,
                               valid_loss_D, valid_loss_G)
                    print "=" * 60
                    sys.stdout.flush()
                    if dump_callback is not None:
                        dump_callback(i, *m.run_generator(sess, None))


def plot_images(images, cmap):
    for i in range(images.shape[0]):
        x = images[i,:,:,:]
        if x.shape[2] == 1:
            x = x[:,:,0]
        print "#{:d}".format(i)
        sys.stdout.flush()
        plt.imshow(x)
        if cmap is not None:
            plt.set_cmap(cmap)
        # plt.axis("off");
        plt.show()

