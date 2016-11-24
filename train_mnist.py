import os
import sys

import numpy as np
import tensorflow as tf
import scipy.misc


# Add ".." to system path
ML_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if ML_PATH not in sys.path:
    sys.path.append(ML_PATH)

from data_mgr import mnist_data_mgr
import gan_model_m8 as gan_m8
import gan_util


DATA_DIR = "mnist_data"
BATCH_SIZE = 128
IMAGE_DEPTH = 1
Z_DEPTH = 100
GAN_MODEL_CLASS = gan_m8.Model
MODEL_DIR = "mnist_train_log"
LR_D = 0.001
LR_G = 0.001
INIT_STDDEV = 0.02
TOTAL_NUM_STEPS = 100000
DUMP_STEPS = 100


def at_dump(step, zs, xs):
    for i in range(min(6, xs.shape[0])):
        x = xs[i,:,:,:]
        if x.shape[2] == 1:
            x = x[:,:,0]
        file_path = os.path.join(
            MODEL_DIR, "step-{:d}-ex{:d}.png".format(step, i))
        scipy.misc.toimage(x, cmin=0.0, cmax=1.0).save(file_path)


if __name__ == "__main__":
    data_mgr = mnist_data_mgr.DataMgr(
        BATCH_SIZE, binarize=False, data_dir=DATA_DIR)

    gan_util.train(
        data_mgr=data_mgr,
        gan_model_class=GAN_MODEL_CLASS,
        model_dir=MODEL_DIR,
        batch_size=BATCH_SIZE,
        image_depth=IMAGE_DEPTH,
        z_depth=Z_DEPTH,
        lr_D=LR_D,
        lr_G=LR_G,
        init_stddev=INIT_STDDEV,
        total_num_steps=TOTAL_NUM_STEPS,
        dump_steps=DUMP_STEPS,
        dump_callback=at_dump)
