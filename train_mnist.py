import os
import sys
from functools import partial

import numpy as np
import tensorflow as tf
import scipy.misc


# Add ".." to system path
ML_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if ML_PATH not in sys.path:
    sys.path.append(ML_PATH)

from data_mgr import mnist_data_mgr
import gan_model_m28 as gan_m28
import gan_util


DATA_DIR = "mnist_data"
BATCH_SIZE = 128
IMAGE_DEPTH = 1
Z_DEPTH = 100
GAN_MODEL_FACTORY = partial(
    gan_m28.Model, image_depth=IMAGE_DEPTH, z_depth=Z_DEPTH)
MODEL_DIR = "mnist_train_log"
LR_D = 0.001
LR_G = 0.001
INIT_STDDEV = 0.02
TOTAL_NUM_STEPS = 100000
DUMP_STEPS = 100


def at_dump(step, zs, xs):
    return gan_util.dump_images(step, zs, xs, 6, MODEL_DIR)


if __name__ == "__main__":
    data_mgr = mnist_data_mgr.DataMgr(
        batch_size=BATCH_SIZE, binarize=False, data_dir=DATA_DIR)

    gan_util.train(
        data_mgr=data_mgr,
        gan_model_factory=GAN_MODEL_FACTORY,
        model_dir=MODEL_DIR,
        batch_size=BATCH_SIZE,
        lr_D=LR_D,
        lr_G=LR_G,
        init_stddev=INIT_STDDEV,
        total_num_steps=TOTAL_NUM_STEPS,
        dump_steps=DUMP_STEPS,
        dump_callback=at_dump)
