import os
import sys

import numpy as np
import tensorflow as tf
import scipy.misc

# Add ".." to system path
ML_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if ML_PATH not in sys.path:
    sys.path.append(ML_PATH)

from data_mgr import anime_face_data_mgr
import gan_model_m28 as gan_m28
import gan_model_m64 as gan_m64
import gan_util


BATCH_SIZE = 128
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
IMAGE_DEPTH = 3
Z_DEPTH = 100
FORCE_GRAY_SCALE = False
GAN_MODEL_CLASS = gan_m64.Model
MODEL_DIR = "anime_face_train_log"
LR_D = 0.0002
LR_G = 0.0002
INIT_STDDEV = 0.02
TOTAL_NUM_STEPS = 50000
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
    data_mgr = anime_face_data_mgr.DataMgr(
        batch_size=BATCH_SIZE,
        image_width=IMAGE_WIDTH, image_height=IMAGE_HEIGHT,
        force_grayscale=FORCE_GRAY_SCALE)

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
