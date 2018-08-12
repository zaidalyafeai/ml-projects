import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.models import Model
from keras.layers import Conv2D, BatchNormalization, Conv2DTranspose
from keras.layers import  Dropout, Concatenate, Activation, Input
from keras.layers.advanced_activations import LeakyReLU
import matplotlib.pyplot as plt
import cv2

with tf.Session() as sess:
    saver= tf.train.import_meta_graph('export.meta')
    saver.restore(sess, 'export')
    idx = 0
    variables = [v for v in tf.all_variables()]
    print(len(variables))
    idx = 0
    for v in variables:
        out = sess.run(v)
        np.save(str(idx)+'.npy', out)
        idx += 1