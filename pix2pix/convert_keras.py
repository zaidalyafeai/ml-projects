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
import argparse
import tempfile
import shutil


#create temp directory 
dirpath = tempfile.mkdtemp()

parser = argparse.ArgumentParser(description='Convert to keras')
parser.add_argument('--dir', dest = "dir", 
                   help='foulder that contains the checkpoints')
parser.add_argument('--out',
                   help='output directory of the keras model')                      

args = parser.parse_args()


with tf.Session() as sess:
    saver= tf.train.import_meta_graph(args.dir+'/export.meta')
    saver.restore(sess, args.dir+'/export')
    idx = 0
    variables = [v for v in tf.all_variables()]
    idx = 0
    for v in variables:
        out = sess.run(v)
        np.save(dirpath+'/'+str(idx)+'.npy', out)
        idx += 1

print('save weight files')

tf.reset_default_graph()

def gen_conv(x, out_channels):
    y= Conv2D(filters = out_channels, kernel_size = 4, 
                                   strides = (2,2), padding = 'same', 
                                   kernel_initializer= 'zeros', input_shape = [256, 256, 3])(x)
    return y
def batchnorm(x):
    return  BatchNormalization(axis=3)(x, training = 1)

def lrelu(x, a):
    return LeakyReLU(alpha=a)(x)

def gen_deconv(x, out_channels):
    y = Conv2DTranspose(out_channels, kernel_size=4, strides=(2, 2), padding="same")(x)
    return y


def generator():
    ngf = 32

    input = Input(shape = [256, 256, 3])
    layers = []
    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
    x = gen_conv(input, ngf)
    layers.append(x)
    
    layer_specs = [
        ngf * 2, # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
        ngf * 4, # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
        ngf * 8, # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
        ngf * 8, # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
        ngf * 8, # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
        ngf * 8, # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
        ngf * 8, # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
    ]
    
    for out_channels in layer_specs:
        x = lrelu(layers[-1], 0.2)
        # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
        x = gen_conv(x, out_channels)
        x = batchnorm(x)
        layers.append(x)
        
    layer_specs = [
        (ngf * 8, 0.5),   # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
        (ngf * 8, 0.5),   # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
        (ngf * 8, 0.5),   # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
        (ngf * 8, 0.0),   # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
        (ngf * 4, 0.0),   # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
        (ngf * 2, 0.0),   # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
        (ngf, 0.0),       # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
    ]

    num_encoder_layers = len(layers)
    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1

        if decoder_layer == 0:
            in_data = layers[-1]
        else:
            in_data = Concatenate(axis=3)([layers[-1], layers[skip_layer]])
        x = Activation('relu')(in_data)
        # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
        x = gen_deconv(x, out_channels)
        x = batchnorm(x)

        if dropout > 0.0:
            x = Dropout(dropout)(x)
        layers.append(x)
            
    x = Concatenate(axis=3)([layers[-1], layers[0]])
    x = Activation('relu')(x)
    x = gen_deconv(x, 3)
    output = Activation('tanh')(x)
    layers.append(output)
    
    return Model(inputs = input, outputs = output)

model = generator()
print('model generated')

weights = [] 
for i in range(0, 88):
    name = dirpath+'/'+str(i)+'.npy'
    weights.append(np.load(name))

idx = 0 
for layer in model.layers[1:]:
    if 'conv2d' in layer.name:
        W = weights[idx]
        b = weights[idx+1]
        layer.set_weights([W, b])
        idx += 2
    elif 'batch' in layer.name:
        g = weights[idx]
        b = weights[idx+1]
        m = weights[idx+2]
        v = weights[idx+3]
        layer.set_weights([g, b, m, v])
        idx += 4 
    else:
        continue

print('weights loaded')

model.save(args.out+'/keras.h5')
print('model saved to',  args.out)

shutil.rmtree(dirpath)
print('temp files removed')