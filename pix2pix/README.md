# Implementation of pix2pix 

![alt text](https://raw.githubusercontent.com/zaidalyafeai/zaidalyafeai.github.io/master/images/pix2pix.PNG)


## Implementation 

Based on the Tensorflow implementation 

https://github.com/affinelayer/pix2pix-tensorflow

## Training 

Use tf_pix2pix.ipynb file for training. You can run it on colab using this 

https://colab.research.google.com/github/zaidalyafeai/zaidalyafeai.github.io/pix2pix/tf_pix2pix.ipynb

## Convert the model to TensorFlow.js 

1. First export the model by changing the mode to export. This will create export files. 
2. Use this the `extract_weights.py` script to save the weights files as `.npy` files. 
3. Create a keras model using `save_keras.py` to load the weights and generate a keras.h5 file 
4. Convert to web format using 

`tensorflowjs_converter --input_format keras keras.h5 output_directory`

