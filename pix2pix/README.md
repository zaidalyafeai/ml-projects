# Implementation of pix2pix 

![alt text](https://raw.githubusercontent.com/zaidalyafeai/zaidalyafeai.github.io/master/images/pix2pix.PNG)


## Implementation 

Based on the Tensorflow implementation 

https://github.com/affinelayer/pix2pix-tensorflow

## Training 

Use `tf_pix2pix.ipynb` notebook for training. You can run it on colab using this link 

https://colab.research.google.com/github/zaidalyafeai/zaidalyafeai.github.io/blob/master/pix2pix/tf_pix2pix.ipynb

Training depends on the dataset. Here you can find many datasets 

https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/

Make sure to choose the correct direction `AtoB` or `BtoA` depending on the dataset. 

## Convert the model to TensorFlow.js 

1. First export the model by changing the `mode` to `export`. This will create export files. 
2. Use this the `extract_weights.py` script to save the weights files as `.npy` files. 
3. Create a keras model using `save_keras.py` to load the weights and generate a `keras.h5` file 
4. Install tensorflowjs package using 
`pip install tensorflowjs`
5. Convert the model  

`tensorflowjs_converter --input_format keras keras.h5 output_directory`

