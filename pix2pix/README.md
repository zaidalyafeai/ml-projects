# Implementation of pix2pix 

![alt text](https://github.com/zaidalyafeai/zaidalyafeai.github.io/blob/master/pix2pix/pix2pix.gif)

## Demo 

Check here https://zaidalyafeai.github.io/pix2pix/celeb.html

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
2. use the `convert_keras.py` script to convert the model to keras 
`python convert_keras.py --dir input_dir --out output_dir`
3. Install tensorflowjs package using 
`pip install tensorflowjs`
4. Convert the model  

`tensorflowjs_converter --input_format keras keras.h5 output_directory`

## Train on your dataset 

Use these scripts 

https://github.com/zaidalyafeai/pix2pix/tree/master/scripts/edges

The process first uses a caffe model to create mat files. Then you can use matlab to generate edges. If you faced some difficulties with that you can use `cv2.canny` to extract the edge map of the input iamges. 

## Processed Dataset 

Check `cats.zip` which contains 1000 images of cats. It was obtained from http://www.robots.ox.ac.uk/~vgg/data/pets/ by 
first using the segmentation to extract the cats and replace the background with white. Then the previous step was used 
to generate the edges. 

Also, `pokemon.zip` contains 800 images of Pokemons that were optained from https://www.kaggle.com/kvpratama/pokemon-images-dataset. The edges were extracted using canny edge extractor. 


