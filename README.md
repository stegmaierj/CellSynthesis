# CellSynthesis [![DOI](https://zenodo.org/badge/384055323.svg)](https://zenodo.org/badge/latestdoi/384055323)

[3D Fluorescence Microscopy Data Synthesis for Segmentation and Benchmarking](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0260509)\
*D. Eschweiler, M. Rethwisch, M. Jarchow, S. Koppers, J. Stegmaier, PLoS ONE, 2021*


This repository contains code used for our 3D cell simulation and synthesis pipeline. We are actively working on improving synthetic data quality, computation performance and user-friendliness.


### Data Preparation
The data needs to be in a hdf5 format containing image data for the network input and positional + shape information as output.
The data is assumed to be in a structure similar to the following schematic.

`-|data_root`<br>
`----|experiment1`<br>
`--------|images`<br>
`--------|masks`<br>
`----|experiment2`<br>
`--------|images`<br>
`--------|masks`<br>

To prepare your own TIF data, proceed as explained in the following steps:
1. Convert the data using `utils.h5_converter.prepare_images` and `utils.h5_converter.prepare_masks` to prepare image and mask data, respectively.
2. Create a .csv filelist using `utils.csv_generator.create_csv`, while the input is assumed to be a list of tuples containing image-mask pairs -> <br>
`[('experiment1/images_converted/im_1.h5', 'experiment1/masks_converted/mask_1.h5'),`<br>
  `...,`<br>
  `('experiment2/images_converted/im_n.h5', 'experiment2/masks_converted/mask_n.h5')]`<br>
  
  
### Training and Application
This pipeline was tested on Ubuntu 18 and 20 using python 3.7.
A minimally required list of python packages and versions can be found in `requirements.txt`.
For training and application use the provided `train_script.py` and `apply_script.py` and make sure to adjust the data paths in the `models.GAN` accordingly.
