# CellSynthesis [![DOI](https://zenodo.org/badge/384055323.svg)](https://zenodo.org/badge/latestdoi/384055323)

3D Fluorescence Microscopy Data Synthesis for Segmentation and Benchmarking\
*Copyright (C) 2021 D. Eschweiler, M. Rethwisch, M. Jarchow, S. Koppers, J. Stegmaier*


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
For training and application use the provided scripts and make sure to adjust the data paths in the `models.GAN` accordingly.
