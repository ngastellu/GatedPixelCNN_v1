# GatedPixelCNN_v1

GatedPixelCNN impelmentation used to generate outputs in our paper "Simulating Amorphous Materials on Multiple Length-Scales using Deep Learning", based on the architecture used in https://papers.nips.cc/paper/6527-conditional-image-generation-with-pixelcnn-decoders.pdf .


## Instructions

Runs on Python 3.7 and PyTorch 1.3. 

Before running, create output directories 'ckpts', 'samples', 'outputs', 'logfiles', 'raw_outputs' in the main directory.

Add the dataset you wish to train on into the build_dataset function in utils.py.

Model parameters can be set either in main.py for running one at a time, or for array submission a parameter file may be generated using create_batch_parameters.py.

The code has built-in analysis scripts to compare quality of generated samples as compared to training data. 
