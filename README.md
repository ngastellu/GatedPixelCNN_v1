# GatedPixelCNN_v1

GatedPixelCNN impelmentation used to generate outputs in our paper "Simulating Amorphous Materials on Multiple Length-Scales using Deep Learning", based on the architecture used in https://papers.nips.cc/paper/6527-conditional-image-generation-with-pixelcnn-decoders.pdf .


## Instructions

Runs on Python 3.7 and PyTorch 1.3.

Before running, create output directories 'ckpts', 'samples', 'outputs', 'logfiles', 'raw_outputs' in the main directory.

Required packages: torch, torchvision, numpy, torchsummary, tqdm, pickle, argparse. 

Evaluates without error on cuda-enabled GPU.

Add the dataset you wish to train on into the build_dataset function in utils.py.

Model parameters can be set either in main.py for running one at a time, or for array submission a parameter file may be generated using create_batch_parameters.py (example slurm submission script enclosed).

The code will automatically set batch sizes and converge to the required tightness of fit. Note that to reduce overfitting, test_margin in auto_convergence in utils.py should be set <1.

The code has built-in analysis scripts to compare quality of generated samples as compared to training data. The agreement metrics for training_data=1 are the following: 

1. tot - normed sum of all analysis metrics, max=1
2. den - density of occupied pixels
3. en - a local 'energy'. The distribution of number of nearest neighbours
4. corr - 2D pair-correlation
5. fourier - 2D fourier analysis (Note the Fourier analysis will return 0 agreement if input and outputs are different sizes)

For training_data=2, the code instead does bond analysis (length, order, angles), with parameters set in accuracy_metrics.py
