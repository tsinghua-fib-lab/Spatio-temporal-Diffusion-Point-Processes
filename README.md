# Spatio-temporal Diffusion Point Processes

This repo contains the codes and data for our submitted KDD'23 research track paper under review:

The code is tested under a Linux desktop with torch 1.7 and Python 3.8.10.

## Installation

### Environment
- Tested OS: Linux
- Python >= 3.7
- PyTorch == 1.7.1
- Tensorboard

### Dependencies
1. Install PyTorch 1.7.1 with the correct CUDA version.

## Model Training

Use the following command to train a DSTPP model on `Earthquake` dataset: 

``
python app.py --dataset Earthquake --mode train
``



## Note

The implemention is based on *[DDPM](https://github.com/lucidrains/denoising-diffusion-pytorch)*.
