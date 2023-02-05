# Spatio-temporal Diffusion Point Processes

This repo contains the codes and data for our submitted KDD'23 research track paper under review:

The code is tested under a Linux desktop with torch 1.7 and Python 3.7.10.

## Installation

### Environment
- Tested OS: Linux
- Python >= 3.7
- PyTorch == 1.7.1
- Tensorboard

### Dependencies
1. Install PyTorch 1.7.1 with the correct CUDA version.

## Model Training

Use the following command to train DSTPP on `Earthquake` dataset: 

``
python app.py --dataset Earthquake --mode train --timesteps 200 --samplingsteps 200 --batch_size 64 --total_epochs 2000
``

To train DSTPP on other datasets:

``
python app.py --dataset COVID19 --mode train --timesteps 200 --samplingsteps 200 --batch_size 64 --total_epochs 2000
``

``
python app.py --dataset Pinwheel --mode train --timesteps 200 --samplingsteps 200 --batch_size 256 --total_epochs 2000 
``

``
python app.py --dataset HawkesGMM --mode train --timesteps 200 --samplingsteps 200 --batch_size 256 --total_epochs 2000
``

``
python app.py --dataset Mobility --mode train --timesteps 200 --samplingsteps 200 --batch_size 128 --total_epochs 2000 
``

``
python app.py --dataset Citibike --mode train --timesteps 200 --samplingsteps 200 --batch_size 128 --total_epochs 2000 
``

``
python app.py --dataset Independent --mode train --timesteps 200 --samplingsteps 200 --batch_size 128 --total_epochs 2000 
``


## Note

The implemention is based on *[DDPM](https://github.com/lucidrains/denoising-diffusion-pytorch)*.
