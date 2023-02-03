# Spatio-temporal Diffusion Point Processes

This is our Pytorch implementation for DSTPP.

The code is tested under a Linux desktop with torch 1.7 and Python 3.8.10.

## Model Training

Use the following command to train a STDPP model on `Earthquake` dataset: 

``
python app.py --dataset Earthquake --mode train
``



## Note

The implemention is based on *[DDPM](https://github.com/lucidrains/denoising-diffusion-pytorch)*.
