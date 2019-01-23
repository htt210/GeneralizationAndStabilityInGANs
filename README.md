# Improving Generalization And Stability of GANs
Code for paper "Improving Generalization And Stability of GANs". If you use this code please consider citing our paper.
```text
@inproceedings{
thanh-tung2018improving,
title={Improving Generalization and Stability of Generative Adversarial Networks},
author={Hoang Thanh-Tung and Truyen Tran and Svetha Venkatesh},
booktitle={International Conference on Learning Representations},
year={2019},
url={https://openreview.net/forum?id=ByxPYjC5KQ},
}
```
## Requirements
- Pytorch 0.4.1 
## Usage
```
python GradientPenaltiesGAN.py --help 

usage: GradientPenaltiesGAN.py [-h] [--nhidden NHIDDEN] [--gnlayers GNLAYERS]
                               [--dnlayers DNLAYERS] [--gradweight GRADWEIGHT]
                               [--gweight GWEIGHT] [--dweight DWEIGHT]
                               [--niters NITERS] [--device DEVICE]
                               [--batch_size BATCH_SIZE] [--center CENTER]
                               [--LAMBDA LAMBDA] [--alpha ALPHA] [--lrg LRG]
                               [--lrd LRD] [--dataset DATASET] [--scale SCALE]
                               [--loss LOSS] [--optim OPTIM]
                               [--ncritic NCRITIC]

optional arguments:
  -h, --help            show this help message and exit
  --nhidden NHIDDEN     number of hidden neurons
  --gnlayers GNLAYERS   number of hidden layers in generator
  --dnlayers DNLAYERS   number of hidden layers in discriminator/critic
  --gradweight GRADWEIGHT
                        weight of the new grad in the moving average
  --gweight GWEIGHT     weight of the new G
  --dweight DWEIGHT     weight of the new D
  --niters NITERS       number of iterations
  --device DEVICE       id of the gpu. -1 for cpu
  --batch_size BATCH_SIZE
                        batch size
  --center CENTER       gradpen center
  --LAMBDA LAMBDA       gradpen weight
  --alpha ALPHA         interpolation weight between reals and fakes
  --lrg LRG             lr for G
  --lrd LRD             lr for D
  --dataset DATASET     dataset to use: 8Gaussians | 25Gaussians | swissroll
  --scale SCALE         data scaling
  --loss LOSS           gan | wgan
  --optim OPTIM         optimizer to use
  --ncritic NCRITIC     critic iters / generator iter
```

The code for ImageNet experiment is available at: https://github.com/LMescheder/GAN_stability. 

