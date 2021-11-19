#!/bin/bash

# Results CIFAR-100 80% symetric noise (using our optimal lambda = 450)
#
#  ATTENTION!!! ADAPT CUDA VISIBLE DEVICE #
#
CUDA_VISIBLE_DEVICES=0 python3 main_cifar.py --r 0.8 --noise_mode sym --lambda_u 450  --dataset cifar100 --data_path ./../Datasets/CIFAR100  --experiment-name CF100_sym_0.8_weights_run1 --method selfsup --net resnet18
CUDA_VISIBLE_DEVICES=0 python3 main_cifar.py --r 0.8 --noise_mode sym --lambda_u 450  --dataset cifar100 --data_path ./../Datasets/CIFAR100  --experiment-name CF100_sym_0.8_weights_run2 --method selfsup --net resnet18
CUDA_VISIBLE_DEVICES=0 python3 main_cifar.py --r 0.8 --noise_mode sym --lambda_u 450  --dataset cifar100 --data_path ./../Datasets/CIFAR100  --experiment-name CF100_sym_0.8_weights_run3 --method selfsup --net resnet18
CUDA_VISIBLE_DEVICES=0 python3 main_cifar.py --r 0.8 --noise_mode sym --lambda_u 450  --dataset cifar100 --data_path ./../Datasets/CIFAR100  --experiment-name CF100_sym_0.8_weights_run4 --method selfsup --net resnet18
CUDA_VISIBLE_DEVICES=0 python3 main_cifar.py --r 0.8 --noise_mode sym --lambda_u 450  --dataset cifar100 --data_path ./../Datasets/CIFAR100  --experiment-name CF100_sym_0.8_weights_run5 --method selfsup --net resnet18