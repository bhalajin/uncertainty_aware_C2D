#!/bin/bash

# Results CIFAR-100 50% symetric noise
#
#  ATTENTION!!! ADAPT CUDA VISIBLE DEVICE #
#
CUDA_VISIBLE_DEVICES=1 python3 main_cifar.py --r 0.5 --noise_mode sym --lambda_u 150  --dataset cifar100 --data_path ./../Datasets/CIFAR100  --experiment-name CF100_sym_0.5_weights_run1 --method selfsup --net resnet18
CUDA_VISIBLE_DEVICES=1 python3 main_cifar.py --r 0.5 --noise_mode sym --lambda_u 150  --dataset cifar100 --data_path ./../Datasets/CIFAR100  --experiment-name CF100_sym_0.5_weights_run2 --method selfsup --net resnet18
CUDA_VISIBLE_DEVICES=1 python3 main_cifar.py --r 0.5 --noise_mode sym --lambda_u 150  --dataset cifar100 --data_path ./../Datasets/CIFAR100  --experiment-name CF100_sym_0.5_weights_run3 --method selfsup --net resnet18
CUDA_VISIBLE_DEVICES=1 python3 main_cifar.py --r 0.5 --noise_mode sym --lambda_u 150  --dataset cifar100 --data_path ./../Datasets/CIFAR100  --experiment-name CF100_sym_0.5_weights_run4 --method selfsup --net resnet18
CUDA_VISIBLE_DEVICES=1 python3 main_cifar.py --r 0.5 --noise_mode sym --lambda_u 150  --dataset cifar100 --data_path ./../Datasets/CIFAR100  --experiment-name CF100_sym_0.5_weights_run5 --method selfsup --net resnet18
