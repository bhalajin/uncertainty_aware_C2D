#!/bin/bash

# Results CIFAR-100 40% asymetric noise
#
#  ATTENTION!!! ADAPT CUDA VISIBLE DEVICE #
#
CUDA_VISIBLE_DEVICES=1 python3 main_cifar.py --r 0.4 --noise_mode asym --lambda_u 0  --dataset cifar100 --data_path ./../Datasets/CIFAR100  --experiment-name CF100_asym_0.4_weights_run1 --method selfsup --net resnet18
CUDA_VISIBLE_DEVICES=1 python3 main_cifar.py --r 0.4 --noise_mode asym --lambda_u 0  --dataset cifar100 --data_path ./../Datasets/CIFAR100  --experiment-name CF100_asym_0.4_weights_run2 --method selfsup --net resnet18
CUDA_VISIBLE_DEVICES=1 python3 main_cifar.py --r 0.4 --noise_mode asym --lambda_u 0  --dataset cifar100 --data_path ./../Datasets/CIFAR100  --experiment-name CF100_asym_0.4_weights_run3 --method selfsup --net resnet18
CUDA_VISIBLE_DEVICES=1 python3 main_cifar.py --r 0.4 --noise_mode asym --lambda_u 0  --dataset cifar100 --data_path ./../Datasets/CIFAR100  --experiment-name CF100_asym_0.4_weights_run4 --method selfsup --net resnet18
CUDA_VISIBLE_DEVICES=1 python3 main_cifar.py --r 0.4 --noise_mode asym --lambda_u 0  --dataset cifar100 --data_path ./../Datasets/CIFAR100  --experiment-name CF100_asym_0.4_weights_run5 --method selfsup --net resnet18