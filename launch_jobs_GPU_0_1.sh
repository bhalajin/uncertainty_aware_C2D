#!/bin/bash

# 3 extra runs to finish 90% with lambda = 400
CUDA_VISIBLE_DEVICES=0 python3 main_cifar.py --r 0.9 --noise_mode sym --lambda_u 400  --dataset cifar10 --data_path ./../Datasets/CIFAR10  --experiment-name sym_0.9_weights_run3 --method selfsup --net resnet18
CUDA_VISIBLE_DEVICES=0 python3 main_cifar.py --r 0.9 --noise_mode sym --lambda_u 400  --dataset cifar10 --data_path ./../Datasets/CIFAR10  --experiment-name sym_0.9_weights_run4 --method selfsup --net resnet18
CUDA_VISIBLE_DEVICES=0 python3 main_cifar.py --r 0.9 --noise_mode sym --lambda_u 400  --dataset cifar10 --data_path ./../Datasets/CIFAR10  --experiment-name sym_0.9_weights_run5 --method selfsup --net resnet18


# LAMBDA_U EXPERIMENTS (noise = 0.9, sym, CIFAR10)
#CUDA_VISIBLE_DEVICES=0 python3 main_cifar.py --r 0.9 --noise_mode sym --lambda_u 400  --dataset cifar10 --data_path ./../Datasets/CIFAR10  --experiment-name sym_0.9_weights_run1 --method selfsup --net resnet18
#CUDA_VISIBLE_DEVICES=0 python3 main_cifar.py --r 0.9 --noise_mode sym --lambda_u 400  --dataset cifar10 --data_path ./../Datasets/CIFAR10  --experiment-name sym_0.9_weights_run2 --method selfsup --net resnet18
#CUDA_VISIBLE_DEVICES=0 python3 main_cifar.py --r 0.9 --noise_mode sym --lambda_u 450  --dataset cifar10 --data_path ./../Datasets/CIFAR10  --experiment-name sym_0.9_weights_run1 --method selfsup --net resnet18
#CUDA_VISIBLE_DEVICES=0 python3 main_cifar.py --r 0.9 --noise_mode sym --lambda_u 450  --dataset cifar10 --data_path ./../Datasets/CIFAR10  --experiment-name sym_0.9_weights_run2 --method selfsup --net resnet18


# COMMANDS EXECUTED FOR 0.8 SYM
#CUDA_VISIBLE_DEVICES=0 python3 main_cifar.py --r 0.8 --noise_mode sym --lambda_u 25 --dataset cifar10 --data_path ./../Datasets/CIFAR10  --experiment-name sym_0.8_weights_run1 --method selfsup --net resnet18
#CUDA_VISIBLE_DEVICES=0 python3 main_cifar.py --r 0.8 --noise_mode sym --lambda_u 25 --dataset cifar10 --data_path ./../Datasets/CIFAR10  --experiment-name sym_0.8_weights_run2 --method selfsup --net resnet18
#CUDA_VISIBLE_DEVICES=0 python3 main_cifar.py --r 0.8 --noise_mode sym --lambda_u 25 --dataset cifar10 --data_path ./../Datasets/CIFAR10  --experiment-name sym_0.8_weights_run3 --method selfsup --net resnet18
#CUDA_VISIBLE_DEVICES=0 python3 main_cifar.py --r 0.8 --noise_mode sym --lambda_u 25 --dataset cifar10 --data_path ./../Datasets/CIFAR10  --experiment-name sym_0.8_weights_run4 --method selfsup --net resnet18
#CUDA_VISIBLE_DEVICES=0 python3 main_cifar.py --r 0.8 --noise_mode sym --lambda_u 25 --dataset cifar10 --data_path ./../Datasets/CIFAR10  --experiment-name sym_0.8_weights_run5 --method selfsup --net resnet18



# COMMANDS EXECUTED FOR 0.4 ASYM
#python3 main_cifar.py --gpuid 0 --r 0.4 --noise_mode asym --lambda_u 0  --dataset cifar10 --data_path ./../Datasets/CIFAR10  --experiment-name asym_0.4_weights_run1 --method selfsup --net resnet18
#python3 main_cifar.py --gpuid 0 --r 0.4 --noise_mode asym --lambda_u 0  --dataset cifar10 --data_path ./../Datasets/CIFAR10  --experiment-name asym_0.4_weights_run2 --method selfsup --net resnet18
#python3 main_cifar.py --gpuid 0 --r 0.4 --noise_mode asym --lambda_u 0  --dataset cifar10 --data_path ./../Datasets/CIFAR10  --experiment-name asym_0.4_weights_run3 --method selfsup --net resnet18
#python3 main_cifar.py --gpuid 0 --r 0.4 --noise_mode asym --lambda_u 0  --dataset cifar10 --data_path ./../Datasets/CIFAR10  --experiment-name asym_0.4_weights_run4 --method selfsup --net resnet18
#python3 main_cifar.py --gpuid 0 --r 0.4 --noise_mode asym --lambda_u 0  --dataset cifar10 --data_path ./../Datasets/CIFAR10  --experiment-name asym_0.4_weights_run5 --method selfsup --net resnet18
