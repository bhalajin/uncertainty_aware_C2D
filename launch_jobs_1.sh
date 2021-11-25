#!/bin/bash

# Data Generation Experiments WITH WEIGHTS
python3 main_cifar.py --r 0.5 --noise_mode sym --lambda_u 25  --dataset cifar10 --data_path ./../Datasets/CIFAR10  --experiment-name sym_0.5_weights_DATA_run1 --method selfsup --net resnet18

# LAMBDA_U EXPERIMENTS (noise = 0.8)
#python3 main_cifar.py --r 0.8 --noise_mode sym --lambda_u 300  --dataset cifar10 --data_path ./../Datasets/CIFAR10  --experiment-name sym_0.8_weights_run1 --method selfsup --net resnet18
#python3 main_cifar.py --r 0.8 --noise_mode sym --lambda_u 300  --dataset cifar10 --data_path ./../Datasets/CIFAR10  --experiment-name sym_0.8_weights_run2 --method selfsup --net resnet18
#python3 main_cifar.py --r 0.8 --noise_mode sym --lambda_u 350  --dataset cifar10 --data_path ./../Datasets/CIFAR10  --experiment-name sym_0.8_weights_run1 --method selfsup --net resnet18
#python3 main_cifar.py --r 0.8 --noise_mode sym --lambda_u 350  --dataset cifar10 --data_path ./../Datasets/CIFAR10  --experiment-name sym_0.8_weights_run2 --method selfsup --net resnet18


# LAMBDA_U EXPERIMENTS (noise = 0.9)
#python3 main_cifar.py --r 0.9 --noise_mode sym --lambda_u 200  --dataset cifar10 --data_path ./../Datasets/CIFAR10  --experiment-name sym_0.9_weights__lambda_u_200_run1 --method selfsup --net resnet18
#python3 main_cifar.py --r 0.9 --noise_mode sym --lambda_u 200  --dataset cifar10 --data_path ./../Datasets/CIFAR10  --experiment-name sym_0.9_weights__lambda_u_200_run2 --method selfsup --net resnet18
#python3 main_cifar.py --r 0.9 --noise_mode sym --lambda_u 250  --dataset cifar10 --data_path ./../Datasets/CIFAR10  --experiment-name sym_0.9_weights__lambda_u_250_run1 --method selfsup --net resnet18
#python3 main_cifar.py --r 0.9 --noise_mode sym --lambda_u 250  --dataset cifar10 --data_path ./../Datasets/CIFAR10  --experiment-name sym_0.9_weights__lambda_u_250_run2 --method selfsup --net resnet18


# FIRST SET OF RESULTS
#python3 main_cifar.py --r 0.5 --noise_mode sym --lambda_u 25  --dataset cifar10 --data_path ./../Datasets/CIFAR10  --experiment-name sym_0.5_weights_run1 --method selfsup --net resnet18
#python3 main_cifar.py --r 0.5 --noise_mode sym --lambda_u 25  --dataset cifar10 --data_path ./../Datasets/CIFAR10  --experiment-name sym_0.5_weights_run2 --method selfsup --net resnet18
#python3 main_cifar.py --r 0.5 --noise_mode sym --lambda_u 25  --dataset cifar10 --data_path ./../Datasets/CIFAR10  --experiment-name sym_0.5_weights_run3 --method selfsup --net resnet18
#python3 main_cifar.py --r 0.5 --noise_mode sym --lambda_u 25  --dataset cifar10 --data_path ./../Datasets/CIFAR10  --experiment-name sym_0.5_weights_run4 --method selfsup --net resnet18
#python3 main_cifar.py --r 0.5 --noise_mode sym --lambda_u 25  --dataset cifar10 --data_path ./../Datasets/CIFAR10  --experiment-name sym_0.5_weights_run5 --method selfsup --net resnet18
