#!/bin/bash

./launch_jobs_asym_40.sh

# LAMBDA_U EXPERIMENTS (noise = 0.9, sym, CIFAR10)
#CUDA_VISIBLE_DEVICES=1 python3 main_cifar.py --r 0.9 --noise_mode sym --lambda_u 600  --dataset cifar10 --data_path ./../Datasets/CIFAR10  --experiment-name sym_0.9_weights_run1 --method selfsup --net resnet18
#CUDA_VISIBLE_DEVICES=1 python3 main_cifar.py --r 0.9 --noise_mode sym --lambda_u 600  --dataset cifar10 --data_path ./../Datasets/CIFAR10  --experiment-name sym_0.9_weights_run2 --method selfsup --net resnet18
#CUDA_VISIBLE_DEVICES=1 python3 main_cifar.py --r 0.9 --noise_mode sym --lambda_u 650  --dataset cifar10 --data_path ./../Datasets/CIFAR10  --experiment-name sym_0.9_weights_run1 --method selfsup --net resnet18
#CUDA_VISIBLE_DEVICES=1 python3 main_cifar.py --r 0.9 --noise_mode sym --lambda_u 650  --dataset cifar10 --data_path ./../Datasets/CIFAR10  --experiment-name sym_0.9_weights_run2 --method selfsup --net resnet18

# EXPERIMENTS 0.9 SYM
#CUDA_VISIBLE_DEVICES=1 python3 main_cifar.py --r 0.9 --noise_mode sym --lambda_u 50  --dataset cifar10 --data_path ./../Datasets/CIFAR10  --experiment-name sym_0.9_weights_run1 --method selfsup --net resnet18
#CUDA_VISIBLE_DEVICES=1 python3 main_cifar.py --r 0.9 --noise_mode sym --lambda_u 50  --dataset cifar10 --data_path ./../Datasets/CIFAR10  --experiment-name sym_0.9_weights_run2 --method selfsup --net resnet18
#CUDA_VISIBLE_DEVICES=1 python3 main_cifar.py --r 0.9 --noise_mode sym --lambda_u 50  --dataset cifar10 --data_path ./../Datasets/CIFAR10  --experiment-name sym_0.9_weights_run3 --method selfsup --net resnet18
#CUDA_VISIBLE_DEVICES=1 python3 main_cifar.py --r 0.9 --noise_mode sym --lambda_u 50  --dataset cifar10 --data_path ./../Datasets/CIFAR10  --experiment-name sym_0.9_weights_run4 --method selfsup --net resnet18
#CUDA_VISIBLE_DEVICES=1 python3 main_cifar.py --r 0.9 --noise_mode sym --lambda_u 50  --dataset cifar10 --data_path ./../Datasets/CIFAR10  --experiment-name sym_0.9_weights_run5 --method selfsup --net resnet18

# EXPERIMENTS 0.2% SYM 
#python3 main_cifar.py --gpuid 1 --r 0.2 --noise_mode sym --lambda_u 0  --dataset cifar10 --data_path ./../Datasets/CIFAR10  --experiment-name sym_0.2_weights_run1 --method selfsup --net resnet18
#python3 main_cifar.py --gpuid 1 --r 0.2 --noise_mode sym --lambda_u 0  --dataset cifar10 --data_path ./../Datasets/CIFAR10  --experiment-name sym_0.2_weights_run2 --method selfsup --net resnet18
#python3 main_cifar.py --gpuid 1 --r 0.2 --noise_mode sym --lambda_u 0  --dataset cifar10 --data_path ./../Datasets/CIFAR10  --experiment-name sym_0.2_weights_run3 --method selfsup --net resnet18
#python3 main_cifar.py --gpuid 1 --r 0.2 --noise_mode sym --lambda_u 0  --dataset cifar10 --data_path ./../Datasets/CIFAR10  --experiment-name sym_0.2_weights_run4 --method selfsup --net resnet18
#python3 main_cifar.py --gpuid 1 --r 0.2 --noise_mode sym --lambda_u 0  --dataset cifar10 --data_path ./../Datasets/CIFAR10  --experiment-name sym_0.2_weights_run5 --method selfsup --net resnet18
