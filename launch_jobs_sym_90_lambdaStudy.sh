#!/bin/bash

# Results CIFAR-100 90% symetric noise (LAMBDA STUDY, Tau=0.03)
# machine: 210, GPU0
#CUDA_VISIBLE_DEVICES=0 python3 main_cifar.py --p_threshold 0.03 --r 0.9 --noise_mode sym --lambda_u 200  --dataset cifar100 --data_path ./../Datasets/CIFAR100  --experiment-name CF100_sym_0.9_weights_run1 --method selfsup --net resnet18
#CUDA_VISIBLE_DEVICES=0 python3 main_cifar.py --p_threshold 0.03 --r 0.9 --noise_mode sym --lambda_u 300  --dataset cifar100 --data_path ./../Datasets/CIFAR100  --experiment-name CF100_sym_0.9_weights_run1 --method selfsup --net resnet18
#CUDA_VISIBLE_DEVICES=0 python3 main_cifar.py --p_threshold 0.03 --r 0.9 --noise_mode sym --lambda_u 250  --dataset cifar100 --data_path ./../Datasets/CIFAR100  --experiment-name CF100_sym_0.9_weights_run1 --method selfsup --net resnet18
#CUDA_VISIBLE_DEVICES=0 python3 main_cifar.py --p_threshold 0.03 --r 0.9 --noise_mode sym --lambda_u 100  --dataset cifar100 --data_path ./../Datasets/CIFAR100  --experiment-name CF100_sym_0.9_weights_run1 --method selfsup --net resnet18

# machine: 123, GPU0
#CUDA_VISIBLE_DEVICES=0 python3 main_cifar.py --p_threshold 0.03 --r 0.9 --noise_mode sym --lambda_u 500  --dataset cifar100 --data_path ./../Datasets/CIFAR100  --experiment-name CF100_sym_0.9_weights_run1 --method selfsup --net resnet18
#CUDA_VISIBLE_DEVICES=0 python3 main_cifar.py --p_threshold 0.03 --r 0.9 --noise_mode sym --lambda_u 350  --dataset cifar100 --data_path ./../Datasets/CIFAR100  --experiment-name CF100_sym_0.9_weights_run1 --method selfsup --net resnet18
#CUDA_VISIBLE_DEVICES=0 python3 main_cifar.py --p_threshold 0.03 --r 0.9 --noise_mode sym --lambda_u 450  --dataset cifar100 --data_path ./../Datasets/CIFAR100  --experiment-name CF100_sym_0.9_weights_run1 --method selfsup --net resnet18
#CUDA_VISIBLE_DEVICES=0 python3 main_cifar.py --p_threshold 0.03 --r 0.9 --noise_mode sym --lambda_u 150  --dataset cifar100 --data_path ./../Datasets/CIFAR100  --experiment-name CF100_sym_0.9_weights_run1 --method selfsup --net resnet18

# machine: 54, GPU0
#CUDA_VISIBLE_DEVICES=0 python3 main_cifar.py --p_threshold 0.03 --r 0.9 --noise_mode sym --lambda_u 600  --dataset cifar100 --data_path ./../Datasets/CIFAR100  --experiment-name CF100_sym_0.9_weights_run1 --method selfsup --net resnet18
#CUDA_VISIBLE_DEVICES=0 python3 main_cifar.py --p_threshold 0.03 --r 0.9 --noise_mode sym --lambda_u 550  --dataset cifar100 --data_path ./../Datasets/CIFAR100  --experiment-name CF100_sym_0.9_weights_run1 --method selfsup --net resnet18
#CUDA_VISIBLE_DEVICES=0 python3 main_cifar.py --p_threshold 0.03 --r 0.9 --noise_mode sym --lambda_u 0  --dataset cifar100 --data_path ./../Datasets/CIFAR100  --experiment-name CF100_sym_0.9_weights_run1 --method selfsup --net resnet18
#CUDA_VISIBLE_DEVICES=0 python3 main_cifar.py --p_threshold 0.03 --r 0.9 --noise_mode sym --lambda_u 25  --dataset cifar100 --data_path ./../Datasets/CIFAR100  --experiment-name CF100_sym_0.9_weights_run1 --method selfsup --net resnet18

# machine: 63, GPU0
CUDA_VISIBLE_DEVICES=0 python3 main_cifar.py --p_threshold 0.03 --r 0.9 --noise_mode sym --lambda_u 700  --dataset cifar100 --data_path ./../Datasets/CIFAR100  --experiment-name CF100_sym_0.9_weights_run1 --method selfsup --net resnet18
CUDA_VISIBLE_DEVICES=0 python3 main_cifar.py --p_threshold 0.03 --r 0.9 --noise_mode sym --lambda_u 650  --dataset cifar100 --data_path ./../Datasets/CIFAR100  --experiment-name CF100_sym_0.9_weights_run1 --method selfsup --net resnet18
CUDA_VISIBLE_DEVICES=0 python3 main_cifar.py --p_threshold 0.03 --r 0.9 --noise_mode sym --lambda_u 50  --dataset cifar100 --data_path ./../Datasets/CIFAR100  --experiment-name CF100_sym_0.9_weights_run1 --method selfsup --net resnet18
CUDA_VISIBLE_DEVICES=0 python3 main_cifar.py --p_threshold 0.03 --r 0.9 --noise_mode sym --lambda_u 75  --dataset cifar100 --data_path ./../Datasets/CIFAR100  --experiment-name CF100_sym_0.9_weights_run1 --method selfsup --net resnet18

# Results CIFAR-100 90% symetric noise (LAMBDA STUDY)
# machine: 210, GPU0
#CUDA_VISIBLE_DEVICES=0 python3 main_cifar.py --r 0.9 --noise_mode sym --lambda_u 200  --dataset cifar100 --data_path ./../Datasets/CIFAR100  --experiment-name CF100_sym_0.9_weights_run1 --method selfsup --net resnet18
#CUDA_VISIBLE_DEVICES=0 python3 main_cifar.py --r 0.9 --noise_mode sym --lambda_u 300  --dataset cifar100 --data_path ./../Datasets/CIFAR100  --experiment-name CF100_sym_0.9_weights_run1 --method selfsup --net resnet18
#CUDA_VISIBLE_DEVICES=0 python3 main_cifar.py --r 0.9 --noise_mode sym --lambda_u 250  --dataset cifar100 --data_path ./../Datasets/CIFAR100  --experiment-name CF100_sym_0.9_weights_run1 --method selfsup --net resnet18

# machine: 123, GPU0
#CUDA_VISIBLE_DEVICES=0 python3 main_cifar.py --r 0.9 --noise_mode sym --lambda_u 500  --dataset cifar100 --data_path ./../Datasets/CIFAR100  --experiment-name CF100_sym_0.9_weights_run1 --method selfsup --net resnet18
#CUDA_VISIBLE_DEVICES=0 python3 main_cifar.py --r 0.9 --noise_mode sym --lambda_u 350  --dataset cifar100 --data_path ./../Datasets/CIFAR100  --experiment-name CF100_sym_0.9_weights_run1 --method selfsup --net resnet18
#CUDA_VISIBLE_DEVICES=0 python3 main_cifar.py --r 0.9 --noise_mode sym --lambda_u 450  --dataset cifar100 --data_path ./../Datasets/CIFAR100  --experiment-name CF100_sym_0.9_weights_run1 --method selfsup --net resnet18

# machine: 54, GPU0
#CUDA_VISIBLE_DEVICES=0 python3 main_cifar.py --r 0.9 --noise_mode sym --lambda_u 600  --dataset cifar100 --data_path ./../Datasets/CIFAR100  --experiment-name CF100_sym_0.9_weights_run1 --method selfsup --net resnet18
#CUDA_VISIBLE_DEVICES=0 python3 main_cifar.py --r 0.9 --noise_mode sym --lambda_u 550  --dataset cifar100 --data_path ./../Datasets/CIFAR100  --experiment-name CF100_sym_0.9_weights_run1 --method selfsup --net resnet18

# machine: 63, GPU0
#CUDA_VISIBLE_DEVICES=0 python3 main_cifar.py --r 0.9 --noise_mode sym --lambda_u 700  --dataset cifar100 --data_path ./../Datasets/CIFAR100  --experiment-name CF100_sym_0.9_weights_run1 --method selfsup --net resnet18
#CUDA_VISIBLE_DEVICES=0 python3 main_cifar.py --r 0.9 --noise_mode sym --lambda_u 650  --dataset cifar100 --data_path ./../Datasets/CIFAR100  --experiment-name CF100_sym_0.9_weights_run1 --method selfsup --net resnet18


# Results CIFAR-100 90% symetric noise (using our optimal lambda = 450)
#
#  ATTENTION!!! ADAPT CUDA VISIBLE DEVICE #
#
#CUDA_VISIBLE_DEVICES=0 python3 main_cifar.py --r 0.9 --noise_mode sym --lambda_u 400  --dataset cifar100 --data_path ./../Datasets/CIFAR100  --experiment-name CF100_sym_0.9_weights_run1 --method selfsup --net resnet18
#CUDA_VISIBLE_DEVICES=0 python3 main_cifar.py --r 0.9 --noise_mode sym --lambda_u 400  --dataset cifar100 --data_path ./../Datasets/CIFAR100  --experiment-name CF100_sym_0.9_weights_run2 --method selfsup --net resnet18
#CUDA_VISIBLE_DEVICES=0 python3 main_cifar.py --r 0.9 --noise_mode sym --lambda_u 400  --dataset cifar100 --data_path ./../Datasets/CIFAR100  --experiment-name CF100_sym_0.9_weights_run3 --method selfsup --net resnet18
#CUDA_VISIBLE_DEVICES=0 python3 main_cifar.py --r 0.9 --noise_mode sym --lambda_u 400  --dataset cifar100 --data_path ./../Datasets/CIFAR100  --experiment-name CF100_sym_0.9_weights_run4 --method selfsup --net resnet18
#CUDA_VISIBLE_DEVICES=0 python3 main_cifar.py --r 0.9 --noise_mode sym --lambda_u 400  --dataset cifar100 --data_path ./../Datasets/CIFAR100  --experiment-name CF100_sym_0.9_weights_run5 --method selfsup --net resnet18
