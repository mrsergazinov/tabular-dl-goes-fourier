#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a30:1
#SBATCH --ntasks=1
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=output_%A_%a.out
#SBATCH --array=0-11

# Activate the Conda environment
source ~/.bashrc
conda activate tabllm

commands=(
  "python optim_hyperparam.py --model_name TabTransformer --output_file results.txt --n_trials 50 --dataset_name adult --num_encoder FourierFeaturesCos --no_num_encoder_trainable --scaler SquareScalingFeatures"
  "python optim_hyperparam.py --model_name TabTransformer --output_file results.txt --n_trials 50 --dataset_name otto_group --num_encoder FourierFeaturesCos --no_num_encoder_trainable --scaler SquareScalingFeatures"
  "python optim_hyperparam.py --model_name TabTransformer --output_file results.txt --n_trials 50 --dataset_name california_housing --num_encoder FourierFeaturesCos --no_num_encoder_trainable --scaler SquareScalingFeatures"
  "python optim_hyperparam.py --model_name TabTransformer --output_file results.txt --n_trials 50 --dataset_name higgs --num_encoder FourierFeaturesCos --no_num_encoder_trainable --scaler SquareScalingFeatures"
  "python optim_hyperparam.py --model_name MLP --output_file results.txt --n_trials 50 --dataset_name adult --num_encoder FourierFeaturesCos --no_num_encoder_trainable --scaler SquareScalingFeatures"
  "python optim_hyperparam.py --model_name MLP --output_file results.txt --n_trials 50 --dataset_name otto_group --num_encoder FourierFeaturesCos --no_num_encoder_trainable --scaler SquareScalingFeatures"
  "python optim_hyperparam.py --model_name MLP --output_file results.txt --n_trials 50 --dataset_name california_housing --num_encoder FourierFeaturesCos --no_num_encoder_trainable --scaler SquareScalingFeatures"
  "python optim_hyperparam.py --model_name MLP --output_file results.txt --n_trials 50 --dataset_name higgs --num_encoder FourierFeaturesCos --no_num_encoder_trainable --scaler SquareScalingFeatures"
  "python optim_hyperparam.py --model_name ModernNCA --output_file results.txt --n_trials 50 --dataset_name adult --num_encoder FourierFeaturesCos --no_num_encoder_trainable --scaler SquareScalingFeatures"
  "python optim_hyperparam.py --model_name ModernNCA --output_file results.txt --n_trials 50 --dataset_name otto_group --num_encoder FourierFeaturesCos --no_num_encoder_trainable --scaler SquareScalingFeatures"
  "python optim_hyperparam.py --model_name ModernNCA --output_file results.txt --n_trials 50 --dataset_name california_housing --num_encoder FourierFeaturesCos --no_num_encoder_trainable --scaler SquareScalingFeatures"
  "python optim_hyperparam.py --model_name ModernNCA --output_file results.txt --n_trials 50 --dataset_name higgs --num_encoder FourierFeaturesCos --no_num_encoder_trainable --scaler SquareScalingFeatures"
)

${commands[$SLURM_ARRAY_TASK_ID]}