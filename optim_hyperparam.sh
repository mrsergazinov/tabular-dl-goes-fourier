#!/bin/bash

#SBATCH --partition=xgpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a30:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=168:00:00
#SBATCH --array=0-2
#SBATCH --output=output_%A_%a.out

# Activate the Conda environment
source ~/.bashrc
conda activate tabllm

# Define the list of models
model_list=("MLP" "TabTransformer" "ModernNCA")
model=${model_list[$SLURM_ARRAY_TASK_ID]}

# Define the list of datasets
datasets=("gesture_phase" "churn" "house" "santander" "covertype" "microsoft")

# Loop over datasets and run the command
for dataset in "${datasets[@]}"; do
    python optim_hyperparam.py --model_name "$model" --output_file results.txt --n_trials 50 --dataset_name "$dataset"
done