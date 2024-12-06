#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a30:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --array=0-2
#SBATCH --output=output_%A_%a.out

# Activate the Conda environment
source ~/.bashrc
conda activate tabllm

# Define the list of models
model_list=("MLP" "TabTransformer" "ModernNCA")
model=${model_list[$SLURM_ARRAY_TASK_ID]}

# Define the list of datasets
datasets=("adult" "otto_group" "california_housing" "higgs")

# Loop over datasets and run the command
for dataset in "${datasets[@]}"; do
    python optim_hyperparam.py --model_name "$model" --output_file results.txt --n_trials 50 --dataset_name "$dataset"
done