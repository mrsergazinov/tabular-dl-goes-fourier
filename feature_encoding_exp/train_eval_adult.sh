#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a30:1              # Request 1 A30 GPU per task
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4             # Adjust CPU cores per task if needed
#SBATCH --mem=16G                     # Adjust memory per task if needed
#SBATCH --time=06:00:00
#SBATCH --array=0-2                   # Create an array job with indices from 0 to 5
#SBATCH --output=output_%A_%a.out     # Output file for each array task

# Activate the Conda environment
source ~/.bashrc
conda activate tabllm

# Execute the command corresponding to the array task ID
CMD=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" train_eval_adult.txt)
eval $CMD
