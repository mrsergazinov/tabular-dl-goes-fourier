#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a30:1
#SBATCH --ntasks=1
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=output_%A_%a.out

# Activate the Conda environment
source ~/.bashrc
conda activate tabllm

python optim_hyperparam.py --model_name TabTransformer --output_file results.txt --n_trials 50 --dataset_name adult --num_encoder FourierFeatures --num_encoder_trainable

# python optim_hyperparam.py --model_name TabTransformer --output_file results.txt --n_trials 50 --dataset_name otto_group --num_encoder ComboFeatures --no_num_encoder_trainable --scaler SquareScalingFeatures

# Define the lists
# models=("MLP" "TabTransformer" "ModernNCA")
# datasets=("adult" "otto_group" "california_housing" "higgs")
# encoder_options=(
#   "--num_encoder FourierFeatures --no_num_encoder_trainable --scaler SquareScalingFeatures"
#   "--num_encoder BinningFeatures --no_num_encoder_trainable --scaler SquareScalingFeatures"
#   "--num_encoder ComboFeatures --no_num_encoder_trainable --scaler SquareScalingFeatures"
#   "--num_encoder FourierFeatures --num_encoder_trainable"
# )

# # Generate all combinations
# combinations=()
# for model in "${models[@]}"; do
#   for dataset in "${datasets[@]}"; do
#     for encoder_option in "${encoder_options[@]}"; do
#       combinations+=("$model|$dataset|$encoder_option")
#     done
#   done
# done

# # Total number of combinations
# total_combinations=${#combinations[@]}

# # Combinations per task (48 combinations / 12 tasks)
# combinations_per_task=4

# # Calculate start and end indices for this task
# start_index=$(( SLURM_ARRAY_TASK_ID * combinations_per_task ))
# end_index=$(( start_index + combinations_per_task - 1 ))

# # Adjust end index if it exceeds total combinations
# if [ $end_index -ge $total_combinations ]; then
#   end_index=$(( total_combinations - 1 ))
# fi

# # Loop over assigned combinations
# for idx in $(seq $start_index $end_index); do
#   IFS='|' read -r model dataset encoder_option <<< "${combinations[$idx]}"

#   # Run the command
#   python optim_hyperparam.py --model_name "$model" \
#     --output_file results.txt --n_trials 50 \
#     --dataset_name "$dataset" $encoder_option
# done