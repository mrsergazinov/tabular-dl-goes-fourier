#!/bin/bash

#SBATCH --partition=gpu                           # Request GPU partition
#SBATCH --nodes=1                                 # Use 1 node
#SBATCH --gres=gpu:a30:4                          # Request 4 A30 GPUs
#SBATCH --time=06:00:00                           # Request 4 hours of wall time
#SBATCH --ntasks=1                                # Only open 1 instance of the server
#SBATCH --cpus-per-task=4                         # Use 4 CPU cores
#SBATCH --mem=128G                                # Use 128GB of RAM
#SBATCH --output=server.output.%j                 # Send output stream to file named 'server.output.{jobid}'

# Activate the Conda environment
source ~/.bashrc
conda activate tabllm                             # Activate the Conda environment named "tabllm"

# Start JupyterLab on the allocated GPU node and dynamic port
python train_eval_adult.py --model_name MLP --num_encoder None --output_file results.txt
python train_eval_adult.py --model_name MLP --num_encoder BinningFeatures --num_encoder_trainable --output_file results.txt
python train_eval_adult.py --model_name MLP --num_encoder FourierFeatures --num_encoder_trainable --output_file results.txt
python train_eval_adult.py --model_name MLP --num_encoder BinningFeatures --no_num_encoder_trainable --output_file results.txt
python train_eval_adult.py --model_name MLP --num_encoder FourierFeatures --no_num_encoder_trainable --output_file results.txt

python train_eval_adult.py --model_name TabTransformer --num_encoder None --output_file results.txt
python train_eval_adult.py --model_name TabTransformer --num_encoder BinningFeatures --num_encoder_trainable --output_file results.txt
python train_eval_adult.py --model_name TabTransformer --num_encoder FourierFeatures --num_encoder_trainable --output_file results.txt
python train_eval_adult.py --model_name TabTransformer --num_encoder BinningFeatures --no_num_encoder_trainable --output_file results.txt
python train_eval_adult.py --model_name TabTransformer --num_encoder FourierFeatures --no_num_encoder_trainable --output_file results.txt

python train_eval_adult.py --model_name ModernNCA --num_encoder None --output_file results.txt
python train_eval_adult.py --model_name ModernNCA --num_encoder BinningFeatures --num_encoder_trainable --output_file results.txt
python train_eval_adult.py --model_name ModernNCA --num_encoder FourierFeatures --num_encoder_trainable --output_file results.txt
python train_eval_adult.py --model_name ModernNCA --num_encoder BinningFeatures --no_num_encoder_trainable --output_file results.txt
python train_eval_adult.py --model_name ModernNCA --num_encoder FourierFeatures --no_num_encoder_trainable --output_file results.txt
