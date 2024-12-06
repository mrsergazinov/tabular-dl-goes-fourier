
import os
import random
import time
import typing as ty
import yaml
import argparse

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import optuna

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Import models and encoders
os.chdir('/home/mrsergazinov/TabLLM/')
from base_models.mlp import MLP
from base_models.tabTransformer import TabTransformer
from base_models.modernNCA import ModernNCA
from encoders.numEncoders import (
    FourierFeatures, 
    BinningFeatures, 
    ComboFeatures,
    SquareScalingFeatures,
)
from train_eval import (
    set_seed,
    load_data,
    preprocess_data,
    train_and_evaluate_model
)

# Predefined seeds
SEEDS = [42, 7, 123, 2020, 999, 77, 88, 1010, 2021, 3030]

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train and evaluate models with different parameters.')
    parser.add_argument('--dataset_name', type=str, default='adult',
                        choices=['adult', 'california_housing', 'otto_group', 'higgs'], help='Name of the dataset to use.')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Fraction of the dataset to include in the test split.')
    parser.add_argument('--model_name', type=str, default='TabTransformer',
                        choices=['MLP', 'TabTransformer', 'ModernNCA'], help='Name of the model to use.')
    parser.add_argument('--num_encoder', type=str, default='None',
                        choices=['None', 'FourierFeatures', 'BinningFeatures', 'ComboFeatures'], help='Numerical encoder to use.')
    parser.add_argument('--num_encoder_trainable', action='store_true', help='Set numerical encoder as trainable.')
    parser.add_argument('--no_num_encoder_trainable', dest='num_encoder_trainable', action='store_false', help='Set numerical encoder as not trainable.')
    parser.set_defaults(num_encoder_trainable=True)
    parser.add_argument('--scaler', type=str, default=None,
                        choices=['SquareScalingFeatures'], help='Name of the scaler to use.')
    parser.add_argument('--config_file', type=str, default='configs/hyperparam.yaml',
                        help='Path to the configuration file.')
    parser.add_argument('--output_file', type=str, default='results.txt',
                        help='Path to the output text file.')
    parser.add_argument('--n_run', type=int, default=10, help='Number of runs for replication.')
    parser.add_argument('--n_trials', type=int, default=10, help='Number of trials for optimization.')
    args = parser.parse_args()
    return args

def objective(trial):
    # Load the configuration files
    with open(params['config_file'], 'r') as file:
        config = yaml.safe_load(file)
    if params['num_encoder'] is not None:
        with open('configs/' + params['model_name'] + '_' + params['dataset_name'] + '.yaml', 'r') as file:
            model_config = yaml.safe_load(file)

    # If encoder is not none, then model hyperparameters should've been set
    if params['num_encoder'] is not None:
        # load model best hyperparams for this dataset
        config[params['model_name']] = model_config[params['model_name']]
        config['training'] = model_config['training']
        print(config)
        for key, value in config[params['num_encoder']].items():
            if isinstance(value[0], float):
                config[params['num_encoder']][key] = trial.suggest_float(key, value[0], value[1])
            elif isinstance(value[0], int):
                config[params['num_encoder']][key] = trial.suggest_int(key, value[0], value[1])
            else:
                raise ValueError(f'Invalid type for {key}: {type(value)}')
    else:
        for key, value in config[params['model_name']].items():
            if isinstance(value[0], float):
                config[params['model_name']][key] = trial.suggest_float(key, value[0], value[1])
            elif isinstance(value[0], int):
                config[params['model_name']][key] = trial.suggest_int(key, value[0], value[1])
            else:
                raise ValueError(f'Invalid type for {key}: {type(value)}')
        for key, value in config['training'].items():
            if isinstance(value[0], float):
                config['training'][key] = trial.suggest_float(key, value[0], value[1])
            elif isinstance(value[0], int):
                config['training'][key] = trial.suggest_int(key, value[0], value[1])
            else:
                raise ValueError(f'Invalid type for {key}: {type(value)}')

    # Train and evaluate model
    metric = train_and_evaluate_model(
        X_train_num=X_train_num,
        X_test_num=X_val_num,
        X_train_cat=X_train_cat,
        X_test_cat=X_val_cat,
        y_train=y_train,
        y_test=y_val,
        task_type=task_type,
        config=config,
        params=params,
        verbose_training=True,
    )

    return metric

if __name__ == '__main__':
    args = parse_arguments()

    # Set parameters
    params = {
        'dataset_name': args.dataset_name,
        'test_size': args.test_size,
        'model_name': args.model_name,
        'num_encoder': None if args.num_encoder == 'None' else args.num_encoder,
        'num_encoder_trainable': args.num_encoder_trainable,
        'scaler': args.scaler,
        'config_file': args.config_file,
        'output_file': args.output_file,
        'random_state': 0,
    }

    # Run the experiment
    X, y, task_type = load_data(params)
    (y_train, 
     y_val, 
     y_test, 
     X_train_num, 
     X_val_num, 
     X_test_num, 
     X_train_cat, 
     X_val_cat, 
     X_test_cat) = preprocess_data(X, y, task_type, params)
    if task_type in ('binary_classification', 'multiclass_classification'):
        direction = 'maximize'
    else:
        direction = 'minimize'
    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=args.n_trials)
    best_params = study.best_params

    # Load the configuration files
    # Load the configuration file
    with open(params['config_file'], 'r') as file:
        config = yaml.safe_load(file)
    if params['num_encoder'] is not None:
        with open('configs/' + params['model_name'] + '_' + params['dataset_name'] + '.yaml', 'r') as file:
            model_config = yaml.safe_load(file)

    # Save best hyperparams
    if params['num_encoder'] is not None:
        model_config[params['num_encoder']] = {key: best_params[key] for key in config[params['num_encoder']]}
    else:
        model_config = {}
        model_config[params['model_name']] = {key: best_params[key] for key in config[params['model_name']]}
        model_config['training'] = {key: best_params[key] for key in config['training']}
                                            
    # Save best params in the config file for this dataset
    if params['num_encoder'] is not None:
        path = f'configs/{params["model_name"]}_{params["num_encoder"]}_{params["scaler"]}_{params["num_encoder_trainable"]}_{params["dataset_name"]}.yaml'
    else:
        path = f'configs/{params["model_name"]}_{params["dataset_name"]}.yaml'
    with open(path, 'w') as file:
        yaml.dump(model_config, file)

    # Train and evaluate the model with the best hyperparameters
    metrics = []
    for idx, seed in enumerate(SEEDS[:args.n_run]):
        set_seed(seed)
        params['random_state'] = seed
        metric = train_and_evaluate_model(
            X_train_num=X_train_num,
            X_test_num=X_test_num,
            X_train_cat=X_train_cat,
            X_test_cat=X_test_cat,
            y_train=y_train,
            y_test=y_test,
            task_type=task_type,
            params=params,
            config=model_config,
            verbose_training=False,
        )
        with open(args.output_file, 'a') as f:
            f.write(f'Parameters: {params}\n')
            f.write(f'Run {idx+1}/{args.n_run} | Seed: {seed} | Accuracy: {metric:.2f}%\n')
            f.write('-------------------------\n')
        metrics.append(metric)

    # Print the average accuracy
    mean_accuracy = sum(metrics) / args.n_run
    std_accuracy = np.std(metrics)
    with open(args.output_file, 'a') as f:
        f.write('Parameters: {}\n'.format(params))
        f.write('Overall Average Metric over {} runs: {:.2f}% ± {:.2f}%\n'.format(args.n_run, mean_accuracy, std_accuracy))
        f.write('=========================\n')