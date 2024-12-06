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

MODELS = {
    'MLP': MLP,
    'TabTransformer': TabTransformer,
    'ModernNCA': ModernNCA
}

ENCODERS = {
    'FourierFeatures': FourierFeatures,
    'BinningFeatures': BinningFeatures,
    'ComboFeatures': ComboFeatures,
}

SCALERS = {
    'SquareScalingFeatures': SquareScalingFeatures,
}

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
    parser.add_argument('--config_file', type=str, default='configs/adult.yaml',
                        help='Path to the configuration file.')
    parser.add_argument('--output_file', type=str, default='results.txt',
                        help='Path to the output text file.')
    parser.add_argument('--n_run', type=int, default=10, help='Number of runs for replication.')
    args = parser.parse_args()
    return args

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def accuracy_score(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    with torch.no_grad():
        _, predicted = torch.max(outputs, dim=1)
        correct = (predicted == targets).sum().item()
        accuracy = correct / targets.size(0)
    return accuracy * 100

def root_mean_squared_error(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    with torch.no_grad():
        loss = torch.sqrt(torch.nn.functional.mse_loss(outputs, targets))
    return loss.item()

def load_data(
        params: ty.Dict[str, ty.Any],
    ) -> ty.Tuple[pd.DataFrame, pd.Series, str]:
    dataset_name = params['dataset_name']
    if dataset_name == 'adult':
        X, y = fetch_openml('adult', version=2, as_frame=True, return_X_y=True)
        task_type = 'binary_classification'
    elif dataset_name == 'california_housing':
        X, y = fetch_openml('california_housing', version=7, as_frame=True, return_X_y=True)
        task_type = 'regression'
    elif dataset_name == 'otto_group':
        X, y = fetch_openml('Otto-Group-Product-Classification-Challenge', version=1, as_frame=True, return_X_y=True)
        task_type = 'multiclass_classification'
    elif dataset_name == 'higgs':
        X, y = fetch_openml('higgs', version=2, as_frame=True, return_X_y=True)
        task_type = 'binary_classification'
    else:
        raise ValueError(f'Invalid dataset_name: {dataset_name}')
    return X, y, task_type

def preprocess_data(
        X: pd.DataFrame, 
        y: pd.Series, 
        task_type: str, 
        params: ty.Dict[str, ty.Any],
    ) -> ty.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Remove rows with missing values in features or target
    missing_rows = X.isnull().any(axis=1) | y.isnull()
    X = X[~missing_rows].reset_index(drop=True)
    y = y[~missing_rows].reset_index(drop=True)

    # Split data into training, validation, and test sets
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=params['test_size'], random_state=params['random_state']
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=params['test_size'], random_state=params['random_state']
    )

    # Identify numerical and categorical columns
    numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # Scale numerical features
    if numerical_columns:
        numerical_transformer = StandardScaler()
        X_train_num = numerical_transformer.fit_transform(X_train[numerical_columns])
        X_val_num = numerical_transformer.transform(X_val[numerical_columns])
        X_test_num = numerical_transformer.transform(X_test[numerical_columns])
        X_train_num = torch.tensor(X_train_num, dtype=torch.float32)
        X_val_num = torch.tensor(X_val_num, dtype=torch.float32)
        X_test_num = torch.tensor(X_test_num, dtype=torch.float32)
    else:
        X_train_num = X_val_num = X_test_num = None

    # Encode categorical features
    if categorical_columns:
        # Handle unseen categories in the validation and test sets
        for col in categorical_columns:
            train_categories = set(X_train[col])
            most_frequent_cat = X_train[col].value_counts().idxmax()
            X_val[col] = X_val[col].apply(
                lambda x: x if x in train_categories else most_frequent_cat
            )
            X_test[col] = X_test[col].apply(
                lambda x: x if x in train_categories else most_frequent_cat
            )

        if params['model_name'] == 'TabTransformer':
            # Label encode categorical features
            label_encoders = {}
            X_train_cat = X_train[categorical_columns].copy()
            X_val_cat = X_val[categorical_columns].copy()
            X_test_cat = X_test[categorical_columns].copy()
            for col in categorical_columns:
                le = LabelEncoder()
                X_train_cat[col] = le.fit_transform(X_train_cat[col])
                X_val_cat[col] = le.transform(X_val_cat[col])
                X_test_cat[col] = le.transform(X_test_cat[col])
                label_encoders[col] = le
            X_train_cat = torch.tensor(X_train_cat.values, dtype=torch.long)
            X_val_cat = torch.tensor(X_val_cat.values, dtype=torch.long)
            X_test_cat = torch.tensor(X_test_cat.values, dtype=torch.long)
        else:
            # One-hot encode categorical features
            X_train_cat = pd.get_dummies(X_train[categorical_columns], drop_first=True)
            X_val_cat = pd.get_dummies(X_val[categorical_columns], drop_first=True)
            X_test_cat = pd.get_dummies(X_test[categorical_columns], drop_first=True)
            X_val_cat = X_val_cat.reindex(columns=X_train_cat.columns, fill_value=False)
            X_test_cat = X_test_cat.reindex(columns=X_train_cat.columns, fill_value=False)
            X_train_cat = torch.tensor(X_train_cat.values, dtype=torch.float32)
            X_val_cat = torch.tensor(X_val_cat.values, dtype=torch.float32)
            X_test_cat = torch.tensor(X_test_cat.values, dtype=torch.float32)
    else:
        X_train_cat = X_val_cat = X_test_cat = None

    # Encode target variable
    if task_type == 'binary_classification':
        le_target = LabelEncoder()
        y_train = le_target.fit_transform(y_train)
        y_val = le_target.transform(y_val)
        y_test = le_target.transform(y_test)
        y_train = torch.tensor(y_train, dtype=torch.long)
        y_val = torch.tensor(y_val, dtype=torch.long)
        y_test = torch.tensor(y_test, dtype=torch.long)
    elif task_type == 'multiclass_classification':
        # Label encode target variable and convert to tensor
        le_target = LabelEncoder()
        y_train = le_target.fit_transform(y_train)
        y_val = le_target.transform(y_val)
        y_test = le_target.transform(y_test)
        y_train = torch.tensor(y_train, dtype=torch.long)
        y_val = torch.tensor(y_val, dtype=torch.long)
        y_test = torch.tensor(y_test, dtype=torch.long)
    else:
        # Scale target variable to be between 0 and 1
        scaler = StandardScaler()
        y_train = scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
        y_val = scaler.transform(y_val.values.reshape(-1, 1)).flatten()
        y_test = scaler.transform(y_test.values.reshape(-1, 1)).flatten()
        y_train = torch.tensor(y_train, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)

    return y_train, y_val, y_test, X_train_num, X_val_num, X_test_num, X_train_cat, X_val_cat, X_test_cat

def train_and_evaluate_model(
        X_train_num: torch.Tensor,
        X_test_num: torch.Tensor,
        X_train_cat: torch.Tensor,
        X_test_cat: torch.Tensor,
        y_train: torch.Tensor,
        y_test: torch.Tensor,
        task_type: str,
        config: ty.Dict[str, ty.Any],
        params: ty.Dict[str, ty.Any],
        verbose_training: bool = True,
) -> float:
    # Numerical feature encoder
    if (not params['num_encoder_trainable']) and (params['num_encoder'] is not None):
        # Encode numerical features via random features
        num_encoder = ENCODERS[params['num_encoder']](
            n_features=X_train_num.shape[1],
            **config[params['num_encoder']],
            trainable=params['num_encoder_trainable'],
        )
        with torch.no_grad():
            X_train_num = num_encoder(X_train_num)
            X_test_num = num_encoder(X_test_num)
        # Optionally, learnable scaling of random features
        if params['scaler'] is not None:
            num_encoder = SCALERS[params['scaler']](
                n_features=X_train_num.shape[1],
            )
        else:
            num_encoder = None
    elif params['num_encoder'] is not None:
        num_encoder = ENCODERS[params['num_encoder']](
            n_features=X_train_num.shape[1],
            **config[params['num_encoder']],
        )
    else:
        num_encoder = None

    # Determine input dimensions
    d_in_num = X_train_num.shape[1]
    d_in_cat = X_train_cat.shape[1] if X_train_cat is not None else None
    if task_type in ['binary_classification', 'multiclass_classification']:
        d_out = len(np.unique(y_train))
    elif task_type == 'regression':
        d_out = 1
    else:
        raise ValueError(f'Invalid task_type: {task_type}')

    # Define the model
    model = MODELS[params['model_name']](
        d_in_num=d_in_num,
        d_in_cat=d_in_cat,
        d_out=d_out,
        num_encoder=num_encoder,
        **config[params['model_name']],
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Define loss and metric based on task_type
    if task_type == 'regression':
        criterion = torch.nn.MSELoss()
        metric = root_mean_squared_error  # Replace with appropriate function
    elif task_type == 'binary_classification':
        criterion = torch.nn.CrossEntropyLoss()
        metric = accuracy_score  # Replace with appropriate function
    else:  # multi-class classification
        criterion = torch.nn.CrossEntropyLoss()
        metric = accuracy_score  # Replace with appropriate function

    # Start training
    model.fit(
        X_num_train=X_train_num,
        X_cat_train=X_train_cat,
        y_train=y_train,
        criterion=criterion,
        verbose=verbose_training,
        **config['training'],
    )

    # Evaluate the model
    metric = model.evaluate(
        X_num_test=X_test_num,
        X_cat_test=X_test_cat,
        y_test=y_test,
        criterion=metric,
        batch_size=32,
        verbose=verbose_training,
    )

    return metric

def objective(trial):
    # Sample hyperparameters
    with open(params['config_file'], 'r') as file:
        config = yaml.safe_load(file)

    # Sample model hyperparameters
    for key, value in config[params['model_name']].items():
        if isinstance(value[0], float):
            config[params['model_name']][key] = trial.suggest_float(key, value[0], value[1])
        elif isinstance(value[0], int):
            config[params['model_name']][key] = trial.suggest_int(key, value[0], value[1])
        else:
            raise ValueError(f'Invalid type for {key}: {type(value)}')
        
    # Sample traininig hyperparameters
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
        verbose_training=False,
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
     y_val, y_test, 
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
    study.optimize(objective, n_trials=100)
    best_params = study.best_params

    # Save best params in the config file for this dataset
    with open(params['config_file'], 'r') as file:
        config = yaml.safe_load(file)
    for key, value in best_params.items():
        if key in config[params['model_name']]:
            config[params['model_name']][key] = value
        elif key in config['training']:
            config['training'][key] = value
    with open('~/configs/' + params['model_name'] + '_' + params['dataset_name'] + '.yaml', 'w') as file:
        yaml.dump(config, file)

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
            config=config,
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