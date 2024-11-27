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

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Import models and encoders
os.chdir('/home/mrsergazinov/TabLLM/feature_encoding_exp/')
from base_models.mlp import MLP
from base_models.tabTransformer import TabTransformer
from base_models.modernNCA import ModernNCA
from encoders.numEncoders import FourierFeatures, BinningFeatures

MODELS = {
    'MLP': MLP,
    'TabTransformer': TabTransformer,
    'ModernNCA': ModernNCA
}

ENCODERS = {
    'FourierFeatures': FourierFeatures,
    'BinningFeatures': BinningFeatures,
}

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train and evaluate models with different parameters.')
    parser.add_argument('--model_name', type=str, default='TabTransformer',
                        choices=['MLP', 'TabTransformer', 'ModernNCA'], help='Name of the model to use.')
    parser.add_argument('--num_encoder', type=str, default='None',
                        choices=['None', 'FourierFeatures', 'BinningFeatures'], help='Numerical encoder to use.')
    parser.add_argument('--num_encoder_trainable', action='store_true', help='Set numerical encoder as trainable.')
    parser.add_argument('--no_num_encoder_trainable', dest='num_encoder_trainable', action='store_false', help='Set numerical encoder as not trainable.')
    parser.set_defaults(num_encoder_trainable=True)
    parser.add_argument('--output_file', type=str, default='results.txt',
                        help='Path to the output text file.')
    args = parser.parse_args()
    return args

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def accuracy_criterion(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    with torch.no_grad():
        _, predicted = torch.max(outputs, dim=1)
        correct = (predicted == targets).sum().item()
        accuracy = correct / targets.size(0)
    return accuracy * 100

if __name__ == '__main__':
    args = parse_arguments()

    # Load config
    with open('configs/adult.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Set seed
    set_seed(config.get('seed', 42))  # Use default seed if not specified

    # Set parameters
    params = {
        'model_name': args.model_name,
        'num_encoder': None if args.num_encoder == 'None' else args.num_encoder,
        'num_encoder_trainable': args.num_encoder_trainable
    }

    # Load dataset
    data = fetch_openml("adult", version=2, as_frame=True)
    X = data['data']
    y = data['target']

    # Identify categorical and numerical columns
    categorical_columns = X.select_dtypes(include=['category', 'object']).columns.tolist()
    numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Split the data into training and test sets before processing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config.get('seed', 42)
    )

    # Encode the target variable
    le_target = LabelEncoder()
    y_train = le_target.fit_transform(y_train)
    y_test = le_target.transform(y_test)

    # Process categorical columns
    d_in_cat = None
    if params['model_name'] == 'TabTransformer':
        # encode with label encoding
        d_in_cat = []
        for col in categorical_columns:
            le = LabelEncoder()
            X_train[col] = le.fit_transform(X_train[col])
            X_test[col] = le.transform(X_test[col])
            d_in_cat.append(len(le.classes_))
        X_train_cat = X_train[categorical_columns].copy()
        X_test_cat = X_test[categorical_columns].copy()
    else:
        X_train_cat = pd.get_dummies(X_train[categorical_columns], drop_first=True)
        X_test_cat = pd.get_dummies(X_test[categorical_columns], drop_first=True)

        # Align the test and train categorical features to prevent data leakage
        X_train_cat, X_test_cat = X_train_cat.align(X_test_cat, join='left', axis=1, fill_value=0)

    # Process numerical columns
    numerical_transformer = StandardScaler()
    X_train_num = numerical_transformer.fit_transform(X_train[numerical_columns])
    X_test_num = numerical_transformer.transform(X_test[numerical_columns])

    if (not params['num_encoder_trainable']) and (params['num_encoder'] is not None):
        num_encoder = ENCODERS[params['num_encoder']](
            n_features=X_train_num.shape[1],
            **config[params['num_encoder']],
        )
        with torch.no_grad():
            X_train_num = num_encoder(torch.from_numpy(X_train_num).float(), trainable=False)
            X_test_num = num_encoder(torch.from_numpy(X_test_num).float(), trainable=False)
            X_train_num = X_train_num.numpy()
            X_test_num = X_test_num.numpy()
        num_encoder = None
    elif params['num_encoder'] is not None:
        num_encoder = ENCODERS[params['num_encoder']](
            n_features=X_train_num.shape[1],
            **config[params['num_encoder']],
        )
    else:
        num_encoder = None

    # Convert to tensors
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)
    X_train_num = torch.tensor(X_train_num, dtype=torch.float32)
    X_test_num = torch.tensor(X_test_num, dtype=torch.float32)
    X_train_cat = torch.tensor(X_train_cat.values, dtype=torch.float32)
    X_test_cat = torch.tensor(X_test_cat.values, dtype=torch.float32)

    if params['model_name'] == 'TabTransformer':
        X_train_cat = X_train_cat.to(torch.long)
        X_test_cat = X_test_cat.to(torch.long)

    # Determine input dimensions
    d_in_num = X_train_num.shape[1]
    d_in_cat = X_train_cat.shape[1] if d_in_cat is None else d_in_cat
    d_out = len(np.unique(y_train))

    # Define the model
    model = MODELS[params['model_name']](
        d_in_num=d_in_num,
        d_in_cat=d_in_cat,
        d_out=d_out,
        num_encoder=num_encoder,
        **config[params['model_name']],
    )

    # Move model to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Define the loss criterion
    loss_criterion = nn.CrossEntropyLoss()

    # Start training
    model.fit(
        X_num_train=X_train_num,
        X_cat_train=X_train_cat,
        y_train=y_train,
        criterion=loss_criterion,
        verbose=True,
        **config['training'],
    )

    # Evaluate the model
    avg_accuracy = model.evaluate(
        X_num_test=X_test_num,
        X_cat_test=X_test_cat,
        y_test=y_test,
        criterion=accuracy_criterion,
        batch_size=32,
        verbose=True,
    )

    # Write the parameters and evaluation result to the output file
    with open(args.output_file, 'a') as f:
        f.write('Parameters: {}\n'.format(params))
        f.write('Average Accuracy: {:.2f}%\n'.format(avg_accuracy))
        f.write('-------------------------\n')
