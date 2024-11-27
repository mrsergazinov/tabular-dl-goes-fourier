import os
import random
import time
import typing as ty

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from base_models.mlp import MLP
from feature_encoding_exp.encoders.numEncoders import FourierFeatures

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Constants
SEED = 42
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4

D_LAYERS = [64, 32]
DROPOUT = 0.5

N_FREQUENCIES = 16
FREQUENCY_SCALE = 10.0

set_seed(SEED)

if __name__ == '__main__':

    # Load dataset
    data = fetch_openml("adult", version=2, as_frame=True)
    X = data['data']
    y = data['target']

    # Identify categorical and numerical columns
    categorical_columns = X.select_dtypes(include=['category', 'object']).columns.tolist()
    numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Split the data into training and test sets before processing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    # Encode the target variable
    le_target = LabelEncoder()
    y_train = le_target.fit_transform(y_train)
    y_test = le_target.transform(y_test)

    # Process categorical columns
    X_train_cat = pd.get_dummies(X_train[categorical_columns], drop_first=True)
    X_test_cat = pd.get_dummies(X_test[categorical_columns], drop_first=True)

    # Align the test and train categorical features to prevent data leakage
    X_train_cat, X_test_cat = X_train_cat.align(X_test_cat, join='left', axis=1, fill_value=0)

    # Scale numerical columns
    numerical_transformer = StandardScaler()
    X_train_num = numerical_transformer.fit_transform(X_train[numerical_columns])
    X_test_num = numerical_transformer.transform(X_test[numerical_columns])

    # Convert to tensors
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)
    X_train_num = torch.tensor(X_train_num, dtype=torch.float32)
    X_test_num = torch.tensor(X_test_num, dtype=torch.float32)
    X_train_cat = torch.tensor(X_train_cat.values, dtype=torch.float32)
    X_test_cat = torch.tensor(X_test_cat.values, dtype=torch.float32)

    # Determine input dimensions
    d_in_num = X_train_num.shape[1]
    d_in_cat = X_train_cat.shape[1]
    d_out = len(np.unique(y_train))

    # Define numerical feature encoder
    num_encoder = FourierFeatures(
        n_features=d_in_num,
        n_frequencies=N_FREQUENCIES,
        frequency_scale=FREQUENCY_SCALE,
    )

    # Define the model
    model = MLP(
        d_in_num=d_in_num,
        d_in_cat=d_in_cat,
        d_out=d_out,
        d_layers=D_LAYERS,
        dropout=DROPOUT,
        num_encoder=num_encoder,
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
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # Define the accuracy criterion function
    def accuracy_criterion(outputs: torch.Tensor, targets: torch.Tensor) -> float:
        with torch.no_grad():
            _, predicted = torch.max(outputs, dim=1)
            correct = (predicted == targets).sum().item()
            accuracy = correct / targets.size(0)
        return accuracy

    # Evaluate the model using accuracy
    model.evaluate(
        X_num_test=X_test_num,
        X_cat_test=X_test_cat,
        y_test=y_test,
        criterion=accuracy_criterion,
        batch_size=BATCH_SIZE,
    )
