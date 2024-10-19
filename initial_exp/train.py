import os
import random
import time
from datetime import timedelta

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
)
from torch.utils.data import DataLoader, TensorDataset

# Import model classes from tabular_llama_model.py
from models import TabTransformer, TabularLLaMA


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config


def load_dataset(config):
    data = fetch_openml(config['dataset']['name'], version=config['dataset']['version'], as_frame=True)
    X = data['data'].copy()
    y = data['target']

    categorical_columns = X.select_dtypes(include=['category', 'object']).columns.tolist()
    numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    transformers = []
    if categorical_columns:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns))
    if numerical_columns:
        transformers.append(('num', StandardScaler(), numerical_columns))

    preprocessor = ColumnTransformer(transformers)
    X = preprocessor.fit_transform(X)
    X = pd.DataFrame(X.toarray() if hasattr(X, 'toarray') else X)

    le_target = LabelEncoder()
    y = le_target.fit_transform(y)
    input_dim = X.shape[1]

    return X, y, input_dim, le_target


def setup(rank, world_size, backend='nccl'):
    dist.init_process_group(
        backend=backend,
        init_method='env://',
        rank=rank,
        world_size=world_size,
        timeout=timedelta(minutes=10)
    )
    torch.cuda.set_device(rank)
    print(f"Process {rank} initialized.")


def cleanup():
    dist.destroy_process_group()


def train(rank, world_size, config):
    setup(rank, world_size)
    try:
        set_seed(config['dataset']['random_state'] + rank)

        X, y, input_dim, le_target = load_dataset(config)
        config['tabtransformer']['input_dim'] = input_dim
        output_classes = len(le_target.classes_)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config['dataset']['test_size'], random_state=config['dataset']['random_state']
        )

        X_train = X_train.apply(pd.to_numeric, errors='coerce')
        X_test = X_test.apply(pd.to_numeric, errors='coerce')
        X_train = X_train.fillna(0)
        X_test = X_test.fillna(0)

        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            test_dataset, num_replicas=world_size, rank=rank, shuffle=False
        )

        train_loader = DataLoader(
            train_dataset, batch_size=config['dataset']['batch_size'], sampler=train_sampler
        )
        test_loader = DataLoader(
            test_dataset, batch_size=config['dataset']['batch_size'], sampler=test_sampler
        )

        tabtransformer_config = config['tabtransformer']
        llama_config = config['llama']
        use_llama = llama_config['use_llama']
        start_layer = llama_config.get('start_layer', 0)
        end_layer = llama_config.get('end_layer', 1)
        token = os.getenv('HUGGINGFACE_TOKEN')  # Retrieve the token from environment variables

        # Initialize base model
        base_model = TabTransformer(
            input_dim=tabtransformer_config['input_dim'],
            output_dim=tabtransformer_config['output_dim'],
            num_heads=tabtransformer_config['num_heads'],
            num_layers=tabtransformer_config['num_layers']
        )

        # Initialize the combined model with the option to remove positional embeddings
        model = TabularLLaMA(
            base_model,
            base_output_dim=tabtransformer_config['output_dim'],
            llama_model_name=llama_config['model_name'] if use_llama else None,
            output_classes=output_classes,
            use_llama=use_llama,
            start_layer=start_layer,
            end_layer=end_layer,
            use_positional_embeddings=False,  # Set to False to remove positional embeddings
            token=token  # Pass the token here
        )

        device = torch.device(f'cuda:{rank}')
        torch.cuda.set_device(device)
        model.to(device)

        # Define FSDP parameters
        fsdp_params = dict(
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            device_id=torch.cuda.current_device(),
        )

        # Wrap only modules with trainable parameters
        model.base_model = FSDP(model.base_model, **fsdp_params)
        model.classifier = FSDP(model.classifier, **fsdp_params)

        if model.use_llama:
            model.mapper1 = FSDP(model.mapper1, **fsdp_params)
            model.mapper2 = FSDP(model.mapper2, **fsdp_params)
            # Do not wrap model.llama since parameters are frozen

        # Define loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config['training']['learning_rate']
        )

        # Training loop
        for epoch in range(config['training']['epochs']):
            model.train()
            train_sampler.set_epoch(epoch)
            start_time = time.time()
            epoch_loss = 0.0
            correct = 0
            total = 0

            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                optimizer.zero_grad()

                output = model(X_batch)
                loss = criterion(output, y_batch)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * y_batch.size(0)
                _, predicted = torch.max(output, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

            total_tensor = torch.tensor(total, device=device)
            correct_tensor = torch.tensor(correct, device=device)
            epoch_loss_tensor = torch.tensor(epoch_loss, device=device)

            dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(epoch_loss_tensor, op=dist.ReduceOp.SUM)

            epoch_loss = epoch_loss_tensor.item() / total_tensor.item()
            epoch_acc = 100 * correct_tensor.item() / total_tensor.item()
            epoch_time = time.time() - start_time

            if rank == 0:
                print(f'Epoch [{epoch+1}/{config["training"]["epochs"]}] | Loss: {epoch_loss:.4f} | '
                      f'Accuracy: {epoch_acc:.2f}% | Time: {epoch_time:.2f}s')

        evaluate(rank, model, test_loader, criterion, device)

    except Exception as e:
        print(f"Exception on rank {rank}: {e}")
        raise
    finally:
        cleanup()
        print(f"Process {rank} cleaned up successfully.")


def evaluate(rank, model, test_loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output = model(X_batch)
            loss = criterion(output, y_batch)
            test_loss += loss.item() * y_batch.size(0)

            _, predicted = torch.max(output, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

    total_tensor = torch.tensor(total, device=device)
    correct_tensor = torch.tensor(correct, device=device)
    test_loss_tensor = torch.tensor(test_loss, device=device)

    dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(test_loss_tensor, op=dist.ReduceOp.SUM)

    accuracy = 100 * correct_tensor.item() / total_tensor.item()
    test_loss = test_loss_tensor.item() / total_tensor.item()

    if rank == 0:
        print(f'Test Loss: {test_loss:.4f} | Test Accuracy: {accuracy:.2f}%')


def main():
    config = load_config('config.yaml')

    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    print(f"Running FSDP on {world_size} processes. Global rank: {rank}")

    train(rank, world_size, config)


if __name__ == "__main__":
    main()
