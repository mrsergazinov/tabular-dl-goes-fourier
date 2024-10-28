import os
import random
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset
from retrieval_models import ModernNCA

# Define a custom dataset to keep track of indices
class IndexedTensorDataset(Dataset):
    def __init__(self, tensors):
        self.tensors = tensors
        self.indices = torch.arange(len(tensors[0]), dtype=torch.long)

    def __getitem__(self, index):
        return (*[tensor[index] for tensor in self.tensors], self.indices[index])

    def __len__(self):
        return len(self.tensors[0])


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
    # Fetch the dataset
    data = fetch_openml(config['dataset']['name'], version=config['dataset']['version'], as_frame=True)
    X = data['data'].copy()
    y = data['target']

    # Identify categorical and numerical columns
    categorical_columns = X.select_dtypes(include=['category', 'object']).columns.tolist()
    numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Label encode each categorical column
    le_categorical = {}
    for col in categorical_columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        le_categorical[col] = le  # Save each encoder if needed for inverse transformation later

    # Scale numerical columns
    numerical_transformer = StandardScaler()
    X[numerical_columns] = numerical_transformer.fit_transform(X[numerical_columns])

    # Combine the numerical and categorical columns
    X = pd.DataFrame(X, columns=numerical_columns + categorical_columns)

    # Encode the target variable
    le_target = LabelEncoder()
    y = le_target.fit_transform(y)

    # Get input dimension
    input_dim = X.shape[1]

    return X, y, input_dim, le_target


def train(config):
    set_seed(config['dataset']['random_state'])

    X, y, input_dim, le_target = load_dataset(config)
    output_classes = len(le_target.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config['dataset']['test_size'], random_state=config['dataset']['random_state']
    )

    X_train = X_train.apply(pd.to_numeric, errors='coerce').fillna(0)
    X_test = X_test.apply(pd.to_numeric, errors='coerce').fillna(0)

    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Use IndexedTensorDataset to keep track of indices
    train_dataset = IndexedTensorDataset((X_train_tensor, y_train_tensor))
    test_dataset = IndexedTensorDataset((X_test_tensor, y_test_tensor))

    # Use standard DataLoader
    train_loader = DataLoader(
        train_dataset, batch_size=config['dataset']['batch_size'], shuffle=True,
        pin_memory=True, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config['dataset']['batch_size'], shuffle=False,
        pin_memory=True, num_workers=4
    )

    # Initialize the ModernNCA model
    model = ModernNCA(
        d_in=input_dim,
        d_out=output_classes,
        dim=config['model']['dim'],
        dropout=config['model']['dropout'],
        n_frequencies=config['model']['n_frequencies'],
        frequency_scale=config['model']['frequency_scale'],
        d_embedding=config['model']['d_embedding'],
        lite=config['model']['lite'],
        temperature=config['model']['temperature'],
        sample_rate=config['model']['sample_rate'],
        use_llama=config['model']['use_llama'],
        llama_model_name=config['model']['llama_model_name'],
        start_layer=config['model']['start_layer'],
        end_layer=config['model']['end_layer']
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.float()
    model.to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    # Training loop
    for epoch in range(config['training']['epochs']):
        model.train()
        start_time = time.time()
        epoch_loss = 0.0
        correct = 0
        total = 0

        for itr, (X_batch, y_batch, idx_batch) in enumerate(train_loader):
            X_batch = X_batch.float().to(device)
            y_batch = y_batch.to(device)

            # candidate_x and candidate_y (exclude current batch)
            mask = torch.isin(torch.arange(X_train_tensor.shape[0]), idx_batch)
            candidate_x = X_train_tensor[~mask].float().to(device)
            candidate_y = y_train_tensor[~mask].to(device)

            optimizer.zero_grad()

            # Forward pass with candidate embeddings
            logits = model(
                x=X_batch,
                y=y_batch,
                candidate_x=candidate_x,
                candidate_y=candidate_y,
                is_train=True
            )
            loss = criterion(logits, y_batch)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * y_batch.size(0)
            _, predicted = torch.max(logits, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

            # print every 100 batches
            if itr % 50 == 0:
                print(f'Epoch [{epoch+1}/{config["training"]["epochs"]}]: Batch [{itr+1}/{len(train_loader)}] | Accuracy: {correct/total:.2f}')

        epoch_loss = epoch_loss / total
        epoch_acc = 100 * correct / total
        epoch_time = time.time() - start_time

        print(f'Epoch [{epoch+1}/{config["training"]["epochs"]}] | Loss: {epoch_loss:.4f} | '
              f'Accuracy: {epoch_acc:.2f}% | Time: {epoch_time:.2f}s')

    evaluate(model, test_loader, criterion, device, X_train_tensor, y_train_tensor)


def evaluate(model, test_loader, criterion, device, X_train_tensor, y_train_tensor):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch, idx_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # Use entire training data as candidates during evaluation
            candidate_x = X_train_tensor.to(device)
            candidate_y = y_train_tensor.to(device)

            # Forward pass
            logits = model(
                x=X_batch,
                y=None,
                candidate_x=candidate_x,
                candidate_y=candidate_y,
                is_train=False
            )

            # Convert logits to predictions
            _, predicted = torch.max(logits, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

            # Compute loss
            loss = criterion(logits, y_batch)
            test_loss += loss.item() * y_batch.size(0)

    accuracy = 100 * correct / total
    test_loss = test_loss / total

    print(f'Test Loss: {test_loss:.4f} | Test Accuracy: {accuracy:.2f}%')


def main():
    config = load_config('retrieval_config.yaml')

    print("Running training on a single device.")

    train(config)


if __name__ == "__main__":
    main()
