import math
import time

from typing import List, Optional, Callable 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

class MLP(nn.Module):
    def __init__(
            self,
            *,
            d_in_num: int,
            d_in_cat: int,
            d_out: int,
            num_layers: int,
            d_layers: int,   
            dropout: float,
            num_encoder: Optional[nn.Module] = None,
        ) -> None:
        super().__init__()
        
        #----------------------------------------------
        # Define the numerical encoder
        self.num_encoder = num_encoder
        d_in_num = d_in_num if num_encoder is None else num_encoder.d_out
        #----------------------------------------------

        d_in = d_in_num + d_in_cat if d_in_cat is not None else d_in_num
        self.dropout = dropout
        self.d_out = d_out
        
        d_layers = [(d_in, d_layers)] + [(d_layers, d_layers) for _ in range(num_layers - 1)]
        self.layers = nn.ModuleList(
            [
                nn.Linear(d_in, d_out)
                for (d_in, d_out) in d_layers
            ]
        )
        self.head = nn.Linear(d_layers[-1][1], d_out)


    def forward(self, x_num, x_cat = None):
        #----------------------------------------------
        # Transform numerical features
        if self.num_encoder is not None:
            x_num = self.num_encoder(x_num)
        #----------------------------------------------
        x = x_num
        if x_cat is not None:
            x = torch.cat([x, x_cat], dim=1)
        for layer in self.layers:
            x = layer(x)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, self.dropout, self.training)
        logit = self.head(x)    
        if self.d_out == 1:
            logit = logit.squeeze()    
        return  logit
    
    def fit(
            self, 
            X_num_train: torch.Tensor,
            X_cat_train: Optional[torch.Tensor],
            y_train: torch.Tensor,
            criterion: nn.Module,
            batch_size: int,
            epochs: int,
            learning_rate: float,
            weight_decay: float,
            verbose: bool = True,
        ) -> None:
        super().train()

        # Determine the device 
        device = next(self.parameters()).device

        # Define the dataset and dataloader
        if X_cat_train is not None:
            train_dataset = TensorDataset(X_num_train, X_cat_train, y_train)
        else:
            train_dataset = TensorDataset(X_num_train, y_train)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
        )

        # Define loss and optimizer
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Training loop
        for epoch in range(epochs):
            start_time = time.time()
            epoch_loss = 0.0
            correct = 0
            total = 0

            for itr, batch in enumerate(train_loader):
                if X_cat_train is not None:
                    X_num_batch, X_cat_batch, y_batch = batch
                    X_cat_batch = X_cat_batch.to(device)
                else:   
                    X_num_batch, y_batch = batch
                    X_cat_batch = None
                X_num_batch = X_num_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()
                # Forward pass with separate categorical and numerical features
                logits = self(X_num_batch, X_cat_batch)
                loss = criterion(logits, y_batch)
                loss += self.num_encoder.regularization_loss() if self.num_encoder is not None else 0
                loss.backward()
                optimizer.step()
                if self.num_encoder is not None:
                    self.num_encoder.clamp_weights()

                epoch_loss += loss.item() * y_batch.size(0)
                total += y_batch.size(0)
                if verbose and itr % 50 == 0:
                    print(f'Iteration [{itr}/{len(train_loader)}] | Loss: {loss.item():.4f}')

            epoch_loss = epoch_loss / total
            epoch_time = time.time() - start_time
            if verbose: 
                print(f'Epoch [{epoch+1}/{epochs}] | Loss: {epoch_loss:.4f} | Time: {epoch_time:.2f}s')


    def evaluate(
        self,
        X_num_test: torch.Tensor,
        X_cat_test: Optional[torch.Tensor],
        y_test: torch.Tensor,
        criterion: Callable[[torch.Tensor, torch.Tensor], float],
        batch_size: int,
        verbose: bool = True,
    ) -> float:
        self.eval()

        # Determine the device
        device = next(self.parameters()).device

        # Define the dataset and dataloader
        if X_cat_test is not None:
            test_dataset = TensorDataset(X_num_test, X_cat_test, y_test)
        else:
            test_dataset = TensorDataset(X_num_test, y_test)
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
        )

        # Evaluate the model
        total_metric = 0.0
        total_samples = 0
        with torch.no_grad():
            for batch in test_loader:
                if X_cat_test is not None:
                    X_num_batch, X_cat_batch, y_batch = batch
                    X_cat_batch = X_cat_batch.to(device)
                else:
                    X_num_batch, y_batch = batch
                    X_cat_batch = None
                X_num_batch = X_num_batch.to(device)
                y_batch = y_batch.to(device)

                # Forward pass
                logits = self(X_num_batch, X_cat_batch)

                # Compute metric using the provided criterion
                metric = criterion(logits, y_batch)
                total_metric += metric * y_batch.size(0)
                total_samples += y_batch.size(0)

        average_metric = total_metric / total_samples
        if verbose:
            print(f'Evaluation Metric: {average_metric:.4f}')
        return average_metric
