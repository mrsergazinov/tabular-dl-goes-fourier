import time

from typing import List, Optional, Callable 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

class IndexedTensorDataset(Dataset):
    def __init__(
            self, 
            tensors_num: torch.Tensor,
            tensors_cat: Optional[torch.Tensor],
            targets: torch.Tensor,
        ) -> None:
        self.tensors_num = tensors_num
        self.tensors_cat = tensors_cat if tensors_cat is not None else None
        self.targets = targets
        self.indices = torch.arange(len(tensors_num), dtype=torch.long)

    def __getitem__(self, index):
        if self.tensors_cat is None:
            return (self.tensors_num[index], None, self.targets[index], self.indices[index])
        return (self.tensors_num[index], self.tensors_cat[index], self.targets[index], self.indices[index])

    def __len__(self):
        return len(self.tensors_num)

class ModernNCA(nn.Module):
    def __init__(
        self,
        d_in_num: int,
        d_in_cat: int,
        d_out: int,
        dim: int,
        dropout: float,
        temperature: float = 1.0,
        num_encoder: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()

        #----------------------------------------------
        # Define the numerical encoder
        self.num_encoder = num_encoder
        d_in_num = d_in_num if num_encoder is None else num_encoder.d_out 
        #----------------------------------------------

        self.d_in_num = d_in_num
        self.d_in_cat = d_in_cat if d_in_cat is not None else 0
        self.d_out = d_out
        self.dim = dim
        self.dropout = dropout
        self.temperature = temperature

        # Define the encoder layer
        d_in_total = d_in_num + d_in_cat
        self.encoder = nn.Sequential(
            nn.Linear(d_in_total, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(
        self,
        x_num: torch.Tensor,
        x_cat: Optional[torch.Tensor],
        candidate_x_num: torch.Tensor,
        candidate_x_cat: Optional[torch.Tensor],
        candidate_y: torch.Tensor
    ) -> torch.Tensor:
        #----------------------------------------------
        # Transform numerical features
        if self.num_encoder is not None:
            x_num = self.num_encoder(x_num)
            candidate_x_num = self.num_encoder(candidate_x_num)
        #----------------------------------------------

        # Concatenate numerical and categorical features
        x = x_num
        candidate_x = candidate_x_num
        if x_cat is not None:
            x = torch.cat([x, x_cat], dim=1)
        if candidate_x_cat is not None:
            candidate_x = torch.cat([candidate_x, candidate_x_cat], dim=1)

        x = self.encoder(x) # Shape: [batch_size, dim]
        candidate_x = self.encoder(candidate_x) # Shape: [num_candidates, dim]

        if self.d_out > 1:
            candidate_y = F.one_hot(candidate_y, self.d_out).float()
        elif candidate_y.dim() == 1:
            candidate_y = candidate_y.unsqueeze(-1).float()

        # Batch softmax
        logits, logsumexp = 0, 0
        for idx in range(0, candidate_y.shape[0], 5000):
            batch_candidate_y = candidate_y[idx:idx+5000]
            batch_candidate_x = candidate_x[idx:idx+5000]
            distances = torch.cdist(x, batch_candidate_x, p=2) / self.temperature
            exp_distances = torch.exp(-distances)
            logsumexp += torch.logsumexp(exp_distances, dim=1)
            logits += torch.mm(exp_distances, batch_candidate_y)
        
        if self.d_out > 1:
            logits = torch.log(logits) - logsumexp.unsqueeze(1)
        
        return logits

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
        sample_rate: float = 0.1,
        verbose: bool = True,
    ) -> None:
        self.train()
        device = next(self.parameters()).device

        # Define the dataset and dataloader
        train_dataset = IndexedTensorDataset(X_num_train, X_cat_train, y_train)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
        )

        # Define optimizer
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

            for itr, (X_num_batch, X_cat_batch, y_batch, idx_batch) in enumerate(train_loader):
                X_num_batch = X_num_batch.to(device)
                X_cat_batch = X_cat_batch.to(device) if X_cat_batch is not None else None
                y_batch = y_batch.to(device)

                # Exclude current batch indices from candidates
                mask = torch.ones(X_num_train.shape[0], dtype=torch.bool)
                mask[idx_batch] = False

                candidate_x_num = X_num_train[mask]
                candidate_x_cat = X_cat_train[mask] if X_cat_train is not None else None
                candidate_y = y_train[mask]

                # Sample candidates according to sample_rate
                num_candidates = int(len(candidate_y) * sample_rate)
                if num_candidates < len(candidate_y):
                    indices = torch.randperm(len(candidate_y))[:num_candidates]
                    candidate_x_num = candidate_x_num[indices]
                    candidate_x_cat = candidate_x_cat[indices] if candidate_x_cat is not None else None
                    candidate_y = candidate_y[indices]
                candidate_x_num = candidate_x_num.to(device)
                candidate_x_cat = candidate_x_cat.to(device) if candidate_x_cat is not None else None
                candidate_y = candidate_y.to(device)

                optimizer.zero_grad()
                # Forward pass
                logits = self(
                    x_num=X_num_batch,
                    x_cat=X_cat_batch,
                    candidate_x_num=candidate_x_num,
                    candidate_x_cat=candidate_x_cat,
                    candidate_y=candidate_y
                )

                # Compute loss
                loss = criterion(logits, y_batch)
                loss += self.num_encoder.regularization_loss() if self.num_encoder is not None else 0
                loss.backward()
                optimizer.step()
                if self.num_encoder is not None:
                    self.num_encoder.clamp_weights()

                epoch_loss += loss.item() * y_batch.size(0)
                total += y_batch.size(0)

                # Compute accuracy
                if self.d_out > 1:
                    preds = logits.argmax(dim=1)
                    correct += (preds == y_batch).sum().item()
                else:
                    preds = (logits > 0).float()
                    correct += (preds == y_batch).sum().item()

                if verbose and itr % 50 == 0:
                    print(f'Iteration [{itr}/{len(train_loader)}] | Loss: {loss.item():.4f}')

            epoch_loss = epoch_loss / total
            epoch_time = time.time() - start_time
            accuracy = correct / total

            if verbose:
                print(f'Epoch [{epoch+1}/{epochs}] | Loss: {epoch_loss:.4f} | Accuracy: {accuracy:.4f} | Time: {epoch_time:.2f}s')

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
        device = next(self.parameters()).device

        # Define the dataset and dataloader
        test_dataset = IndexedTensorDataset(X_num_test, X_cat_test, y_test)
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
        )

        total_metric = 0.0
        total_samples = 0

        # Prepare candidate data (using test data as candidates)
        candidate_x_num = X_num_test.to(device)
        candidate_x_cat = X_cat_test.to(device) if X_cat_test is not None else None
        candidate_y = y_test.to(device)

        with torch.no_grad():
            for X_num_batch, X_cat_batch, y_batch, idx_batch in test_loader:
                X_num_batch = X_num_batch.to(device)
                X_cat_batch = X_cat_batch.to(device) if X_cat_batch is not None else None
                y_batch = y_batch.to(device)

                # Exclude current batch indices from candidates
                mask = torch.ones(len(candidate_y), dtype=torch.bool)
                mask[idx_batch] = False
                candidate_x_num_batch = candidate_x_num[mask]
                candidate_x_cat_batch = candidate_x_cat[mask] if candidate_x_cat is not None else None
                candidate_y_batch = candidate_y[mask]

                # Forward pass
                logits = self(
                    x_num=X_num_batch,
                    x_cat=X_cat_batch,
                    candidate_x_num=candidate_x_num_batch,
                    candidate_x_cat=candidate_x_cat_batch,
                    candidate_y=candidate_y_batch
                )

                # Compute metric
                metric = criterion(logits, y_batch)
                total_metric += metric * y_batch.size(0)
                total_samples += y_batch.size(0)

        average_metric = total_metric / total_samples
        if verbose:
             print(f'Evaluation Metric: {average_metric:.4f}')
        return average_metric

       