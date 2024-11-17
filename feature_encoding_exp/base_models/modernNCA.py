

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class ModernNCA(nn.Module):
    def __init__(
        self,
        d_in_num: int,
        d_in_cat: int,
        d_out: int,
        dim: int,
        dropout: float,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.d_in_num = d_in_num
        self.d_in_cat = d_in_cat
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