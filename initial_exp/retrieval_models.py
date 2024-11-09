import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from typing import Optional
from models import HuggingFaceLLaMA

class PLREmbeddings(nn.Module):
    """The PLR embeddings from the paper 'On Embeddings for Numerical Features in Tabular Deep Learning'.
    
    Additionally, the 'lite' option is added. Setting it to `False` gives you the original PLR
    embedding from the above paper. We noticed that `lite=True` makes the embeddings
    noticeably more lightweight without critical performance loss, and we used that for our model.
    """  # noqa: E501

    def __init__(
        self,
        n_features: int,
        n_frequencies: int,
        frequency_scale: float,
        d_embedding: int,
        lite: bool,
    ) -> None:
        super().__init__()
        self.frequencies = Parameter(
            torch.normal(0.0, frequency_scale, (n_features, n_frequencies))
        )
        self.lite = lite
        if lite:
            self.linear = nn.Linear(2 * n_frequencies, d_embedding)
        else:
            self.weight = Parameter(torch.empty(n_features, 2 * n_frequencies, d_embedding))
            self.bias = Parameter(torch.empty(n_features, d_embedding))
            self.reset_parameters()
        self.activation = nn.ReLU()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            for i in range(self.weight.shape[0]):
                layer = nn.Linear(self.weight.shape[1], self.weight.shape[2])
                self.weight[i] = layer.weight.T
                self.bias[i] = layer.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 2
        x = 2 * torch.pi * self.frequencies[None] * x[..., None]
        x = torch.cat([torch.cos(x), torch.sin(x)], -1)
        if self.lite:
            x = self.linear(x)
        else:
            x = x[..., None] * self.weight[None]
            x = x.sum(-2)
            x = x + self.bias[None]
        x = self.activation(x)
        return x

class ModernNCA(nn.Module):
    def __init__(
        self,
        *,
        d_in_num: int,
        d_in_cat: int,
        d_out: int,
        dim: int,
        dropout: float,
        n_frequencies: int,
        frequency_scale: float,
        d_embedding: int,
        lite: bool,
        temperature: float = 1.0,
        sample_rate: float = 0.8,
        use_llama: bool = False,
        llama_model_name: str = "bert-base-uncased",
        start_layer: int = 0,
        end_layer: int = 1,
    ) -> None:
        super().__init__()
        self.d_in_num = d_in_num
        self.d_in_cat = d_in_cat
        self.d_out = d_out
        self.dim = dim
        self.dropout = dropout
        self.temperature = temperature
        self.sample_rate = sample_rate

        # Initialize PLREmbeddings directly
        self.num_embeddings = PLREmbeddings(
            n_features=d_in_num,
            n_frequencies=n_frequencies,
            frequency_scale=frequency_scale,
            d_embedding=d_embedding,
            lite=lite
        )

        # Define the encoder layer
        d_in_total = d_in_num * d_embedding + d_in_cat
        self.encoder = nn.Linear(d_in_total, dim)

        # LLAMA model
        self.use_llama = use_llama
        if use_llama:
            self.llama = HuggingFaceLLaMA(
                llama_model_name,
                start_layer=start_layer,
                end_layer=end_layer,
            )

            # Dimensional mapping between base model and LLaMA
            llama_hidden_dim = self.llama.model.config.hidden_size
            self.mapper1 = nn.Sequential(
                nn.Linear(d_in_total, llama_hidden_dim),
            )

    def llama_encoder(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mapper1(x) 
        x = x.unsqueeze(1).half() 
        x = self.llama(x)
        x = x.squeeze(1).float()
        # x = self.mapper2(x)
        return x

    def forward(
        self,
        x_num: torch.Tensor,
        x_cat: Optional[torch.Tensor],
        y: Optional[torch.Tensor],
        candidate_x_num: torch.Tensor,
        candidate_x_cat: Optional[torch.Tensor],
        candidate_y: torch.Tensor,
        is_train: bool
    ) -> torch.Tensor:
        if is_train:
            data_size = candidate_x_num.shape[0]
            retrieval_size = int(data_size * self.sample_rate)
            sample_idx = torch.randperm(data_size)[:retrieval_size]
        

        # Apply PLR embeddings to numerical features
        x = self.num_embeddings(x_num).flatten(1)          # Shape: [batch_size, d_in * d_embedding]
        candidate_x = self.num_embeddings(candidate_x_num).flatten(1)  # Shape: [num_candidates, d_in * d_embedding]

        # Concatenate numerical and categorical features
        if x_cat is not None:
            x = torch.cat([x, x_cat], dim=1)
        if candidate_x_cat is not None:
            candidate_x = torch.cat([candidate_x, candidate_x_cat], dim=1)

        # LLM encoder
        if self.use_llama:
            x = self.llama_encoder(x)
            candidate_x = self.llama_encoder(candidate_x)
        else:
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
        logits = torch.log(logits) - logsumexp.unsqueeze(1)
        
        return logits
