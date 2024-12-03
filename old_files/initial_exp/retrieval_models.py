import numpy as np
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
        use_llama: bool = False,
        llama_model_name: str = "bert-base-uncased",
        start_layer: int = 0,
        end_layer: int = 1,
        **kwargs,
    ) -> None:
        super().__init__()
        self.d_in_num = d_in_num
        self.d_in_cat = d_in_cat
        self.d_out = d_out
        self.dim = dim
        self.dropout = dropout
        self.temperature = temperature

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
        candidate_y: torch.Tensor
    ) -> torch.Tensor:
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

# Version 1: Gaussian kernel
# class KernelNCA(nn.Module):
#     def __init__(
#         self,
#         d_in_num: int,
#         d_in_cat: int,
#         d_out: int,
#         temperature: float = 1.0,
#         kernel_type: str = 'gaussian',
#         kernel_lengthscale: float = 1.0,
#         **kwargs,
#     ) -> None:
#         super().__init__()
#         self.d_in_num = d_in_num
#         self.d_in_cat = d_in_cat
#         self.d_out = d_out
#         self.temperature = temperature

#         # Learnable per-feature length scales for the kernel
#         self.kernel_lengthscale = nn.Parameter(
#             torch.full((d_in_num,), kernel_lengthscale)
#         )

#     def kernel_function(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
#         # x1: (batch_size, num_features)
#         # x2: (num_candidates, num_features)
#         distances = (x1.unsqueeze(1) - x2.unsqueeze(0)) ** 2  # (batch_size, num_candidates, num_features)
#         scaled_distances = distances / (2 * self.kernel_lengthscale ** 2)
#         total_distance = scaled_distances.sum(dim=-1)  # Sum over features: (batch_size, num_candidates)
#         similarities = torch.exp(-total_distance)
#         return similarities  # Shape: (batch_size, num_candidates)

#     def forward(
#         self,
#         x_num: torch.Tensor,
#         x_cat: Optional[torch.Tensor],
#         y: Optional[torch.Tensor],
#         candidate_x_num: torch.Tensor,
#         candidate_x_cat: Optional[torch.Tensor],
#         candidate_y: torch.Tensor
#     ) -> torch.Tensor:
#         # Initialize accumulators
#         logits_numerator = torch.zeros(x_num.size(0), self.d_out, device=x_num.device)
#         sum_similarities = torch.zeros(x_num.size(0), device=x_num.device)
#         batch_size = 5000  # Adjust as needed

#         for idx in range(0, candidate_y.shape[0], batch_size):
#             batch_candidate_y = candidate_y[idx:idx+batch_size]
#             batch_candidate_x_num = candidate_x_num[idx:idx+batch_size]
#             batch_candidate_x_cat = candidate_x_cat[idx:idx+batch_size] if candidate_x_cat is not None else None

#             # Compute similarities between numerical features
#             num_similarities = self.kernel_function(x_num, batch_candidate_x_num)  # (batch_size, batch_size)

#             # Compute similarities between categorical features
#             if x_cat is not None and batch_candidate_x_cat is not None:
#                 # Compute similarities (e.g., using dot product)
#                 cat_similarities = (x_cat @ batch_candidate_x_cat.T)  # Shape: (batch_size, batch_size)
#                 # Normalize similarities if necessary
#                 max_cat_similarity = x_cat.size(1)
#                 cat_similarities = cat_similarities / max_cat_similarity
#             else:
#                 cat_similarities = torch.ones_like(num_similarities)

#             # Combine similarities
#             total_similarities = num_similarities * cat_similarities
#             # Apply temperature scaling
#             similarities = total_similarities / self.temperature

#             # Ensure candidate_y is correctly shaped
#             if self.d_out > 1:
#                 batch_candidate_y = F.one_hot(batch_candidate_y, self.d_out).float()
#             elif batch_candidate_y.dim() == 1:
#                 batch_candidate_y = batch_candidate_y.unsqueeze(-1).float()

#             # Accumulate results
#             sum_similarities += similarities.sum(dim=1)  # (batch_size,)
#             logits_numerator += similarities @ batch_candidate_y  # (batch_size, d_out)

#         # Compute logits
#         logits = torch.log(logits_numerator + 1e-8) - torch.log(sum_similarities + 1e-8).unsqueeze(1)
#         return logits

# Version 2: Neural network-based kernel
# class KernelNCA(nn.Module):
#     def __init__(
#         self,
#         *,
#         d_in_num: int,
#         d_in_cat: int,
#         d_out: int,
#         temperature: float = 1.0,
#         sample_rate: float = 0.8,
#         kernel_hidden_dim: int = 128,
#         **kwargs,
#     ) -> None:
#         super().__init__()
#         self.d_in_num = d_in_num
#         self.d_in_cat = d_in_cat
#         self.d_out = d_out
#         self.temperature = temperature
#         self.sample_rate = sample_rate

#         # Neural network to compute kernel similarities
#         input_dim = 2 * (d_in_num + d_in_cat)  # Since we concatenate features of x1 and x2
#         self.kernel_net = nn.Sequential(
#             nn.Linear(input_dim, kernel_hidden_dim),
#             nn.ReLU(),
#             nn.Linear(kernel_hidden_dim, kernel_hidden_dim),
#             nn.ReLU(),
#             nn.Linear(kernel_hidden_dim, 1),
#             nn.Softplus(),  # Ensure output is positive
#         )

#     def kernel_function(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
#         # x1: (batch_size, feature_dim)
#         # x2: (num_candidates, feature_dim)
#         batch_size, feature_dim = x1.size()
#         num_candidates = x2.size(0)

#         # Compute pairwise combinations efficiently using broadcasting
#         x1_expanded = x1.unsqueeze(1).expand(batch_size, num_candidates, feature_dim)
#         x2_expanded = x2.unsqueeze(0).expand(batch_size, num_candidates, feature_dim)

#         # Concatenate features
#         pair_features = torch.cat([x1_expanded, x2_expanded], dim=-1)  # (batch_size, num_candidates, 2 * feature_dim)
#         pair_features = pair_features.view(-1, pair_features.size(-1))  # Flatten for processing

#         # Compute similarities
#         similarities = self.kernel_net(pair_features).view(batch_size, num_candidates)
#         return similarities  # Shape: (batch_size, num_candidates)

    # def forward(
    #     self,
    #     x_num: torch.Tensor,
    #     x_cat: Optional[torch.Tensor],
    #     y: Optional[torch.Tensor],
    #     candidate_x_num: torch.Tensor,
    #     candidate_x_cat: Optional[torch.Tensor],
    #     candidate_y: torch.Tensor
    # ) -> torch.Tensor:
    #     # Concatenate numerical and categorical features
    #     x = [x_num]
    #     candidate_x = [candidate_x_num]
    #     if x_cat is not None and candidate_x_cat is not None:
    #         x.append(x_cat)
    #         candidate_x.append(candidate_x_cat)

    #     x_combined = torch.cat(x, dim=1)  # (batch_size, feature_dim)
    #     candidate_x_combined = torch.cat(candidate_x, dim=1)  # (num_candidates, feature_dim)

    #     # Ensure candidate_y is correctly shaped
    #     if self.d_out > 1:
    #         candidate_y = F.one_hot(candidate_y, self.d_out).float()
    #     elif candidate_y.dim() == 1:
    #         candidate_y = candidate_y.unsqueeze(-1).float()

    #     # Initialize accumulators
    #     logits_numerator = torch.zeros(x_num.size(0), self.d_out, device=x_num.device)
    #     sum_similarities = torch.zeros(x_num.size(0), device=x_num.device)
    #     batch_size = 5000  # Adjust based on memory constraints

    #     for idx in range(0, candidate_y.shape[0], batch_size):
    #         batch_candidate_x_combined = candidate_x_combined[idx:idx+batch_size]
    #         batch_candidate_y = candidate_y[idx:idx+batch_size]

    #         # Compute similarities
    #         batch_similarities = self.kernel_function(x_combined, batch_candidate_x_combined) / self.temperature
            
    #         # Accumulate sum of similarities and logits numerator
    #         sum_similarities += batch_similarities.sum(dim=1)
    #         logits_numerator += batch_similarities @ batch_candidate_y

    #     # Compute logits
    #     logits = torch.log(logits_numerator + 1e-8) - torch.log(sum_similarities + 1e-8).unsqueeze(1)
    #     return logits

# Version 3 (neural, separate kernel for num and cat features)
# class KernelNCA(nn.Module):
#     def __init__(
#         self,
#         *,
#         d_in_num: int,
#         d_in_cat: int,
#         d_out: int,
#         temperature: float = 1.0,
#         sample_rate: float = 0.8,
#         num_kernel_hidden_dim: int = 64,
#         cat_kernel_hidden_dim: int = 64,
#         **kwargs,
#     ) -> None:
#         super().__init__()
#         self.d_in_num = d_in_num
#         self.d_in_cat = d_in_cat
#         self.d_out = d_out
#         self.temperature = temperature
#         self.sample_rate = sample_rate

#         # Neural network to compute kernel similarities for numerical features
#         num_input_dim = 2 * d_in_num  # Since we concatenate features of x1 and x2
#         self.num_kernel_net = nn.Sequential(
#             nn.Linear(num_input_dim, num_kernel_hidden_dim),
#             nn.ReLU(),
#             nn.Linear(num_kernel_hidden_dim, num_kernel_hidden_dim),
#             nn.ReLU(),
#             nn.Linear(num_kernel_hidden_dim, 1),
#             nn.Softplus(),  # Ensure output is positive
#         )

#         # Neural network to compute kernel similarities for categorical features
#         if d_in_cat > 0:
#             cat_input_dim = 2 * d_in_cat  # Since we concatenate features of x1 and x2
#             self.cat_kernel_net = nn.Sequential(
#                 nn.Linear(cat_input_dim, cat_kernel_hidden_dim),
#                 nn.ReLU(),
#                 nn.Linear(cat_kernel_hidden_dim, cat_kernel_hidden_dim),
#                 nn.ReLU(),
#                 nn.Linear(cat_kernel_hidden_dim, 1),
#                 nn.Softplus(),  # Ensure output is positive
#             )
#         else:
#             self.cat_kernel_net = None

#     def num_kernel_function(self, x1_num: torch.Tensor, x2_num: torch.Tensor) -> torch.Tensor:
#         # x1_num: (batch_size, num_features)
#         # x2_num: (batch_size_candidate, num_features)
#         batch_size, num_features = x1_num.size()
#         num_candidates = x2_num.size(0)

#         # Compute pairwise combinations efficiently using broadcasting
#         x1_expanded = x1_num.unsqueeze(1).expand(batch_size, num_candidates, num_features)
#         x2_expanded = x2_num.unsqueeze(0).expand(batch_size, num_candidates, num_features)

#         # Concatenate features
#         pair_features = torch.cat([x1_expanded, x2_expanded], dim=-1)  # (batch_size, num_candidates, 2 * num_features)
#         pair_features = pair_features.view(-1, pair_features.size(-1))  # Flatten for processing

#         # Compute similarities
#         similarities = self.num_kernel_net(pair_features).view(batch_size, num_candidates)
#         return similarities  # Shape: (batch_size, num_candidates)

#     def cat_kernel_function(self, x1_cat: torch.Tensor, x2_cat: torch.Tensor) -> torch.Tensor:
#         # x1_cat: (batch_size, cat_features)
#         # x2_cat: (batch_size_candidate, cat_features)
#         batch_size, cat_features = x1_cat.size()
#         num_candidates = x2_cat.size(0)

#         # Compute pairwise combinations efficiently using broadcasting
#         x1_expanded = x1_cat.unsqueeze(1).expand(batch_size, num_candidates, cat_features)
#         x2_expanded = x2_cat.unsqueeze(0).expand(batch_size, num_candidates, cat_features)

#         # Concatenate features
#         pair_features = torch.cat([x1_expanded, x2_expanded], dim=-1)  # (batch_size, num_candidates, 2 * cat_features)
#         pair_features = pair_features.view(-1, pair_features.size(-1))  # Flatten for processing

#         # Compute similarities
#         similarities = self.cat_kernel_net(pair_features).view(batch_size, num_candidates)
#         return similarities  # Shape: (batch_size, num_candidates)

#     def forward(
#         self,
#         x_num: torch.Tensor,
#         x_cat: Optional[torch.Tensor],
#         y: Optional[torch.Tensor],
#         candidate_x_num: torch.Tensor,
#         candidate_x_cat: Optional[torch.Tensor],
#         candidate_y: torch.Tensor
#     ) -> torch.Tensor:
#         # Ensure candidate_y is correctly shaped
#         if self.d_out > 1:
#             candidate_y = F.one_hot(candidate_y, self.d_out).float()
#         elif candidate_y.dim() == 1:
#             candidate_y = candidate_y.unsqueeze(-1).float()

#         # Initialize accumulators
#         logits_numerator = torch.zeros(x_num.size(0), self.d_out, device=x_num.device)
#         sum_similarities = torch.zeros(x_num.size(0), device=x_num.device)
#         batch_size = 5000  # Adjust based on memory constraints

#         for idx in range(0, candidate_y.shape[0], batch_size):
#             batch_candidate_x_num = candidate_x_num[idx:idx+batch_size]
#             batch_candidate_y = candidate_y[idx:idx+batch_size]

#             # Compute similarities for numerical features
#             num_similarities = self.num_kernel_function(x_num, batch_candidate_x_num)  # (batch_size, batch_candidate_size)

#             # Compute similarities for categorical features
#             if self.cat_kernel_net is not None and x_cat is not None and candidate_x_cat is not None:
#                 batch_candidate_x_cat = candidate_x_cat[idx:idx+batch_size]

#                 cat_similarities = self.cat_kernel_function(x_cat, batch_candidate_x_cat)  # (batch_size, batch_candidate_size)
#             else:
#                 cat_similarities = torch.ones_like(num_similarities)

#             # Combine similarities
#             total_similarities = num_similarities * cat_similarities

#             # Apply temperature scaling
#             similarities = total_similarities / self.temperature

#             # Accumulate sum of similarities and logits numerator
#             sum_similarities += similarities.sum(dim=1)
#             logits_numerator += similarities @ batch_candidate_y

#         # Compute logits
#         logits = torch.log(logits_numerator + 1e-8) - torch.log(sum_similarities + 1e-8).unsqueeze(1)
#         return logits

# Version 4 (neural with residual)
# class KernelNCA(nn.Module):
#     def __init__(
#         self,
#         *,
#         d_in_num: int,
#         d_in_cat: int,
#         d_out: int,
#         embedding_dim: int = 256,
#         temperature: float = 1.0,
#         sample_rate: float = 0.8,
#         kernel_hidden_dim: int = 512,
#         **kwargs,
#     ) -> None:
#         super().__init__()
#         self.d_in_num = d_in_num
#         self.d_in_cat = d_in_cat
#         self.d_out = d_out
#         self.embedding_dim = embedding_dim
#         self.temperature = temperature
#         self.sample_rate = sample_rate

#         # Linear layers to encode numerical and categorical features
#         self.num_encoder = nn.Sequential(
#             nn.Linear(d_in_num, embedding_dim),
#             nn.ReLU(),
#             nn.Linear(embedding_dim, embedding_dim),
#         )

#         if d_in_cat > 0:
#             self.cat_encoder = nn.Sequential(
#                 nn.Linear(d_in_cat, embedding_dim),
#                 nn.ReLU(),
#                 nn.Linear(embedding_dim, embedding_dim),
#             )
#         else:
#             self.cat_encoder = None

#         # Neural kernel with residual connections
#         self.kernel_net = nn.Sequential(
#             nn.Linear(embedding_dim, kernel_hidden_dim),
#             nn.ReLU(),
#             nn.Linear(kernel_hidden_dim, kernel_hidden_dim),
#             nn.ReLU(),
#             nn.Linear(kernel_hidden_dim, embedding_dim),  # Output dimension matches embedding_dim for residual connection
#         )

#         # Final layer to compute similarity score
#         self.similarity_layer = nn.Linear(embedding_dim, 1)
#         self.activation = nn.Softplus()  # Ensure output is positive

#     def kernel_function(self, x_embed: torch.Tensor, y_embed: torch.Tensor) -> torch.Tensor:
#         # x_embed: (batch_size, embedding_dim)
#         # y_embed: (num_candidates, embedding_dim)

#         # Compute pairwise differences
#         x_expanded = x_embed.unsqueeze(1)  # (batch_size, 1, embedding_dim)
#         y_expanded = y_embed.unsqueeze(0)  # (1, num_candidates, embedding_dim)
#         diff = x_expanded - y_expanded     # (batch_size, num_candidates, embedding_dim)

#         # Flatten for processing
#         diff_flat = diff.view(-1, self.embedding_dim)  # (batch_size * num_candidates, embedding_dim)

#         # Pass through kernel network with residual connections
#         kernel_output = self.kernel_net(diff_flat)  # (batch_size * num_candidates, embedding_dim)
#         kernel_output = kernel_output + diff_flat   # Residual connection

#         # Compute similarity score
#         similarity_scores = self.similarity_layer(kernel_output)  # (batch_size * num_candidates, 1)
#         similarities = self.activation(similarity_scores).view(diff.size(0), diff.size(1))  # (batch_size, num_candidates)

#         return similarities  # Shape: (batch_size, num_candidates)

#     def forward(
#         self,
#         x_num: torch.Tensor,
#         x_cat: Optional[torch.Tensor],
#         y: Optional[torch.Tensor],
#         candidate_x_num: torch.Tensor,
#         candidate_x_cat: Optional[torch.Tensor],
#         candidate_y: torch.Tensor
#     ) -> torch.Tensor:
#         # Encode numerical features
#         x_num_encoded = self.num_encoder(x_num)  # (batch_size, embedding_dim)
#         candidate_x_num_encoded = self.num_encoder(candidate_x_num)  # (num_candidates, embedding_dim)

#         if self.cat_encoder is not None and x_cat is not None and candidate_x_cat is not None:
#             # Encode categorical features
#             x_cat_encoded = self.cat_encoder(x_cat)  # (batch_size, embedding_dim)
#             candidate_x_cat_encoded = self.cat_encoder(candidate_x_cat)  # (num_candidates, embedding_dim)

#             # Combine encoded numerical and categorical features
#             x_embed = x_num_encoded + x_cat_encoded  # (batch_size, embedding_dim)
#             candidate_x_embed = candidate_x_num_encoded + candidate_x_cat_encoded  # (num_candidates, embedding_dim)
#         else:
#             x_embed = x_num_encoded
#             candidate_x_embed = candidate_x_num_encoded

#         # Ensure candidate_y is correctly shaped
#         if self.d_out > 1:
#             candidate_y = F.one_hot(candidate_y, self.d_out).float()
#         elif candidate_y.dim() == 1:
#             candidate_y = candidate_y.unsqueeze(-1).float()

#         # Initialize accumulators
#         logits_numerator = torch.zeros(x_num.size(0), self.d_out, device=x_num.device)
#         sum_similarities = torch.zeros(x_num.size(0), device=x_num.device)
#         batch_size = 5000  # Adjust based on memory constraints

#         for idx in range(0, candidate_y.shape[0], batch_size):
#             batch_candidate_x_embed = candidate_x_embed[idx:idx+batch_size]
#             batch_candidate_y = candidate_y[idx:idx+batch_size]

#             # Compute similarities
#             batch_similarities = self.kernel_function(x_embed, batch_candidate_x_embed) / self.temperature

#             # Accumulate sum of similarities and logits numerator
#             sum_similarities += batch_similarities.sum(dim=1)
#             logits_numerator += batch_similarities @ batch_candidate_y

#         # Compute logits
#         logits = torch.log(logits_numerator + 1e-8) - torch.log(sum_similarities + 1e-8).unsqueeze(1)
#         return logits

# Version 5 (neural kernel with multiple residual blocks + batchnorm)
# class ResidualBlock(nn.Module):
#     def __init__(self, in_dim, hidden_dim, dropout_rate=0.1):
#         super().__init__()
#         self.linear1 = nn.Linear(in_dim, hidden_dim)
#         self.activation1 = nn.SiLU()
#         self.dropout1 = nn.Dropout(dropout_rate)
#         self.linear2 = nn.Linear(hidden_dim, in_dim)
#         self.activation2 = nn.SiLU()
#         self.dropout2 = nn.Dropout(dropout_rate)

#     def forward(self, x):
#         residual = x
#         out = self.linear1(x)
#         out = self.activation1(out)
#         out = self.dropout1(out)
#         out = self.linear2(out)
#         out = self.activation2(out)
#         out = self.dropout2(out)
#         out += residual  # Residual connection
#         return out

# class KernelNCA(nn.Module):
#     def __init__(
#         self,
#         d_in_num: int,
#         d_in_cat: int,
#         d_out: int,
#         embedding_dim: int = 256,
#         kernel_hidden_dim: int = 512,
#         temperature: float = 1.0,
#         sample_rate: float = 0.8,
#         num_residual_blocks: int = 1,  # Number of residual blocks
#         **kwargs,
#     ) -> None:
#         super().__init__()
#         self.d_in_num = d_in_num
#         self.d_in_cat = d_in_cat
#         self.d_out = d_out
#         self.embedding_dim = embedding_dim
#         self.temperature = temperature
#         self.sample_rate = sample_rate

#         # Linear layers to encode numerical and categorical features
#         self.num_encoder = nn.Sequential(
#             nn.Linear(d_in_num, embedding_dim),
#             nn.SiLU(),
#             nn.Linear(embedding_dim, embedding_dim),
#         )

#         if d_in_cat > 0:
#             self.cat_encoder = nn.Sequential(
#                 nn.Linear(d_in_cat, embedding_dim),
#                 nn.SiLU(),
#                 nn.Linear(embedding_dim, embedding_dim),
#             )
#         else:
#             self.cat_encoder = None

#         # Kernel network with multiple residual blocks
#         layers = []
#         for _ in range(num_residual_blocks):
#             layers.append(ResidualBlock(embedding_dim, kernel_hidden_dim))
#         self.kernel_net = nn.Sequential(*layers)

#         # Final layer to compute similarity score
#         self.similarity_layer = nn.Linear(embedding_dim, 1)
#         self.activation = nn.Softplus()  # Ensure output is positive

#     def kernel_function(self, x_embed: torch.Tensor, y_embed: torch.Tensor) -> torch.Tensor:
#         # x_embed: (batch_size, embedding_dim)
#         # y_embed: (num_candidates, embedding_dim)

#         # Compute pairwise differences
#         x_expanded = x_embed.unsqueeze(1)  # (batch_size, 1, embedding_dim)
#         y_expanded = y_embed.unsqueeze(0)  # (1, num_candidates, embedding_dim)
#         diff = x_expanded - y_expanded     # (batch_size, num_candidates, embedding_dim)

#         # Flatten for processing
#         diff_flat = diff.view(-1, self.embedding_dim)  # (batch_size * num_candidates, embedding_dim)

#         # Pass through kernel network with residual blocks
#         kernel_output = self.kernel_net(diff_flat)  # (batch_size * num_candidates, embedding_dim)

#         # Compute similarity score
#         similarity_scores = self.similarity_layer(kernel_output)  # (batch_size * num_candidates, 1)
#         similarities = self.activation(similarity_scores).view(diff.size(0), diff.size(1))  # (batch_size, num_candidates)

#         return similarities  # Shape: (batch_size, num_candidates)

#     def forward(
#         self,
#         x_num: torch.Tensor,
#         x_cat: Optional[torch.Tensor],
#         y: Optional[torch.Tensor],
#         candidate_x_num: torch.Tensor,
#         candidate_x_cat: Optional[torch.Tensor],
#         candidate_y: torch.Tensor
#     ) -> torch.Tensor:
#         # Encode numerical features
#         x_num_encoded = self.num_encoder(x_num)  # (batch_size, embedding_dim)
#         candidate_x_num_encoded = self.num_encoder(candidate_x_num)  # (num_candidates, embedding_dim)

#         if self.cat_encoder is not None and x_cat is not None and candidate_x_cat is not None:
#             # Encode categorical features
#             x_cat_encoded = self.cat_encoder(x_cat)  # (batch_size, embedding_dim)
#             candidate_x_cat_encoded = self.cat_encoder(candidate_x_cat)  # (num_candidates, embedding_dim)

#             # Combine encoded numerical and categorical features
#             x_embed = x_num_encoded + x_cat_encoded 
#             candidate_x_embed = candidate_x_num_encoded + candidate_x_cat_encoded
#         else:
#             x_embed = x_num_encoded
#             candidate_x_embed = candidate_x_num_encoded

#         # Ensure candidate_y is correctly shaped
#         if self.d_out > 1:
#             candidate_y = F.one_hot(candidate_y, self.d_out).float()
#         elif candidate_y.dim() == 1:
#             candidate_y = candidate_y.unsqueeze(-1).float()

#         # Initialize accumulators
#         logits_numerator = torch.zeros(x_num.size(0), self.d_out, device=x_num.device)
#         sum_similarities = torch.zeros(x_num.size(0), device=x_num.device)
#         batch_size = 5000  # Adjust based on memory constraints

#         for idx in range(0, candidate_y.shape[0], batch_size):
#             batch_candidate_x_embed = candidate_x_embed[idx:idx+batch_size]
#             batch_candidate_y = candidate_y[idx:idx+batch_size]

#             # Compute similarities
#             batch_similarities = self.kernel_function(x_embed, batch_candidate_x_embed) / self.temperature

#             # Accumulate sum of similarities and logits numerator
#             sum_similarities += batch_similarities.sum(dim=1)
#             logits_numerator += batch_similarities @ batch_candidate_y

#         # Compute logits
#         logits = torch.log(logits_numerator + 1e-8) - torch.log(sum_similarities + 1e-8).unsqueeze(1)
#         return logits

# Version 6: dot product similarity
# class KernelNCA(nn.Module):
#     def __init__(
#         self,
#         d_in_num: int,
#         d_in_cat: int,
#         d_out: int,
#         embedding_dim: int = 256,
#         kernel_hidden_dim: int = 512,
#         temperature: float = 1.0,
#         sample_rate: float = 0.8,
#         num_residual_blocks: int = 1,  # Number of residual blocks
#         **kwargs,
#     ) -> None:
#         super().__init__()
#         self.d_in_num = d_in_num
#         self.d_in_cat = d_in_cat
#         self.d_out = d_out
#         self.embedding_dim = embedding_dim
#         self.temperature = temperature
#         self.sample_rate = sample_rate

#         # Linear layers to encode numerical and categorical features
#         self.num_encoder = nn.Sequential(
#             nn.Linear(d_in_num, embedding_dim),
#             nn.SiLU(),
#             nn.Linear(embedding_dim, embedding_dim),
#         )

#         if d_in_cat > 0:
#             self.cat_encoder = nn.Sequential(
#                 nn.Linear(d_in_cat, embedding_dim),
#                 nn.SiLU(),
#                 nn.Linear(embedding_dim, embedding_dim),
#             )
#             self.kernel_weight_matrix = nn.Parameter(torch.randn(embedding_dim * 2, embedding_dim * 2))
#         else:
#             self.cat_encoder = None
#             self.kernel_weight_matrix = nn.Parameter(torch.randn(embedding_dim, embedding_dim))

        
#         self.activation = nn.Softplus()

#     def kernel_function(self, x_embed: torch.Tensor, y_embed: torch.Tensor) -> torch.Tensor:
#         # x_embed: (batch_size, embedding_dim)
#         # y_embed: (num_candidates, embedding_dim)
        
#         # Learnable weight matrix
#         W = self.kernel_weight_matrix  # Shape: (embedding_dim, embedding_dim)

#         # Compute pairwise similarities
#         x_proj = x_embed @ W  # Shape: (batch_size, embedding_dim)
#         similarities = x_proj @ y_embed.T  # Shape: (batch_size, num_candidates)

#         # Apply activation function to ensure positivity
#         similarities = self.activation(similarities / self.temperature)
#         return similarities

#     def forward(
#         self,
#         x_num: torch.Tensor,
#         x_cat: Optional[torch.Tensor],
#         y: Optional[torch.Tensor],
#         candidate_x_num: torch.Tensor,
#         candidate_x_cat: Optional[torch.Tensor],
#         candidate_y: torch.Tensor
#     ) -> torch.Tensor:
#         # Encode numerical features
#         x_num_encoded = self.num_encoder(x_num)  # (batch_size, embedding_dim)
#         candidate_x_num_encoded = self.num_encoder(candidate_x_num)  # (num_candidates, embedding_dim)

#         if self.cat_encoder is not None and x_cat is not None and candidate_x_cat is not None:
#             # Encode categorical features
#             x_cat_encoded = self.cat_encoder(x_cat)  # (batch_size, embedding_dim)
#             candidate_x_cat_encoded = self.cat_encoder(candidate_x_cat)  # (num_candidates, embedding_dim)

#             # Combine encoded numerical and categorical features
#             x_embed = torch.cat([x_num_encoded, x_cat_encoded], dim=1)
#             candidate_x_embed = torch.cat([candidate_x_num_encoded, candidate_x_cat_encoded], dim=1)
#         else:
#             x_embed = x_num_encoded
#             candidate_x_embed = candidate_x_num_encoded

#         # Ensure candidate_y is correctly shaped
#         if self.d_out > 1:
#             candidate_y = F.one_hot(candidate_y, self.d_out).float()
#         elif candidate_y.dim() == 1:
#             candidate_y = candidate_y.unsqueeze(-1).float()

#         # Initialize accumulators
#         logits_numerator = torch.zeros(x_num.size(0), self.d_out, device=x_num.device)
#         sum_similarities = torch.zeros(x_num.size(0), device=x_num.device)
#         batch_size = 5000  # Adjust based on memory constraints

#         for idx in range(0, candidate_y.shape[0], batch_size):
#             batch_candidate_x_embed = candidate_x_embed[idx:idx+batch_size]
#             batch_candidate_y = candidate_y[idx:idx+batch_size]

#             # Compute similarities
#             batch_similarities = self.kernel_function(x_embed, batch_candidate_x_embed) / self.temperature

#             # Accumulate sum of similarities and logits numerator
#             sum_similarities += batch_similarities.sum(dim=1)
#             logits_numerator += batch_similarities @ batch_candidate_y

#         # Compute logits
#         logits = torch.log(logits_numerator + 1e-8) - torch.log(sum_similarities + 1e-8).unsqueeze(1)
#         return logits

# Version 7: random feature embedding
# class KernelNCA(nn.Module):
#     def __init__(
#         self,
#         d_in_num: int,
#         d_in_cat: int,
#         d_out: int,
#         n_frequencies: int,
#         frequency_scale: float,
#         temperature: float = 1.0,
#         **kwargs,
#     ) -> None:
#         super().__init__()
#         self.d_in_num = d_in_num
#         self.d_in_cat = d_in_cat
#         self.d_out = d_out
#         self.n_features = d_in_num + d_in_cat  # Total number of features
#         self.n_frequencies = n_frequencies
#         self.frequency_scale = frequency_scale
#         self.temperature = temperature

#         # Initialize frequencies and shifts
#         self.frequencies = Parameter(
#             torch.normal(0.0, frequency_scale, size=(self.n_features, n_frequencies))
#         )
#         self.shifts = Parameter(
#             torch.rand(size=(self.n_features, n_frequencies))
#         )
#         self.factor = Parameter(
#             torch.Tensor([np.sqrt(2.0 / n_frequencies)])
#         )
        
#     def forward(
#         self,
#         x_num: torch.Tensor,
#         x_cat: Optional[torch.Tensor],
#         y: Optional[torch.Tensor],
#         candidate_x_num: torch.Tensor,
#         candidate_x_cat: Optional[torch.Tensor],
#         candidate_y: torch.Tensor
#     ) -> torch.Tensor:
#         x_list = [x_num]
#         candidate_x_list = [candidate_x_num]
#         if x_cat is not None:
#             x_list.append(x_cat)
#         if candidate_x_cat is not None:
#             candidate_x_list.append(candidate_x_cat)

#         x = torch.cat(x_list, dim=1)  # Shape: (batch_size, n_features)
#         candidate_x = torch.cat(candidate_x_list, dim=1)  # Shape: (num_candidates, n_features)

#         # Encode using random features
#         x_encoded = self.factor * torch.cos(2 * torch.pi * self.frequencies[None] * x[..., None] + self.shifts[None])
#         candidate_x_encoded = self.factor * torch.cos(2 * torch.pi * self.frequencies[None] * candidate_x[..., None] + self.shifts[None])
#         x_encoded = x_encoded.flatten(1)
#         candidate_x_encoded = candidate_x_encoded.flatten(1)

#         # Ensure candidate_y is correctly shaped
#         if self.d_out > 1:
#             candidate_y = F.one_hot(candidate_y, self.d_out).float()
#         elif candidate_y.dim() == 1:
#             candidate_y = candidate_y.unsqueeze(-1).float()

#         # Initialize accumulators
#         batch_size = 5000  # Adjust based on memory constraints
#         logits = torch.zeros(x_encoded.size(0), self.d_out, device=x_encoded.device)
#         logsumexp = torch.zeros(x_encoded.size(0), device=x_encoded.device)
#         for idx in range(0, candidate_y.shape[0], batch_size):
#             batch_candidate_x_encoded = candidate_x_encoded[idx:(idx+batch_size)]
#             batch_candidate_y = candidate_y[idx:(idx+batch_size)]

#             # Compute similarities
#             distances = torch.cdist(x_encoded, batch_candidate_x_encoded, p=2) / self.temperature
#             exp_neg_distances = torch.exp(-distances)  # Shape: (batch_size, batch_size_candidate)

#             # Accumulate logsumexp and logits
#             logsumexp += torch.logsumexp(-distances, dim=1)  # Shape: (batch_size,)
#             logits += exp_neg_distances @ batch_candidate_y  # Shape: (batch_size, d_out)

#         logits = torch.log(logits + 1e-8) - logsumexp.unsqueeze(1)
#         return logits

# Version 8: binning
class KernelNCA(nn.Module):
    def __init__(
        self,
        d_in_num: int,
        d_in_cat: int,
        d_out: int,
        n_bins: int = 50,
        beta: float = 10.0,
        delta_scale: float = 60.0,
        temperature: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.d_in_num = d_in_num
        self.d_in_cat = d_in_cat
        self.d_out = d_out
        self.n_features = d_in_num + d_in_cat  # Total number of features
        self.n_bins = n_bins
        self.beta = beta
        self.temperature = temperature
        
        # Initialize learnable grid parameters (delta) and fixed random shifts (u)
        self.delta = Parameter(
            torch.rand(d_in_num, n_bins) * delta_scale  # Learnable delta for each feature
        )
        self.register_buffer("u", torch.rand(d_in_num, n_bins))
        self.factor = Parameter(
            torch.Tensor([np.sqrt(1.0 / n_bins)])
        )

    def clamp_delta(self, min_val=30, max_val=100.0):
        with torch.no_grad():
            self.delta.clamp_(min=min_val, max=max_val)

    def feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the randomized feature map z(x) for input x.
        """
        # Scale u
        scaled_u = self.u * self.delta

        # Compute bin indices
        bin_indices = torch.ceil((x.unsqueeze(-1) - scaled_u[None]) / self.delta[None])

        return bin_indices.flatten(1)

    def forward(
        self,
        x_num: torch.Tensor,
        x_cat: Optional[torch.Tensor],
        y: Optional[torch.Tensor],
        candidate_x_num: torch.Tensor,
        candidate_x_cat: Optional[torch.Tensor],
        candidate_y: torch.Tensor
    ) -> torch.Tensor:
        # Encode numerical 
        x_encoded = self.feature_map(x_num)
        candidate_x_encoded = self.feature_map(candidate_x_num)

        # Concatenate numerical and categorical features
        if x_cat is not None:
            x_encoded = torch.cat([x_encoded, x_cat], dim=1)
        if candidate_x_cat is not None:
            candidate_x_encoded = torch.cat([candidate_x_encoded, candidate_x_cat], dim=1)

        # Ensure candidate_y is correctly shaped
        if self.d_out > 1:
            candidate_y = F.one_hot(candidate_y, self.d_out).float()
        elif candidate_y.dim() == 1:
            candidate_y = candidate_y.unsqueeze(-1).float()

        # Initialize accumulators
        batch_size = 1000  # Adjust based on memory constraints
        logits = torch.zeros(x_encoded.size(0), self.d_out, device=x_encoded.device)
        logsumexp = torch.zeros(x_encoded.size(0), device=x_encoded.device)
        for idx in range(0, candidate_y.shape[0], batch_size):
            batch_candidate_x_encoded = candidate_x_encoded[idx:idx+batch_size]
            batch_candidate_y = candidate_y[idx:idx+batch_size]

            # Compute similarities
            distances = torch.cdist(x_encoded, batch_candidate_x_encoded, p=1) / self.temperature
            exp_neg_distances = torch.exp(-distances)  # Shape: (batch_size, z)

            # Accumulate logsumexp and logits
            logsumexp += torch.logsumexp(-distances, dim=1)  # Shape: (batch_size,)
            logits += exp_neg_distances @ batch_candidate_y  # Shape: (batch_size, d_out)

        logits = torch.log(logits + 1e-8) - logsumexp.unsqueeze(1)
        return logits

