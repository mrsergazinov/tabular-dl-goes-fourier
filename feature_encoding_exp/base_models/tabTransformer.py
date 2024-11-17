import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange, repeat

class TabTransformerModel(nn.Module):
    def __init__(
            self,
            *, 
            categories, 
            num_continuous, 
            dim, 
            depth, 
            heads, 
            dim_head=16, 
            dim_out=1,
            mlp_hidden_mults=(4, 2), 
            num_special_tokens=2, 
            continuous_mean_std=None,
            attn_dropout=0., 
            ff_dropout=0., 
            use_shared_categ_embed=True, 
            shared_categ_dim_divisor=8.
        ):
        super().__init__()

        self.num_categories = len(categories) if categories else 0
        self.num_continuous = num_continuous
        self.num_unique_categories = sum(categories) if categories else 0

        # Category embedding
        shared_embed_dim = dim // shared_categ_dim_divisor if use_shared_categ_embed else 0
        total_tokens = self.num_unique_categories + num_special_tokens
        self.category_embed = nn.Embedding(total_tokens, dim - shared_embed_dim) if categories else None
        self.shared_category_embed = nn.Parameter(torch.zeros(self.num_categories, shared_embed_dim)) if use_shared_categ_embed else None
        if self.shared_category_embed is not None:
            nn.init.normal_(self.shared_category_embed, std=0.02)

        self.categories_offset = torch.cat([torch.tensor([num_special_tokens]), torch.cumsum(torch.tensor(categories), 0)])[:-1] if categories else None

        # Continuous feature normalization
        self.continuous_mean_std = continuous_mean_std
        self.norm = nn.LayerNorm(num_continuous) if num_continuous > 0 else None

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim_head * heads,
            dropout=attn_dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # MLP to logits
        input_size = dim * self.num_categories + num_continuous
        all_dims = [input_size] + [input_size * m for m in mlp_hidden_mults] + [dim_out]
        self.mlp = nn.Sequential(*[
            nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU() if i < len(all_dims) - 2 else nn.Identity())
            for i, (in_dim, out_dim) in enumerate(zip(all_dims[:-1], all_dims[1:]))
        ])

    def forward(self, x_cont, x_categ=None):
        xs = []

        # Categorical input processing
        if x_categ is not None:
            x_categ += self.categories_offset
            categ_embed = self.category_embed(x_categ)
            if self.shared_category_embed is not None:
                shared_embed = repeat(self.shared_category_embed, 'n d -> b n d', b=categ_embed.size(0))
                categ_embed = torch.cat((categ_embed, shared_embed), dim=-1)
            x_categ = rearrange(self.transformer(categ_embed), 'b n d -> b (n d)')
            xs.append(x_categ)

        # Continuous input processing
        if x_cont is not None:
            if self.continuous_mean_std is not None:
                mean, std = self.continuous_mean_std.unbind(dim=-1)
                x_cont = (x_cont - mean) / std
            xs.append(self.norm(x_cont) if self.norm else x_cont)

        # Concatenate and pass through MLP
        return self.mlp(torch.cat(xs, dim=-1))
