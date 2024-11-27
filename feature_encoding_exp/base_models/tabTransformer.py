from typing import Optional, Callable   
import time

import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange, repeat
from torch.utils.data import TensorDataset, DataLoader
from torch import einsum

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# classes
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# attention
class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x, **kwargs):
        return self.net(x)

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 16,
        dropout = 0.
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = sim.softmax(dim = -1)
        dropped_attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', dropped_attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        return self.to_out(out), attn

# transformer
class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        attn_dropout,
        ff_dropout
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout)),
                PreNorm(dim, FeedForward(dim, dropout = ff_dropout)),
            ]))

    def forward(self, x, return_attn = False):
        post_softmax_attns = []

        for attn, ff in self.layers:
            attn_out, post_softmax_attn = attn(x)
            post_softmax_attns.append(post_softmax_attn)

            x = x + attn_out
            x = ff(x) + x
        
        return x
    
# mlp
class MLP(nn.Module):
    def __init__(self, dims, act = None):
        super().__init__()
        dims_pairs = list(zip(dims[:-1], dims[1:]))
        layers = []
        for ind, (dim_in, dim_out) in enumerate(dims_pairs):
            is_last = ind >= (len(dims_pairs) - 1)
            linear = nn.Linear(dim_in, dim_out)
            layers.append(linear)

            if is_last:
                continue

            act = default(act, nn.ReLU())
            layers.append(act)

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class TabTransformer(nn.Module):
    def __init__(
            self,
            *, 
            d_in_cat, 
            d_in_num, 
            d_out=1,
            dim, 
            depth, 
            heads, 
            dim_head=16, 
            mlp_hidden_mults=(4, 2), 
            mlp_act = None,
            num_special_tokens=2, 
            continuous_mean_std=None,
            attn_dropout=0., 
            ff_dropout=0., 
            use_shared_categ_embed=True, 
            shared_categ_dim_divisor=8,
            num_encoder: Optional[nn.Module] = None,
        ) -> None:
        super().__init__()

        #----------------------------------------------
        # Define the numerical encoder
        self.num_encoder = num_encoder
        d_in_num = d_in_num if num_encoder is None else num_encoder.d_out
        #----------------------------------------------

        self.num_categories = 0
        self.num_unique_categories = 0
        if d_in_cat is not None:
            assert all(map(lambda n: n > 0, d_in_cat)), 'number of each category must be positive'
            assert len(d_in_cat) + d_in_num > 0, 'input shape must not be null'

            # categories related calculations
            self.num_categories = len(d_in_cat)
            self.num_unique_categories = sum(d_in_cat)

        # create category embeddings table
        self.num_special_tokens = num_special_tokens
        total_tokens = self.num_unique_categories + num_special_tokens
        if d_in_cat is not None:
            shared_embed_dim = 0 if not use_shared_categ_embed else int(dim // shared_categ_dim_divisor)

            self.category_embed = nn.Embedding(total_tokens, dim - shared_embed_dim)

            # take care of shared category embed
            self.use_shared_categ_embed = use_shared_categ_embed

            if use_shared_categ_embed:
                self.shared_category_embed = nn.Parameter(torch.zeros(self.num_categories, shared_embed_dim))
                nn.init.normal_(self.shared_category_embed, std = 0.02)

            # for automatically offsetting unique category ids to the correct position in the categories embedding table
            if self.num_unique_categories > 0:
                categories_offset = F.pad(torch.tensor(list(d_in_cat)), (1, 0), value = num_special_tokens)
                categories_offset = categories_offset.cumsum(dim = -1)[:-1]
                self.register_buffer('categories_offset', categories_offset)

        # continuous
        self.d_in_num = d_in_num

        if self.d_in_num > 0:
            if exists(continuous_mean_std):
                assert continuous_mean_std.shape == (d_in_num, 2), f'continuous_mean_std must have a shape of ({d_in_num}, 2) where the last dimension contains the mean and variance respectively'
            self.register_buffer('continuous_mean_std', continuous_mean_std)

            self.norm = nn.LayerNorm(d_in_num)

        # transformer
        self.transformer = Transformer(
            dim = dim,
            depth = depth,
            heads = heads,
            dim_head = dim_head,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout
        )

        # mlp to logits
        input_size = (dim * self.num_categories) + d_in_num

        hidden_dimensions = [input_size * t for t in  mlp_hidden_mults]
        all_dimensions = [input_size, *hidden_dimensions, d_out]

        self.mlp = MLP(all_dimensions, act = mlp_act)

    def forward(self, x_cont,x_categ , return_attn = False):
        #----------------------------------------------
        # Transform numerical features
        if self.num_encoder is not None:
            x_cont = self.num_encoder(x_cont)
        #----------------------------------------------

        xs = []
        if x_categ is not None:
            assert x_categ.shape[-1] == self.num_categories, f'you must pass in {self.num_categories} values for your categories input'

        if self.num_unique_categories > 0:
            x_categ = x_categ + self.categories_offset

            categ_embed = self.category_embed(x_categ)

            if self.use_shared_categ_embed:
                shared_categ_embed = repeat(self.shared_category_embed, 'n d -> b n d', b = categ_embed.shape[0])
                categ_embed = torch.cat((categ_embed, shared_categ_embed), dim = -1)

            x = self.transformer(categ_embed, return_attn = True)

            flat_categ = rearrange(x, 'b ... -> b (...)')
            xs.append(flat_categ)
        if x_cont is not None:
            assert x_cont.shape[1] == self.d_in_num, f'you must pass in {self.d_in_num} values for your continuous input'

        if self.d_in_num > 0:
            if exists(self.continuous_mean_std):
                mean, std = self.continuous_mean_std.unbind(dim = -1)
                x_cont = (x_cont - mean) / std

            normed_cont = self.norm(x_cont)
            xs.append(normed_cont)

        x = torch.cat(xs, dim = -1)
        logits = self.mlp(x)
        return logits

    def fit(
            self, 
            X_num_train: torch.Tensor,
            X_cat_train: torch.Tensor,
            y_train: torch.Tensor,
            criterion: nn.Module,
            batch_size: int,
            epochs: int,
            learning_rate: float,
            weight_decay: float,
        ) -> None:
        self.train()

        # Determine the device 
        device = next(self.parameters()).device

        # Define the dataset and dataloader
        train_dataset = TensorDataset(X_num_train, X_cat_train, y_train)
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
            total = 0

            for itr, (X_num_batch, X_cat_batch, y_batch) in enumerate(train_loader):
                X_num_batch = X_num_batch.to(device)
                X_cat_batch = X_cat_batch.to(device)
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
                if itr % 50 == 0:
                    print(f'Iteration [{itr}/{len(train_loader)}] | Loss: {loss.item():.4f}')

            epoch_loss = epoch_loss / total
            epoch_time = time.time() - start_time

            print(f'Epoch [{epoch+1}/{epochs}] | Loss: {epoch_loss:.4f} | Time: {epoch_time:.2f}s')

    def evaluate(
        self,
        X_num_test: torch.Tensor,
        X_cat_test: torch.Tensor,
        y_test: torch.Tensor,
        criterion: Callable[[torch.Tensor, torch.Tensor], float],
        batch_size: int,
    ) -> None:
        self.eval()

        # Determine the device
        device = next(self.parameters()).device

        # Define the dataset and dataloader
        test_dataset = TensorDataset(X_num_test, X_cat_test, y_test)
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
        )

        # Evaluate the model
        total_metric = 0.0
        total_samples = 0
        with torch.no_grad():
            for X_num_batch, X_cat_batch, y_batch in test_loader:
                X_num_batch = X_num_batch.to(device)
                X_cat_batch = X_cat_batch.to(device)
                y_batch = y_batch.to(device)

                # Forward pass
                logits = self(X_num_batch, X_cat_batch)

                # Compute metric using the provided criterion
                metric = criterion(logits, y_batch)
                total_metric += metric * y_batch.size(0)
                total_samples += y_batch.size(0)

        average_metric = total_metric / total_samples
        print(f'Evaluation Metric: {average_metric:.4f}')
