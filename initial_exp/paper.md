# Efficient and learnable feature encoding for tabular deep learning
---

## Setup
In tabular data, we have numerical features $\mathbf{x} \in \mathbb{R}^d$ and categorical features $\mathbf{c} \in \mathbb{N}^c$. 

## Goal
We want to learn an effective encoding function for numerical features, $z: \mathbb{R}^d \rightarrow \mathbb{R}^k$, and also for categorical features, $z: \mathbb{N}^c \rightarrow \mathbb{R}^k$.

---
## Prior works

### Approach [1] for numerical features: learnable Fourier features  
In [1], they propose to encode the numerical features, using learnable Fourier features. Namely, they have the following algorithm:
```python
class PLREmbeddings(nn.Module):
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
        self.linear = nn.Linear(2 * n_frequencies, d_embedding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = 2 * torch.pi * self.frequencies * x.unsqueeze(-1)
        x = torch.cat([torch.cos(x), torch.sin(x)], -1)
        x = self.linear(x)
        x = self.activation(x)
        return x
```

From [2], we know that the inner product of two random Fourier features approximates the kernel function:
$$
\begin{aligned}
z_p &= [\cos(\mathbf{w}_p^T \mathbf{x}), \sin(\mathbf{w}_p^T \mathbf{x})] \\
\mathbf{z} &= \frac{1}{\sqrt{P}}[z_1, z_2, \ldots, z_P] \\
\langle z_p, z_q \rangle \approx k(x, y), \\
\end{aligned}
$$
where the kernel $k(,)$ depends on the choice of the distribuition $\mathbf{w}_p \sim p_{\mathbf{w}}$.

A **few notable differences between implementation of [1] from [2]**:
1. [1] uses learnable frequencies, while [2] uses random frequencies.
2. [1] encodes each feature separately, while [2] encodes all features together. This is most clearly seen from the line `x = 2 * torch.pi * self.frequencies * x.unsqueeze(-1)`. Essentially, this means that a separate kernel is learned for each feature. Namely, the inner product $z(x)^\top z(y) \approx \sum_{i=1}^d k_i(x_i, y_i)$. 

**Main takeaway:** In general, [2] gives a lot of intuition of why [1] works. Essentially, once we encode the features, we are operating in the kernel space. Crucially, this motivates our subseqeuent study of exploring different kernels.


---

## Explorations

### LLM for categorical features
We can concatenate the categorical columns and values into a sentence, which we can then encode via LLM. Here is my starter code for this, using frozen LLM:
```python
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def llm_encoder(X, categorical_columns, batch_size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step 1: Prepare text embeddings
    embeddings = []
    for idx in range(X.shape[0]):
        string = ''
        for column in categorical_columns:
            string += f"{column}: {X[column][idx]}. "
        embeddings.append(string)

    # Load tokenizer and model, move model to device
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('sentence-transformers/paraphrase-MiniLM-L6-v2').to(device)

    # Step 2: Process embeddings in batches
    batch_embeddings = []
    with torch.no_grad():
        for i in range(0, len(embeddings), batch_size):
            batch_texts = embeddings[i:i+batch_size]
            encoded_input = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt').to(device)
            model_output = model(**encoded_input)
            batch_embeddings.append(mean_pooling(model_output, encoded_input['attention_mask']).cpu())
            print(f'Processed {i+batch_size}/{len(embeddings)} embeddings')
    
    # Concatenate all batch embeddings
    embeddings_tensor = torch.cat(batch_embeddings, dim=0)
    embeddings_df = pd.DataFrame(embeddings_tensor.numpy(), columns=[f'embedding_{i}' for i in range(embeddings_tensor.shape[1])])

    # Clean up memory
    del encoded_input, model_output, batch_embeddings, embeddings_tensor, tokenizer, model
    torch.cuda.empty_cache()

    # Step 3: Concatenate embeddings with the original DataFrame and drop categorical columns
    X = pd.concat([X.reset_index(drop=True), embeddings_df], axis=1)
    X = X.drop(columns=categorical_columns)
    
    return X
```

### Other types of kernels
In [2], the authors note that the inner product of **random Fourier features can approximate only certain the kernel function**. For other families of kernel functions, they propose the **random binning features** (see Algorithm 2 on p. 5).


I have started implementing the random binning features here:
```python
class BinningFeat(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_bins: int = 100,
        delta_scale: float = 60.0
        **kwargs,
    ) -> None:
        super().__init__()
        self.n_bins = n_bins
        self.n_features = n_features
        self.delta_scale = delta_scale
        
        # Initialize learnable grid parameters (delta) and fixed random shifts (u)
        self.delta = Parameter(
            torch.rand(n_features, n_bins) * delta_scale  # Learnable delta for each feature
        )
        self.register_buffer("u", torch.rand(d_in_num, n_bins))
        self.factor = Parameter(
            torch.Tensor([np.sqrt(1.0 / n_bins)])
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the randomized feature map z(x) for input x.
        """
        # Scale u
        scaled_u = self.u * self.delta

        # Compute bin indices: floor((x - u) / delta)
        bin_indices = torch.floor((x.unsqueeze(-1) - scaled_u[None]) / self.delta[None])
        bin_indices = self.factor * bin_indices.flatten(1)  # Shape: (batch_size, n_features * n_bins)

        # One hot encode bin indices
        bins = F.one_hot(bin_indices.to(torch.int64)).float().flatten(1)
        return bins
```

However, I am facing two main issues right now:
1. *Non-differentiability*: one-hot encoding is not differentiable. 
2. *Memory issues*: the one-hot encoding is very memory intensive.

I think we can just use the bin indices directly, without one-hot encoding (maybe?).


---
References:
1. [Efficient and learnable feature encoding for tabular deep learning](https://arxiv.org/pdf/2203.05556)  
  - Summary: here they propose learnable Fourier features for tabular data.
2. [Random Features for Large-Scale Kernel Machines](https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf)    
  - Summary: here is the first paper that introduced Fourier features.
3. [Introduction to Random Fourier Features](https://gregorygundersen.com/blog/2019/12/23/random-fourier-features/)  
  - Summary: here is a blog post that explains random Fourier features.