import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModel

# tabular_llama_model.py


class TabTransformer(nn.Module):
    """
    A simplified version of TabTransformer for handling tabular data.
    """
    def __init__(self, input_dim, output_dim, num_heads=8, num_layers=6, embed_dropout=0.1, attn_dropout=0.1):
        super(TabTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, output_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=output_dim, nhead=num_heads, dropout=attn_dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(embed_dropout)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        x = x.unsqueeze(1)  # Add sequence dimension
        x = self.transformer(x)
        x = x.squeeze(1)    # Remove sequence dimension
        return x

class HuggingFaceLLaMA(nn.Module):
    """
    LLaMA model loaded using Hugging Face's transformers library with parameters frozen.
    Allows selection of specific layers and optional positional embeddings.
    """
    def __init__(self, model_name, start_layer=0, end_layer=1, token=None, use_positional_embeddings=True):
        super(HuggingFaceLLaMA, self).__init__()

        # Load the model configuration with token
        config = AutoConfig.from_pretrained(model_name, use_auth_token=token)

        # Load the pre-trained model without the language modeling head
        self.model = AutoModel.from_pretrained(
            model_name,
            config=config,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_auth_token=token
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)

        # Add a new pad_token
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.model.resize_token_embeddings(len(self.tokenizer), pad_to_multiple_of=64)

        # Freeze LLaMA parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Extract specified transformer blocks (layers)
        num_layers = len(self.model.layers)
        end_layer = min(end_layer, num_layers)

        # Get the selected layers
        self.transformer_blocks = self.model.layers[start_layer:end_layer]

        # Include the final layer norm
        self.norm = self.model.norm

        # Store flags
        self.use_positional_embeddings = use_positional_embeddings

    def forward(self, inputs_embeds):
        """
        Forward pass through the selected transformer blocks with optional positional embeddings.

        Args:
            inputs_embeds (torch.Tensor): Input embeddings of shape [batch_size, seq_length, hidden_size]

        Returns:
            torch.Tensor: Output hidden states after transformer blocks and layer norm
        """
        hidden_states = inputs_embeds

        # Generate position_ids if positional embeddings are to be used
        if self.use_positional_embeddings:
            seq_length = hidden_states.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device)
            position_ids = position_ids.unsqueeze(0).expand(hidden_states.size(0), -1)
            # print("Using Positional Embeddings")
        else:
            # Create a tensor of zeros to neutralize positional embeddings
            position_ids = torch.zeros(hidden_states.size(0), hidden_states.size(1), dtype=torch.long, device=hidden_states.device)
            # print("Positional Embeddings Disabled")

        # Pass through the selected transformer blocks
        for layer in self.transformer_blocks:
            outputs = layer(
                hidden_states=hidden_states,
                attention_mask=None,       # Adjust if you have an attention mask
                position_ids=position_ids,
                past_key_value=None,
                use_cache=False,
                output_attentions=False,
            )
            hidden_states = outputs[0]

        # Apply the final layer norm
        hidden_states = self.norm(hidden_states)

        return hidden_states

class TabularLLaMA(nn.Module):
    """Combining a tabular model with optional LLaMA attention and positional embeddings."""
    def __init__(self, base_model, base_output_dim, llama_model_name=None,
                 output_classes=1, use_llama=False, start_layer=0, end_layer=1,
                 use_positional_embeddings=True, token=None):
        super(TabularLLaMA, self).__init__()

        self.base_model = base_model
        self.use_llama = use_llama

        if self.use_llama:
            self.llama = HuggingFaceLLaMA(
                llama_model_name,
                start_layer=start_layer,
                end_layer=end_layer,
                token=token,
                use_positional_embeddings=use_positional_embeddings
            )

            # Dimensional mapping between base model and LLaMA
            llama_hidden_dim = self.llama.model.config.hidden_size
            self.mapper1 = nn.Sequential(
                nn.Linear(base_output_dim, llama_hidden_dim),
            )
            self.mapper2 = nn.Sequential(
                nn.Linear(llama_hidden_dim, base_output_dim),
            )

        self.classifier = nn.Linear(base_output_dim, output_classes)

    def forward(self, x):
        base_out = self.base_model(x)  # Shape: [batch_size, base_output_dim]
        residualout_before = base_out.clone()
        if self.use_llama:
            llama_input = self.mapper1(base_out)  # Map to llama_hidden_dim
            llama_input = llama_input.unsqueeze(1)  # Add sequence dimension

            # Save residual for before LLaMA
            residual_before = llama_input.clone()

            # Ensure llama_input is in torch.float16
            llama_input = llama_input.to(torch.float16)

            # Pass through the LLaMA layers
            llama_out = self.llama(llama_input)

            # Add residual connection after LLaMA
            llama_out = llama_out + residual_before.to(torch.float16)*0.0001

            # Remove the sequence dimension
            llama_out = llama_out.squeeze(1)

            # Convert back to torch.float32
            llama_out = llama_out.to(torch.float32)

            # Map back to base model dimension
            base_out = self.mapper2(llama_out)
            
        base_out = residualout_before + base_out
        # Final residual connection can also be added here if needed
        output = self.classifier(base_out)
        return output



## Use the following code to add the whole LLama

# class HuggingFaceLLaMA(nn.Module):
#     """
#     LLaMA model loaded using Hugging Face's transformers library with parameters frozen.
#     """
#     def __init__(self, model_name):
#         super(HuggingFaceLLaMA, self).__init__()

#         # Load the model configuration
#         config = AutoConfig.from_pretrained(model_name)

#         # Adjust rope_scaling if necessary
#         if hasattr(config, 'rope_scaling'):
#             rope_scaling = config.rope_scaling
#             if rope_scaling is not None:
#                 rope_scaling = {'type': rope_scaling.get('type', 'dynamic'), 'factor': rope_scaling.get('factor', 1.0)}
#                 config.rope_scaling = rope_scaling

#         # Load the model with the modified configuration
#         self.model = AutoModelForCausalLM.from_pretrained(
#             model_name,
#             config=config,
#             torch_dtype=torch.float16,
#             low_cpu_mem_usage=True,
#         )

#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)

#         # Add a new pad_token
#         self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

#         # Resize embeddings with padding to multiple of 64
#         self.model.resize_token_embeddings(len(self.tokenizer), pad_to_multiple_of=64)

#         # Enable gradient checkpointing to save memory
#         self.model.gradient_checkpointing_enable()

#         # Freeze LLaMA parameters
#         for param in self.model.parameters():
#             param.requires_grad = False

#     def forward(self, inputs):
#         outputs = self.model(**inputs, output_hidden_states=True)
#         return outputs

# class TabularLLaMA(nn.Module):
#     """Combining a tabular model with optional LLaMA attention."""
#     def __init__(self, base_model, base_output_dim, llama_model_name=None, output_classes=1, use_llama=False):
#         super(TabularLLaMA, self).__init__()

#         self.base_model = base_model
#         self.use_llama = use_llama

#         if self.use_llama:
#             self.llama = HuggingFaceLLaMA(llama_model_name)

#             # Dimensional mapping between base model and LLaMA
#             llama_hidden_dim = self.llama.model.config.hidden_size
#             self.mapper1 = nn.Sequential(
#                 nn.Linear(base_output_dim, llama_hidden_dim),
#             )
#             self.mapper2 = nn.Sequential(
#                 nn.Linear(llama_hidden_dim, base_output_dim),
#             )

#         self.classifier = nn.Linear(base_output_dim, output_classes)

#     def forward(self, x):
#         base_out = self.base_model(x)

#         if self.use_llama:
#             llama_input = self.mapper1(base_out)
#             llama_input = llama_input.unsqueeze(1)  # Add sequence dimension

#             # Ensure llama_input is in torch.float16
#             llama_input = llama_input.to(torch.float16)

#             # Move llama_input to the same device as LLaMA model
#             llama_input = llama_input.to(next(self.llama.model.parameters()).device)

#             outputs = self.llama.model(inputs_embeds=llama_input, output_hidden_states=True)

#             # Extract the last hidden state
#             llama_out = outputs.hidden_states[-1]

#             # Remove the sequence dimension
#             llama_out = llama_out.squeeze(1)

#             # Convert llama_out back to torch.float32
#             llama_out = llama_out.to(torch.float32)

#             # Map LLaMA output back to the base model dimension
#             base_out = self.mapper2(llama_out) 

#         output = self.classifier(base_out)
#         return output
