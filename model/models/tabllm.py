import math
import typing as ty

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as nn_init
from torch import Tensor

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModel

class HuggingFaceLLaMA(nn.Module):
    """
    LLaMA model loaded using Hugging Face's transformers library with parameters frozen.
    Allows selection of specific layers and optional positional embeddings.
    """
    def __init__(self, model_name, start_layer=0, end_layer=1, hf_token=None):
        super(HuggingFaceLLaMA, self).__init__()

        # Load the model configuration with token
        config = AutoConfig.from_pretrained(model_name, use_auth_token=hf_token)

        # Load the pre-trained model without the language modeling head
        self.model = AutoModel.from_pretrained(
            model_name,
            config=config,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_auth_token=hf_token
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)

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
        seq_length = hidden_states.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device)
        position_ids = position_ids.unsqueeze(0).expand(hidden_states.size(0), -1)

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
    def __init__(
            self, 
            base_model: nn.Module,
            d_out: int, 
            base_output_dim: int, 
            llm_model_name:str = None,
            start_layer: int = 0,
            end_layer: int = 1
            ):
        super(TabularLLaMA, self).__init__()

        self.base_model = base_model
        self.llama = HuggingFaceLLaMA(
            llm_model_name,
            start_layer=start_layer,
            end_layer=end_layer
        )

        # Dimensional mapping between base model and LLaMA
        llama_hidden_dim = self.llama.model.config.hidden_size
        self.mapper1 = nn.Sequential(
            nn.Linear(base_output_dim, llama_hidden_dim),
        )
        self.mapper2 = nn.Sequential(
            nn.Linear(llama_hidden_dim, base_output_dim),
        )

        self.classifier = nn.Linear(base_output_dim, d_out)

    def forward(self, x_num: Tensor, x_cat: ty.Optional[Tensor]):
        x_num = x_num.float()
        
        base_out = self.base_model(x_num, x_cat) # Shape: [batch_size, base_output_dim]
        residualout_before = base_out.clone()

        llama_input = self.mapper1(base_out)  # Map to llama_hidden_dim
        llama_input = llama_input.unsqueeze(1)  # Add sequence dimension

        # Save residual for before LLaMA
        residual_before = llama_input.clone()

        # change dtype to half
        llama_input = llama_input.half()

        # Pass through the LLaMA layers
        llama_out = self.llama(llama_input)

        # Convert back to float32
        llama_out = llama_out.float()

        # Add residual connection after LLaMA
        llama_out = llama_out + residual_before*0.0001

        # Remove the sequence dimension
        llama_out = llama_out.squeeze(1)

        # Map back to base model dimension
        base_out = self.mapper2(llama_out)
            
        base_out = residualout_before + base_out
        # Final residual connection can also be added here if needed
        output = self.classifier(base_out)

        # cehck dtype of output
        return output
