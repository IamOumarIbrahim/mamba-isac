import torch
import torch.nn as nn
from typing import List
from models.selective_scan import BidirectionalMambaBlock

class MambaISACBackbone(nn.Module):
    """
    Stacked Bidirectional Mamba Backbone for joint ISAC channel and target parameter estimation.
    """
    def __init__(
        self,
        d_model: int = 64,
        d_state: int = 16,
        expand: int = 2,
        num_layers: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'norm': nn.LayerNorm(d_model),
                'mamba': BidirectionalMambaBlock(d_model=d_model, d_state=d_state, expand=expand),
                'dropout': nn.Dropout(dropout)
            })
            for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, bidirectional: bool = True) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model) input embeddings
            bidirectional: Whether to perform bidirectional scan
        Returns:
            out: (B, T, d_model) representation tokens
        """
        h = x
        for layer in self.layers:
            residual = h
            h_norm = layer['norm'](h)
            mamba_out = layer['mamba'](h_norm, bidirectional=bidirectional)
            h = residual + layer['dropout'](mamba_out)
            
        out = self.final_norm(h)
        return out
