import torch
import torch.nn as nn
from typing import Dict, Any

def count_parameters(model: nn.Module) -> int:
    """Returns total trainable parameter count."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def estimate_flops(model: nn.Module, sample_input: torch.Tensor) -> int:
    """
    Estimates total Floating Point Operations (FLOPs) including custom Mamba SSM selective scan modules.
    """
    total_flops = 0
    B, C_ri, Nr, Nt, Nc, T = sample_input.shape
    
    for m in model.modules():
        if isinstance(m, nn.Linear):
            total_flops += 2 * B * T * m.in_features * m.out_features
        elif isinstance(m, nn.Conv1d):
            total_flops += 2 * B * m.in_channels * m.out_channels * m.kernel_size[0] * T
        elif m.__class__.__name__ == "SelectiveScanFunction":
            # Selective scan recurrence FLOPs: 4 * B * T * d_inner * d_state
            d_inner = m.d_model
            d_state = m.d_state
            total_flops += 4 * B * T * d_inner * d_state
            
    return total_flops
