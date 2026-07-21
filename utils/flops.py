import torch
import torch.nn as nn
from typing import Dict, Any, Tuple

def count_parameters(model: nn.Module) -> int:
    """Returns total trainable parameter count."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def estimate_flops(model: nn.Module, sample_input: torch.Tensor) -> int:
    """
    Estimates total Floating Point Operations (FLOPs) using thop or analytical fallback.
    """
    try:
        from thop import profile
        flops, params = profile(model, inputs=(sample_input,), verbose=False)
        return int(flops)
    except Exception:
        # Fallback estimation for linear/conv layers
        total_flops = 0
        for m in model.modules():
            if isinstance(m, nn.Linear):
                total_flops += 2 * m.in_features * m.out_features
            elif isinstance(m, nn.Conv1d):
                total_flops += 2 * m.in_channels * m.out_channels * m.kernel_size[0]
        return total_flops
