import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    HAS_MAMBA_SSM = True
except ImportError:
    HAS_MAMBA_SSM = False

class SelectiveScanFunction(nn.Module):
    """
    Selective State-Space (S6) Scan Block.
    Optimized with official CUDA selective_scan_fn when available, falling back to pure PyTorch loop.
    Eq. (2)-(3) in mamba_isac_briefing.tex.
    """
    def __init__(
        self,
        d_model: int = 64,
        d_state: int = 16,
        dt_rank: Optional[int] = None,
        dt_min: float = 0.001,
        dt_max: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = math.ceil(d_model / 16) if dt_rank is None else dt_rank
        
        # State space parameters
        self.A_log = nn.Parameter(torch.log(torch.repeat_interleave(torch.arange(1, d_state + 1, dtype=torch.float32), d_model).view(d_model, d_state)))
        self.D = nn.Parameter(torch.ones(d_model))
        
        # Selective parameter projections: z_t -> Delta_t, B_t, C_t
        self.x_proj = nn.Linear(d_model, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_model, bias=True)
        
        # Initialize dt_proj bias
        dt = torch.exp(
            torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model) input sequence
        Returns:
            out: (B, T, d_model) output sequence
        """
        B, T, D_dim = x.shape
        N_dim = self.d_state
        
        if HAS_MAMBA_SSM and x.is_cuda:
            # Official CUDA Kernel execution path
            x_dbl = self.x_proj(x)
            dt_raw, B_param, C_param = torch.split(x_dbl, [self.dt_rank, N_dim, N_dim], dim=-1)
            
            # Format inputs for the Mamba interface
            x_bdt = x.transpose(1, 2).contiguous() # (B, D_dim, T)
            dt_raw_bdt = dt_raw.transpose(1, 2).contiguous()
            B_param_bnt = B_param.transpose(1, 2).contiguous()
            C_param_bnt = C_param.transpose(1, 2).contiguous()
            A_nd = -torch.exp(self.A_log.float()) # (D_dim, N_dim)
            
            out_bdt = selective_scan_fn(
                x_bdt,
                dt_raw_bdt,
                A_nd,
                B_param_bnt,
                C_param_bnt,
                self.D.float(),
                z=None,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=False
            )
            return out_bdt.transpose(1, 2)
        else:
            # Fallback PyTorch implementation (naive loop)
            A = -torch.exp(self.A_log)
            x_dbl = self.x_proj(x)
            dt_raw, B_param, C_param = torch.split(x_dbl, [self.dt_rank, N_dim, N_dim], dim=-1)
            delta = F.softplus(self.dt_proj(dt_raw))
            
            bar_A = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
            bar_Bx = (delta.unsqueeze(-1) * B_param.unsqueeze(2)) * x.unsqueeze(-1)
            
            h = torch.zeros(B, D_dim, N_dim, device=x.device, dtype=x.dtype)
            y_list = []
            
            for t in range(T):
                h = bar_A[:, t] * h + bar_Bx[:, t]
                y_t = torch.sum(h * C_param[:, t].unsqueeze(1), dim=-1)
                y_list.append(y_t)
                
            y = torch.stack(y_list, dim=1)
            return y + x * self.D

class BidirectionalMambaBlock(nn.Module):
    """
    Bidirectional Mamba Block combining forward and reverse selective scans.
    """
    def __init__(
        self,
        d_model: int = 64,
        d_state: int = 16,
        expand: int = 2,
        d_conv: int = 4
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = expand * d_model
        
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            bias=True,
            padding=d_conv - 1,
            groups=self.d_inner
        )
        
        self.scan_fwd = SelectiveScanFunction(d_model=self.d_inner, d_state=d_state)
        self.scan_bwd = SelectiveScanFunction(d_model=self.d_inner, d_state=d_state)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor, bidirectional: bool = True) -> torch.Tensor:
        B, T, D_dim = x.shape
        
        x_proj, z_gate = self.in_proj(x).chunk(2, dim=-1) # (B, T, d_inner)
        
        x_conv = x_proj.permute(0, 2, 1) # (B, d_inner, T)
        x_conv = self.conv1d(x_conv)[:, :, :T].permute(0, 2, 1) # (B, T, d_inner)
        x_conv = F.silu(x_conv)
        
        y_fwd = self.scan_fwd(x_conv)
        
        if bidirectional:
            x_rev = torch.flip(x_conv, dims=[1])
            y_bwd_rev = self.scan_bwd(x_rev)
            y_bwd = torch.flip(y_bwd_rev, dims=[1])
            y_scan = (y_fwd + y_bwd) / 2.0
        else:
            y_scan = y_fwd
            
        y_gated = y_scan * F.silu(z_gate)
        out = self.out_proj(y_gated)
        return out
