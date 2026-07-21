import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class SelectiveScanFunction(nn.Module):
    """
    Pure PyTorch Selective State-Space (S6) Scan Block.
    Computes input-dependent state transitions (Delta, A, B, C) and linear recurrence.
    Eq. (2)-(3) in mamba_isac_briefing.tex:
        h_t = bar_A_t * h_{t-1} + bar_B_t * z_t
        o_t = C_t * h_t
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
        
        A = -torch.exp(self.A_log) # (D_dim, N_dim)
        
        # Project x to (dt_rank + 2*N_dim)
        x_dbl = self.x_proj(x) # (B, T, dt_rank + 2*N_dim)
        dt_raw, B_param, C_param = torch.split(x_dbl, [self.dt_rank, N_dim, N_dim], dim=-1)
        
        # Discretization step Delta = softplus(dt_proj(dt_raw))
        delta = F.softplus(self.dt_proj(dt_raw)) # (B, T, D_dim)
        
        # Sequential selective scan loop over sequence length T
        # h_t = exp(Delta * A) * h_{t-1} + (Delta * B) * x_t
        h = torch.zeros(B, D_dim, N_dim, device=x.device, dtype=x.dtype)
        y_list = []
        
        for t in range(T):
            delta_t = delta[:, t, :] # (B, D_dim)
            x_t = x[:, t, :]       # (B, D_dim)
            B_t = B_param[:, t, :] # (B, N_dim)
            C_t = C_param[:, t, :] # (B, N_dim)
            
            # Discretized A_bar = exp(delta_t * A)
            # delta_t: (B, D_dim, 1), A: (1, D_dim, N_dim)
            bar_A = torch.exp(delta_t.unsqueeze(-1) * A.unsqueeze(0)) # (B, D_dim, N_dim)
            
            # Discretized B_bar = delta_t * B_t
            bar_B = delta_t.unsqueeze(-1) * B_t.unsqueeze(1) # (B, D_dim, N_dim)
            
            # State update: h_t = bar_A * h_{t-1} + bar_B * x_t
            h = bar_A * h + bar_B * x_t.unsqueeze(-1)
            
            # Output y_t = sum_n (C_t * h_t)
            y_t = torch.sum(h * C_t.unsqueeze(1), dim=-1) # (B, D_dim)
            y_list.append(y_t)
            
        y = torch.stack(y_list, dim=1) # (B, T, D_dim)
        out = y + x * self.D
        return out


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
        
        # In projection
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        
        # 1D Depthwise Conv
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            bias=True,
            padding=d_conv - 1,
            groups=self.d_inner
        )
        
        # Forward and reverse selective scan modules
        self.scan_fwd = SelectiveScanFunction(d_model=self.d_inner, d_state=d_state)
        self.scan_bwd = SelectiveScanFunction(d_model=self.d_inner, d_state=d_state)
        
        # Out projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor, bidirectional: bool = True) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
            bidirectional: If True, combines forward and reverse scans.
        Returns:
            out: (B, T, d_model)
        """
        B, T, D_dim = x.shape
        
        # Linear projection -> x_proj, z_gate
        x_proj, z_gate = self.in_proj(x).chunk(2, dim=-1) # (B, T, d_inner)
        
        # Conv1d
        x_conv = x_proj.permute(0, 2, 1) # (B, d_inner, T)
        x_conv = self.conv1d(x_conv)[:, :, :T].permute(0, 2, 1) # (B, T, d_inner)
        x_conv = F.silu(x_conv)
        
        # Forward selective scan
        y_fwd = self.scan_fwd(x_conv)
        
        if bidirectional:
            # Reverse selective scan along sequence dimension
            x_rev = torch.flip(x_conv, dims=[1])
            y_bwd_rev = self.scan_bwd(x_rev)
            y_bwd = torch.flip(y_bwd_rev, dims=[1])
            y_scan = (y_fwd + y_bwd) / 2.0
        else:
            y_scan = y_fwd
            
        # Gated multiplicative residual
        y_gated = y_scan * F.silu(z_gate)
        
        out = self.out_proj(y_gated)
        return out
