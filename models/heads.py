import torch
import torch.nn as nn
from typing import Tuple

class CommunicationHead(nn.Module):
    """
    Task-specific communication head mapping backbone hidden states (B, T, d_model)
    to full frequency-domain CSI matrix H_c (B, 2, Nr, Nt, Nc, T).
    """
    def __init__(
        self,
        num_subcarriers: int = 64,
        num_time_slots: int = 16,
        num_tx_antennas: int = 4,
        num_rx_antennas: int = 4,
        d_model: int = 64
    ):
        super().__init__()
        self.Nc = num_subcarriers
        self.T = num_time_slots
        self.Nt = num_tx_antennas
        self.Nr = num_rx_antennas
        self.d_model = d_model
        
        # Output dim per time slot: real + imag (2) * Nr * Nt * Nc
        self.out_dim_per_slot = 2 * self.Nr * self.Nt * self.Nc
        
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, self.out_dim_per_slot)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, T, d_model)
        Returns:
            H_c_hat: (B, 2, Nr, Nt, Nc, T) float32 estimated CSI tensor
        """
        B, T, _ = z.shape
        
        # Project each time slot token -> (B, T, 2 * Nr * Nt * Nc)
        out_flat = self.mlp(z) # (B, T, 2 * Nr * Nt * Nc)
        
        # Reshape to (B, T, 2, Nr, Nt, Nc) and permute to (B, 2, Nr, Nt, Nc, T)
        H_c_hat = out_flat.view(B, T, 2, self.Nr, self.Nt, self.Nc).permute(0, 2, 3, 4, 5, 1)
        return H_c_hat


class SensingHead(nn.Module):
    """
    Task-specific regression head estimating target range R (meters)
    and target Doppler shift nu_s (Hz).
    """
    def __init__(self, d_model: int = 64, hidden_dim: int = 128):
        super().__init__()
        # Attention pooling across sequence length T
        self.attn_pool = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Softmax(dim=1)
        )
        
        # Range regression MLP
        self.range_head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Doppler regression MLP
        self.doppler_head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z: (B, T, d_model)
        Returns:
            R_hat: (B,) range estimate in meters
            nu_s_hat: (B,) Doppler estimate in Hz
        """
        # Attention pooling: weights (B, T, 1)
        attn_weights = self.attn_pool(z)
        z_pooled = torch.sum(z * attn_weights, dim=1) # (B, d_model)
        
        R_hat = self.range_head(z_pooled).squeeze(-1) # (B,)
        nu_s_hat = self.doppler_head(z_pooled).squeeze(-1) # (B,)
        
        return R_hat, nu_s_hat
