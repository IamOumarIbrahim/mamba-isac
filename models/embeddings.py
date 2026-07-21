import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DualDomainEmbedding(nn.Module):
    """
    Dual-Domain Input Embedding stage for Mamba-ISAC.
    Embeds raw complex pilot/sensing observation tensors Y_obs (B, 2, Nr, Nt, Nc, T)
    into sequential d_model token representations Z (B, T, d_model).
    Supports arbitrary sequence lengths T dynamically.
    """
    def __init__(
        self,
        num_subcarriers: int = 64,
        num_time_slots: int = 16,
        num_tx_antennas: int = 4,
        num_rx_antennas: int = 4,
        d_model: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        self.Nc = num_subcarriers
        self.T_default = num_time_slots
        self.Nt = num_tx_antennas
        self.Nr = num_rx_antennas
        self.d_model = d_model
        
        # Raw features per time slot: real + imag (2) * Nr * Nt * Nc
        self.in_dim = 2 * self.Nr * self.Nt * self.Nc
        
        # Frequency-domain linear projection
        self.freq_proj = nn.Linear(self.in_dim, d_model)
        
        # 1D Conv over delay/Doppler domain feature extraction
        self.delay_doppler_conv = nn.Sequential(
            nn.Conv1d(self.in_dim, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        )
        
        self.fusion = nn.Linear(2 * d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Initial learned Positional Embedding
        self.max_pos = 256
        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_pos, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, Y_obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Y_obs: (B, 2, Nr, Nt, Nc, T) float32 tensor
        Returns:
            Z: (B, T, d_model) sequence tokens
        """
        B, C_ri, Nr, Nt, Nc, T = Y_obs.shape
        
        # Reshape to (B, T, 2 * Nr * Nt * Nc)
        x_flat = Y_obs.permute(0, 5, 1, 2, 3, 4).reshape(B, T, self.in_dim)
        
        # Frequency domain feature
        z_freq = self.freq_proj(x_flat) # (B, T, d_model)
        
        # Delay-Doppler feature via 1D Conv along T dimension
        z_dd = self.delay_doppler_conv(x_flat.permute(0, 2, 1)).permute(0, 2, 1) # (B, T, d_model)
        
        # Fuse dual domains
        z_fused = self.fusion(torch.cat([z_freq, z_dd], dim=-1))
        
        # Add positional embedding dynamically sliced/interpolated
        if T <= self.max_pos:
            pos = self.pos_embed[:, :T, :]
        else:
            pos = F.interpolate(self.pos_embed.permute(0, 2, 1), size=T, mode='linear', align_corners=False).permute(0, 2, 1)
            
        out = self.norm(z_fused + pos)
        return self.dropout(out)
