import torch
import torch.nn as nn
from models.mamba_isac import MambaISAC, ISACMambaBlock

class MambaNet(nn.Module):
    """
    Standard MambaNet stub that applies sequence modeling across time without explicit 
    spatial-frequency-time factorized blocks.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Simplified backbone mapping raw input to embeddings, followed by standard Mamba blocks
        d_model = config['model']['d_model']
        n_layers = config['model']['n_layers']
        
        Nr = config['dataset']['num_rx_antennas']
        Nt = config['dataset']['num_tx_antennas']
        
        self.input_proj = nn.Linear(Nr * Nt * 2, d_model)
        
        # Standard generic sequence modeling blocks
        self.layers = nn.ModuleList([
            ISACMambaBlock(
                d_model=d_model,
                d_state=config['model']['d_state'],
                d_conv=config['model']['d_conv'],
                expand=config['model']['expand']
            ) for _ in range(n_layers)
        ])
        
        self.comm_head = nn.Linear(d_model, 2)
        self.range_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )
        self.doppler_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )
        
    def forward(self, x):
        # x shape: (B, 2, Nr, Nt, Nc, T)
        B, _, Nr, Nt, Nc, T = x.shape
        x_flat = x.permute(0, 4, 5, 2, 3, 1).reshape(B, Nc, T, Nr * Nt * 2)
        
        # Treat (Nc, T) as a single flat sequence
        seq_len = Nc * T
        x_emb = self.input_proj(x_flat).view(B, seq_len, -1)
        
        hidden = x_emb
        for layer in self.layers:
            hidden = layer(hidden)
            
        # Task heads
        H_c_hat = self.comm_head(hidden).view(B, Nc, T, 2)
        
        pooled = torch.mean(hidden, dim=1)
        R_hat = self.range_head(pooled).squeeze(-1)
        nu_s_hat = self.doppler_head(pooled).squeeze(-1)
        
        return H_c_hat, R_hat, nu_s_hat

class CPMamba(MambaNet):
    """
    CP-Mamba (Cross-Patch Mamba) stub.
    """
    pass

class ChannelMamba(MambaNet):
    """
    ChannelMamba stub focusing on channel estimation.
    """
    pass

class MambaCSP(MambaNet):
    """
    MambaCSP (Cross-Stage Partial) stub.
    """
    pass
