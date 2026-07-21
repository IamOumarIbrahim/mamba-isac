import torch
import torch.nn as nn
from typing import Dict, Any, Tuple

from models.embeddings import DualDomainEmbedding
from models.mamba_backbone import MambaISACBackbone
from models.heads import CommunicationHead, SensingHead

class MambaISAC(nn.Module):
    """
    Complete Mamba-ISAC architecture for joint communication CSI and sensing parameter estimation.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        ds_cfg = config['dataset']
        md_cfg = config['model']
        
        self.Nc = ds_cfg['num_subcarriers']
        self.T = ds_cfg['num_time_slots']
        self.Nt = ds_cfg['num_tx_antennas']
        self.Nr = ds_cfg['num_rx_antennas']
        
        self.d_model = md_cfg['d_model']
        self.d_state = md_cfg['d_state']
        self.expand = md_cfg['expand']
        self.num_layers = md_cfg['num_layers']
        self.dropout = md_cfg['dropout']
        
        # 1. Dual domain input embedding
        self.embedding = DualDomainEmbedding(
            num_subcarriers=self.Nc,
            num_time_slots=self.T,
            num_tx_antennas=self.Nt,
            num_rx_antennas=self.Nr,
            d_model=self.d_model,
            dropout=self.dropout
        )
        
        # 2. Selective state space backbone
        self.backbone = MambaISACBackbone(
            d_model=self.d_model,
            d_state=self.d_state,
            expand=self.expand,
            num_layers=self.num_layers,
            dropout=self.dropout
        )
        
        # 3. Dual task heads
        self.comm_head = CommunicationHead(
            num_subcarriers=self.Nc,
            num_time_slots=self.T,
            num_tx_antennas=self.Nt,
            num_rx_antennas=self.Nr,
            d_model=self.d_model
        )
        
        self.sensing_head = SensingHead(d_model=self.d_model)

    def forward(self, Y_obs: torch.Tensor, bidirectional: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            Y_obs: (B, 2, Nr, Nt, Nc, T) input observation tensor
            bidirectional: Whether to run bidirectional selective scan
            
        Returns:
            H_c_hat: (B, 2, Nr, Nt, Nc, T) estimated comm channel
            R_hat: (B,) estimated target range (meters)
            nu_s_hat: (B,) estimated target Doppler shift (Hz)
        """
        # Embed -> Backbone -> Heads
        z_embed = self.embedding(Y_obs) # (B, T, d_model)
        z_rep = self.backbone(z_embed, bidirectional=bidirectional) # (B, T, d_model)
        
        H_c_hat = self.comm_head(z_rep)
        R_hat, nu_s_hat = self.sensing_head(z_rep)
        
        return H_c_hat, R_hat, nu_s_hat
