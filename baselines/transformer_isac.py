import torch
import torch.nn as nn
from typing import Dict, Any, Tuple

from models.embeddings import DualDomainEmbedding
from models.heads import CommunicationHead, SensingHead

class TransformerISAC(nn.Module):
    """
    Encoder-only Transformer baseline for joint ISAC channel and target parameter estimation.
    Uses multi-head self-attention with O(T^2) complexity, matched in parameter count to Mamba-ISAC.
    Uses residual channel estimation: H_c_hat = Y_obs + Delta_H.
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
        self.nhead = 4
        self.num_layers = md_cfg['num_layers']
        self.dropout = md_cfg['dropout']
        
        # 1. Reuse identical Dual Domain Embedding
        self.embedding = DualDomainEmbedding(
            num_subcarriers=self.Nc,
            num_time_slots=self.T,
            num_tx_antennas=self.Nt,
            num_rx_antennas=self.Nr,
            d_model=self.d_model,
            dropout=self.dropout
        )
        
        # 2. Transformer Encoder Backbone
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.d_model * 2,
            dropout=self.dropout,
            activation='gelu',
            batch_first=True
        )
        self.backbone = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        
        # 3. Reuse identical Dual Task Heads
        self.comm_head = CommunicationHead(
            num_subcarriers=self.Nc,
            num_time_slots=self.T,
            num_tx_antennas=self.Nt,
            num_rx_antennas=self.Nr,
            d_model=self.d_model
        )
        self.sensing_head = SensingHead(d_model=self.d_model)

    def forward(self, Y_obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_embed = self.embedding(Y_obs)
        z_rep = self.backbone(z_embed)
        
        delta_H = self.comm_head(z_rep)
        H_c_hat = Y_obs + delta_H
        
        R_hat, nu_s_hat = self.sensing_head(z_rep)
        
        return H_c_hat, R_hat, nu_s_hat
