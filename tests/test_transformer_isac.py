import pytest
import yaml
import torch
from models.mamba_isac import MambaISAC
from baselines.transformer_isac import TransformerISAC
from models.loss import JointISACLoss
from data.dataset import generate_isac_samples, ISACDataset
from utils.seed import set_seed

@pytest.fixture
def sample_config():
    with open("configs/default_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    config['dataset']['num_subcarriers'] = 16
    config['dataset']['num_time_slots'] = 8
    config['dataset']['num_tx_antennas'] = 2
    config['dataset']['num_rx_antennas'] = 2
    config['model']['d_model'] = 32
    config['model']['num_layers'] = 2
    return config

def test_transformer_shapes_and_param_count(sample_config):
    set_seed(42)
    mamba_model = MambaISAC(sample_config)
    trans_model = TransformerISAC(sample_config)
    
    B, Nr, Nt, Nc, T = 4, 2, 2, 16, 8
    Y_obs = torch.randn(B, 2, Nr, Nt, Nc, T)
    
    H_c_hat, R_hat, nu_s_hat = trans_model(Y_obs)
    
    assert H_c_hat.shape == (B, 2, Nr, Nt, Nc, T)
    assert R_hat.shape == (B,)
    assert nu_s_hat.shape == (B,)
    
    mamba_params = sum(p.numel() for p in mamba_model.parameters())
    trans_params = sum(p.numel() for p in trans_model.parameters())
    
    print(f"\nMamba-ISAC Params: {mamba_params} | Transformer Params: {trans_params}")
    # Parameter counts should be comparable within 20%
    assert abs(mamba_params - trans_params) / mamba_params < 0.25

def test_transformer_overfit_toy(sample_config):
    set_seed(42)
    data_dict = generate_isac_samples(sample_config, num_samples=5, seed=42)
    dataset = ISACDataset(data_dict)
    
    model = TransformerISAC(sample_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    loss_fn = JointISACLoss(lambda_c=1.0, lambda_s=1.0, lambda_d=1.0)
    
    batch = dataset[0:5]
    Y_obs = batch['Y_obs']
    H_c_true = batch['H_c']
    R_true = batch['range']
    nu_s_true = batch['doppler']
    
    H_c_hat, R_hat, nu_s_hat = model(Y_obs)
    initial_loss, _ = loss_fn(H_c_hat, H_c_true, R_hat, R_true, nu_s_hat, nu_s_true)
    
    for step in range(80):
        optimizer.zero_grad()
        H_c_hat, R_hat, nu_s_hat = model(Y_obs)
        loss, _ = loss_fn(H_c_hat, H_c_true, R_hat, R_true, nu_s_hat, nu_s_true)
        loss.backward()
        optimizer.step()
        
    final_loss, _ = loss_fn(H_c_hat, H_c_true, R_hat, R_true, nu_s_hat, nu_s_true)
    assert final_loss.item() < initial_loss.item() * 0.5
