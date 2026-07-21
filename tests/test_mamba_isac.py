import pytest
import time
import yaml
import numpy as np
import torch
from models.mamba_isac import MambaISAC
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

def test_tensor_shapes(sample_config):
    set_seed(42)
    model = MambaISAC(sample_config)
    
    B, Nr, Nt, Nc, T = 4, 2, 2, 16, 8
    Y_obs = torch.randn(B, 2, Nr, Nt, Nc, T)
    
    H_c_hat, R_hat, nu_s_hat = model(Y_obs)
    
    assert H_c_hat.shape == (B, 2, Nr, Nt, Nc, T)
    assert R_hat.shape == (B,)
    assert nu_s_hat.shape == (B,)

def test_gradient_no_nans(sample_config):
    set_seed(42)
    model = MambaISAC(sample_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = JointISACLoss()
    
    B, Nr, Nt, Nc, T = 4, 2, 2, 16, 8
    
    for step in range(20):
        optimizer.zero_grad()
        Y_obs = torch.randn(B, 2, Nr, Nt, Nc, T)
        H_c_true = torch.randn(B, 2, Nr, Nt, Nc, T)
        R_true = torch.full((B,), 50.0)
        nu_s_true = torch.full((B,), 200.0)
        
        H_c_hat, R_hat, nu_s_hat = model(Y_obs)
        loss, _ = loss_fn(H_c_hat, H_c_true, R_hat, R_true, nu_s_hat, nu_s_true)
        loss.backward()
        
        # Check no NaNs/Infs in parameters or gradients
        for name, param in model.named_parameters():
            assert not torch.isnan(param).any(), f"NaN found in param {name}"
            assert not torch.isinf(param).any(), f"Inf found in param {name}"
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"NaN found in grad of {name}"
                assert not torch.isinf(param.grad).any(), f"Inf found in grad of {name}"
                
        optimizer.step()

def test_overfit_toy_sequence(sample_config):
    set_seed(42)
    data_dict = generate_isac_samples(sample_config, num_samples=5, seed=42)
    dataset = ISACDataset(data_dict)
    
    model = MambaISAC(sample_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    loss_fn = JointISACLoss(lambda_c=1.0, lambda_s=0.01, lambda_d=0.001)
    
    batch = dataset[0:5]
    Y_obs = batch['Y_obs']
    H_c_true = batch['H_c']
    R_true = batch['range']
    nu_s_true = batch['doppler_s']
    
    # Measure initial loss
    H_c_hat, R_hat, nu_s_hat = model(Y_obs)
    initial_loss, _ = loss_fn(H_c_hat, H_c_true, R_hat, R_true, nu_s_hat, nu_s_true)
    
    # Train overfit loop
    for step in range(80):
        optimizer.zero_grad()
        H_c_hat, R_hat, nu_s_hat = model(Y_obs)
        loss, _ = loss_fn(H_c_hat, H_c_true, R_hat, R_true, nu_s_hat, nu_s_true)
        loss.backward()
        optimizer.step()
        
    final_loss, _ = loss_fn(H_c_hat, H_c_true, R_hat, R_true, nu_s_hat, nu_s_true)
    
    # Loss should decrease significantly (> 50% reduction on toy batch)
    assert final_loss.item() < initial_loss.item() * 0.5

def test_linear_time_scaling(sample_config):
    set_seed(42)
    model = MambaISAC(sample_config)
    model.eval()
    
    seq_lengths = [8, 16, 32, 64]
    runtimes = []
    
    with torch.no_grad():
        for T_len in seq_lengths:
            sample_config['dataset']['num_time_slots'] = T_len
            Y_obs = torch.randn(2, 2, 2, 2, 16, T_len)
            
            # Warmup
            _ = model(Y_obs)
            
            start = time.perf_counter()
            for _ in range(5):
                _ = model(Y_obs)
            end = time.perf_counter()
            runtimes.append((end - start) / 5.0)
            
    # Empirical linear check: ratio of time ratio to sequence length ratio should be sub-quadratic (< 2.0)
    time_ratio = runtimes[-1] / runtimes[0]
    seq_ratio = seq_lengths[-1] / seq_lengths[0]
    scaling_exponent = np.log(time_ratio) / np.log(seq_ratio)
    
    print(f"\nEmpirical sequence scaling exponent: {scaling_exponent:.2f} (linear is 1.0, quadratic is 2.0)")
    assert scaling_exponent < 1.8, "Mamba-ISAC complexity should be linear O(T), not quadratic"
