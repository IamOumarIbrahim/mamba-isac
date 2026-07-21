import pytest
import yaml
import numpy as np
import torch
from data.dataset import generate_isac_samples, ISACDataset
from utils.seed import set_seed

def test_dataset_generation_and_shapes():
    set_seed(42)
    with open("configs/default_config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    num_samples = 5
    data_dict = generate_isac_samples(config, num_samples=num_samples, seed=42)
    
    # Verify keys
    assert 'H_c' in data_dict
    assert 'Y_obs' in data_dict
    assert 'pilot_mask' in data_dict
    assert 'range' in data_dict
    assert 'velocity' in data_dict
    assert 'doppler_s' in data_dict
    
    # Check tensor shapes
    Nc = config['dataset']['num_subcarriers']
    T = config['dataset']['num_time_slots']
    Nt = config['dataset']['num_tx_antennas']
    Nr = config['dataset']['num_rx_antennas']
    
    assert data_dict['H_c'].shape == (num_samples, Nr, Nt, Nc, T)
    assert data_dict['Y_obs'].shape == (num_samples, Nr, Nt, Nc, T)
    assert data_dict['pilot_mask'].shape == (Nc, T)
    assert data_dict['range'].shape == (num_samples,)
    assert data_dict['velocity'].shape == (num_samples,)
    
    # Test PyTorch dataset wrapper
    dataset = ISACDataset(data_dict)
    assert len(dataset) == num_samples
    
    sample = dataset[0]
    # Y_obs should be real/imag split: (2, Nr, Nt, Nc, T)
    assert sample['Y_obs'].shape == (2, Nr, Nt, Nc, T)
    assert sample['H_c'].shape == (2, Nr, Nt, Nc, T)
    assert sample['range'].dtype == torch.float32
