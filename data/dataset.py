import os
import yaml
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, Any, Tuple

from data.comm_channel import CommunicationChannelGenerator
from data.sensing_channel import SensingChannelGenerator
from data.pilots import ISACPilotAllocator
from utils.seed import set_seed

def generate_isac_samples(
    config: Dict[str, Any],
    num_samples: int,
    seed: int = 42
) -> Dict[str, np.ndarray]:
    """
    Generates joint Communication-Sensing OFDM ISAC dataset split.
    
    Returns dictionary with keys:
        'H_c': (N, Nr, Nt, Nc, T) complex128 clean comm CSI
        'Y_obs': (N, Nr, Nt, Nc, T) complex128 noisy combined pilot/data observation
        'pilot_mask': (Nc, T) bool pilot mask
        'range': (N,) float64 target range ground truth
        'velocity': (N,) float64 target velocity ground truth
        'doppler_s': (N,) float64 target Doppler shift ground truth
    """
    set_seed(seed)
    
    ds_cfg = config['dataset']
    comm_cfg = config['comm_channel']
    sens_cfg = config['sensing_channel']
    pilot_cfg = config['pilots']
    
    Nc = ds_cfg['num_subcarriers']
    T = ds_cfg['num_time_slots']
    Nt = ds_cfg['num_tx_antennas']
    Nr = ds_cfg['num_rx_antennas']
    fc = float(ds_cfg['carrier_frequency'])
    df = float(ds_cfg['subcarrier_spacing'])
    Ts = float(ds_cfg['symbol_duration'])
    c = 3.0e8
    
    comm_gen = CommunicationChannelGenerator(
        num_subcarriers=Nc,
        num_time_slots=T,
        num_tx_antennas=Nt,
        num_rx_antennas=Nr,
        carrier_frequency=fc,
        subcarrier_spacing=df,
        symbol_duration=Ts,
        k_factor_db=float(comm_cfg['k_factor_db']),
        user_velocity=float(comm_cfg['user_velocity'])
    )
    
    sens_gen = SensingChannelGenerator(
        num_subcarriers=Nc,
        num_time_slots=T,
        num_tx_antennas=Nt,
        num_rx_antennas=Nr,
        carrier_frequency=fc,
        subcarrier_spacing=df,
        symbol_duration=Ts
    )
    
    pilot_alloc = ISACPilotAllocator(
        num_subcarriers=Nc,
        num_time_slots=T,
        pilot_spacing=int(pilot_cfg['pilot_spacing'])
    )
    
    _, pilot_mask = pilot_alloc.get_pilot_indices()
    
    H_c_list = []
    Y_obs_list = []
    range_list = []
    velocity_list = []
    doppler_s_list = []
    
    snr_db = float(comm_cfg['snr_db'])
    
    for i in range(num_samples):
        # Generate random target parameters per sample
        r_i = float(np.random.uniform(20.0, 100.0))
        v_i = float(np.random.uniform(-30.0, 30.0))
        azimuth_i = float(np.random.uniform(-np.pi/6, np.pi/6))
        
        # Clean comm channel
        H_c_sample, _ = comm_gen.generate_channel(batch_size=1, snr_db=None)
        H_c_sample = H_c_sample.squeeze(0) # (Nr, Nt, Nc, T)
        
        # Generate sensing echo
        y_s_sample, tau_i, nu_s_i = sens_gen.generate_echo(
            range_m=r_i,
            velocity_ms=v_i,
            reflectivity=1.0 + 0.0j,
            azimuth_rad=azimuth_i,
            snr_db=None
        )
        
        # Joint channel = H_c + sensing echo component
        H_joint = H_c_sample + y_s_sample
        
        # Add AWGN noise
        snr_linear = 10.0 ** (snr_db / 10.0)
        signal_power = np.mean(np.abs(H_joint) ** 2)
        noise_std = np.sqrt(signal_power / (2.0 * snr_linear))
        noise = (np.random.randn(*H_joint.shape) + 1j * np.random.randn(*H_joint.shape)) * noise_std
        
        Y_obs_sample = H_joint + noise
        
        H_c_list.append(H_c_sample)
        Y_obs_list.append(Y_obs_sample)
        range_list.append(r_i)
        velocity_list.append(v_i)
        doppler_s_list.append(nu_s_i)
        
    dataset_dict = {
        'H_c': np.array(H_c_list),              # (N, Nr, Nt, Nc, T)
        'Y_obs': np.array(Y_obs_list),          # (N, Nr, Nt, Nc, T)
        'pilot_mask': pilot_mask,               # (Nc, T)
        'range': np.array(range_list),          # (N,)
        'velocity': np.array(velocity_list),    # (N,)
        'doppler_s': np.array(doppler_s_list)   # (N,)
    }
    return dataset_dict


class ISACDataset(Dataset):
    """
    PyTorch Dataset wrapper for ISAC joint channel and parameter estimation.
    Splits complex matrices into real/imag channels for PyTorch models.
    """
    def __init__(self, data_dict: Dict[str, np.ndarray]):
        # Convert complex (N, Nr, Nt, Nc, T) into real/imag tensor (N, 2, Nr, Nt, Nc, T) or (N, 2*Nr*Nt*Nc, T)
        H_c = data_dict['H_c']
        Y_obs = data_dict['Y_obs']
        
        # Real and imaginary components concatenated along channel dimension
        # Y_obs: shape (N, 2, Nr, Nt, Nc, T)
        Y_real_imag = np.stack([np.real(Y_obs), np.imag(Y_obs)], axis=1)
        H_real_imag = np.stack([np.real(H_c), np.imag(H_c)], axis=1)
        
        self.Y_obs = torch.tensor(Y_real_imag, dtype=torch.float32)
        self.H_c = torch.tensor(H_real_imag, dtype=torch.float32)
        self.pilot_mask = torch.tensor(data_dict['pilot_mask'], dtype=torch.bool)
        self.range = torch.tensor(data_dict['range'], dtype=torch.float32)
        self.velocity = torch.tensor(data_dict['velocity'], dtype=torch.float32)
        self.doppler_s = torch.tensor(data_dict['doppler_s'], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.range)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'Y_obs': self.Y_obs[idx],
            'H_c': self.H_c[idx],
            'pilot_mask': self.pilot_mask,
            'range': self.range[idx],
            'velocity': self.velocity[idx],
            'doppler_s': self.doppler_s[idx]
        }
