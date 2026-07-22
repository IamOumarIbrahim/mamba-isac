import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, Any

from data.comm_channel import CommunicationChannelGenerator
from data.sensing_channel import SensingChannelGenerator
from data.pilots import ISACPilotAllocator

def generate_isac_samples(config: Dict[str, Any], num_samples: int = 100, seed: int = 42) -> Dict[str, np.ndarray]:
    """
    Generates synthetic ISAC dataset samples with randomized target range and velocity.
    """
    np.random.seed(seed)
    
    Nc = int(config['dataset']['num_subcarriers'])
    T = int(config['dataset']['num_time_slots'])
    Nt = int(config['dataset']['num_tx_antennas'])
    Nr = int(config['dataset']['num_rx_antennas'])
    
    comm_gen = CommunicationChannelGenerator(
        num_subcarriers=Nc,
        num_time_slots=T,
        num_tx_antennas=Nt,
        num_rx_antennas=Nr,
        k_factor_db=float(config['comm_channel']['k_factor_db']),
        user_velocity=float(config['comm_channel']['user_velocity']),
        carrier_frequency=float(config['dataset']['carrier_frequency']),
        subcarrier_spacing=float(config['dataset']['subcarrier_spacing']),
        symbol_duration=float(config['dataset']['symbol_duration'])
    )
    
    sens_gen = SensingChannelGenerator(
        num_subcarriers=Nc,
        num_time_slots=T,
        num_tx_antennas=Nt,
        num_rx_antennas=Nr,
        carrier_frequency=float(config['dataset']['carrier_frequency']),
        subcarrier_spacing=float(config['dataset']['subcarrier_spacing']),
        symbol_duration=float(config['dataset']['symbol_duration'])
    )
    
    snr_db = float(config['comm_channel']['snr_db'])
    
    pilot_alloc = ISACPilotAllocator(
        num_subcarriers=Nc,
        num_time_slots=T,
        pilot_spacing=int(config['pilots']['pilot_spacing'])
    )
    _, pilot_mask = pilot_alloc.get_pilot_indices()
    
    H_c, H_c_noisy = comm_gen.generate_channel(batch_size=num_samples, snr_db=snr_db)
    
    target_ranges = np.random.uniform(20.0, 100.0, size=num_samples)
    target_velocities = np.random.uniform(5.0, 40.0, size=num_samples)
    
    Y_obs_list = []
    nu_s_list = []
    
    for i in range(num_samples):
        y_s, tau, nu_s = sens_gen.generate_echo(
            range_m=target_ranges[i],
            velocity_ms=target_velocities[i],
            reflectivity=complex(config['sensing_channel']['reflectivity_mag']),
            azimuth_rad=float(config['sensing_channel']['target_azimuth'])
        )
        
        Y_obs_i = H_c_noisy[i] + y_s
        Y_obs_list.append(Y_obs_i)
        nu_s_list.append(nu_s)
        
    Y_obs = np.array(Y_obs_list)
    nu_s_arr = np.array(nu_s_list)
    
    return {
        'Y_obs': Y_obs,             # Complex (num_samples, Nr, Nt, Nc, T)
        'H_c': H_c,                 # Complex (num_samples, Nr, Nt, Nc, T)
        'pilot_mask': pilot_mask,   # Boolean (Nc, T)
        'range': target_ranges,     # Float (num_samples,)
        'velocity': target_velocities, # Float (num_samples,)
        'doppler': nu_s_arr         # Float (num_samples,)
    }

class ISACDataset(Dataset):
    def __init__(self, data_dict: Dict[str, np.ndarray]):
        Y_complex = data_dict['Y_obs']
        H_complex = data_dict['H_c']
        
        Y_real_imag = np.stack([Y_complex.real, Y_complex.imag], axis=1)
        H_real_imag = np.stack([H_complex.real, H_complex.imag], axis=1)
        
        self.Y_obs = torch.tensor(Y_real_imag, dtype=torch.float32)
        self.H_c = torch.tensor(H_real_imag, dtype=torch.float32)
        self.pilot_mask = torch.tensor(data_dict['pilot_mask'], dtype=torch.bool)
        self.range = torch.tensor(data_dict['range'], dtype=torch.float32)
        self.velocity = torch.tensor(data_dict['velocity'], dtype=torch.float32)
        self.doppler = torch.tensor(data_dict['doppler'], dtype=torch.float32)
        
    def __len__(self):
        return len(self.Y_obs)
        
    def __getitem__(self, idx):
        return {
            'Y_obs': self.Y_obs[idx],
            'H_c': self.H_c[idx],
            'pilot_mask': self.pilot_mask,
            'range': self.range[idx],
            'velocity': self.velocity[idx],
            'doppler': self.doppler[idx]
        }
