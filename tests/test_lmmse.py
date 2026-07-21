import pytest
import numpy as np
from baselines.lmmse import LMMSEEstimator
from data.comm_channel import CommunicationChannelGenerator
from utils.seed import set_seed

def test_lmmse_estimation_and_nmse():
    set_seed(42)
    Nc, T, Nt, Nr = 32, 8, 2, 2
    gen = CommunicationChannelGenerator(
        num_subcarriers=Nc,
        num_time_slots=T,
        num_tx_antennas=Nt,
        num_rx_antennas=Nr,
        k_factor_db=10.0,
        user_velocity=10.0
    )
    
    H_c, H_c_noisy = gen.generate_channel(batch_size=100, snr_db=15.0)
    lmmse = LMMSEEstimator(num_subcarriers=Nc, num_time_slots=T, pilot_spacing=4)
    
    H_flat = H_c.transpose(0, 1, 2, 4, 3).reshape(-1, Nc)
    R_hh = (H_flat.conj().T @ H_flat) / H_flat.shape[0]
    
    H_lmmse = lmmse.estimate_comm_channel(H_c_noisy, snr_db=15.0, R_hh=R_hh)
    assert H_lmmse.shape == H_c.shape
    
    nmse_lmmse = LMMSEEstimator.compute_nmse(H_lmmse, H_c)
    
    # NMSE should be low at 15 dB SNR (< -10 dB)
    assert nmse_lmmse < -10.0
