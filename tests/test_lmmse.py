import pytest
import numpy as np
from baselines.lmmse import LMMSEEstimator
from data.comm_channel import CommunicationChannelGenerator
from data.sensing_channel import SensingChannelGenerator
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
    assert nmse_lmmse < -10.0

def test_lmmse_sensing_recovery_with_cancellation():
    set_seed(42)
    Nc, T, Nt, Nr = 64, 16, 4, 4
    lmmse = LMMSEEstimator(num_subcarriers=Nc, num_time_slots=T)
    sens_gen = SensingChannelGenerator(num_subcarriers=Nc, num_time_slots=T, num_tx_antennas=Nt, num_rx_antennas=Nr)
    comm_gen = CommunicationChannelGenerator(num_subcarriers=Nc, num_time_slots=T, num_tx_antennas=Nt, num_rx_antennas=Nr)
    
    target_r = 55.0
    target_v = 15.0
    
    H_c, _ = comm_gen.generate_channel(batch_size=1, snr_db=None)
    y_s, tau, nu_s = sens_gen.generate_echo(range_m=target_r, velocity_ms=target_v)
    
    Y_obs = (H_c + y_s) # Joint observation
    
    # Estimate comm channel first
    H_c_est = lmmse.estimate_comm_channel(Y_obs, snr_db=20.0)
    
    # Estimate sensing parameters after channel cancellation
    R_hat, nu_s_hat = lmmse.estimate_sensing_parameters(Y_obs, H_c_est=H_c_est)
    
    np.testing.assert_allclose(R_hat[0], target_r, atol=5.0)
    np.testing.assert_allclose(nu_s_hat[0], nu_s, atol=20.0)
