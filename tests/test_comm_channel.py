import pytest
import numpy as np
import scipy.special as sp
from data.comm_channel import CommunicationChannelGenerator
from utils.seed import set_seed

def test_jakes_autocorrelation():
    set_seed(42)
    T = 32
    v = 30.0 # m/s
    gen = CommunicationChannelGenerator(
        num_subcarriers=1,
        num_time_slots=T,
        num_tx_antennas=1,
        num_rx_antennas=1,
        k_factor_db=-100.0, # Pure NLoS Rayleigh
        user_velocity=v
    )
    
    # Generate large batch of NLoS fading sequences
    batch_size = 5000
    h_nlos = gen.generate_nlos_fading(batch_size=batch_size).squeeze() # (batch_size, T)
    
    # Empirical autocorrelation across time lag dt
    # R_emp[tau] = E[ h(t) * h*(t+tau) ]
    emp_autocorr = np.zeros(T)
    for tau in range(T):
        prod = h_nlos[:, :T-tau] * np.conj(h_nlos[:, tau:])
        emp_autocorr[tau] = np.mean(np.real(prod))
        
    # Theoretical Jakes autocorrelation
    t_indices = np.arange(T) * gen.Ts
    theo_autocorr = sp.jv(0, 2 * np.pi * gen.doppler_max * t_indices)
    
    # Check max difference between empirical and theoretical autocorrelation
    np.testing.assert_allclose(emp_autocorr[:10], theo_autocorr[:10], atol=0.05)

def test_rayleigh_limit_k_zero():
    set_seed(42)
    gen = CommunicationChannelGenerator(
        num_subcarriers=16,
        num_time_slots=10,
        num_tx_antennas=2,
        num_rx_antennas=2,
        k_factor_db=-100.0, # Linear K approx 0 -> Rayleigh
        user_velocity=10.0
    )
    
    H_c, _ = gen.generate_channel(batch_size=1000)
    
    # For Rayleigh fading with unit variance E[|H|^2] = 1.0
    power = np.mean(np.abs(H_c) ** 2)
    np.testing.assert_allclose(power, 1.0, atol=0.05)
    
    # Mean of channel coefficient should be near zero (no deterministic LoS component)
    mean_val = np.mean(H_c)
    np.testing.assert_allclose(np.abs(mean_val), 0.0, atol=0.05)
