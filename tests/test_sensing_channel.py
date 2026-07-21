import pytest
import numpy as np
from data.sensing_channel import SensingChannelGenerator
from utils.seed import set_seed

def test_sensing_echo_ground_truth_recovery():
    set_seed(42)
    gen = SensingChannelGenerator(
        num_subcarriers=64,
        num_time_slots=32,
        num_tx_antennas=4,
        num_rx_antennas=4,
        carrier_frequency=28.0e9,
        subcarrier_spacing=120.0e3,
        symbol_duration=8.33e-6
    )
    
    target_range = 65.0 # meters
    target_velocity = 25.0 # m/s
    
    y_s, tau_gt, nu_s_gt = gen.generate_echo(
        range_m=target_range,
        velocity_ms=target_velocity,
        reflectivity=1.0 + 0.0j,
        azimuth_rad=0.1,
        snr_db=None # Noiseless
    )
    
    est_range, est_velocity = gen.recover_range_doppler_2d_fft(y_s)
    
    # Check recovered range and velocity match ground truth within FFT grid resolution
    np.testing.assert_allclose(est_range, target_range, atol=2.0)
    np.testing.assert_allclose(est_velocity, target_velocity, atol=2.0)
