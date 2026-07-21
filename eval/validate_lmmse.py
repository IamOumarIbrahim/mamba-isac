import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
from data.comm_channel import CommunicationChannelGenerator
from baselines.lmmse import LMMSEEstimator
from utils.seed import set_seed

def run_lmmse_snr_sweep():
    set_seed(42)
    snr_db_list = [-5, 0, 5, 10, 15, 20, 25, 30]
    num_samples = 500
    
    Nc, T, Nt, Nr = 64, 16, 4, 4
    pilot_spacing = 4
    
    comm_gen = CommunicationChannelGenerator(
        num_subcarriers=Nc,
        num_time_slots=T,
        num_tx_antennas=Nt,
        num_rx_antennas=Nr,
        k_factor_db=10.0,
        user_velocity=15.0
    )
    
    lmmse = LMMSEEstimator(num_subcarriers=Nc, num_time_slots=T, pilot_spacing=pilot_spacing)
    
    empirical_nmse_db = []
    
    # Generate channels to estimate empirical mean and covariance
    H_c_all, _ = comm_gen.generate_channel(batch_size=1000, snr_db=None) # (1000, Nr, Nt, Nc, T)
    mu_h = np.mean(H_c_all, axis=0) # (Nr, Nt, Nc, T)
    
    # Frequency domain covariance C_hh
    H_flat = (H_c_all - mu_h).transpose(0, 1, 2, 4, 3).reshape(-1, Nc)
    C_hh = (H_flat.conj().T @ H_flat) / H_flat.shape[0]
    
    for snr_db in snr_db_list:
        H_c, H_c_noisy = comm_gen.generate_channel(batch_size=num_samples, snr_db=snr_db)
        
        H_lmmse = lmmse.estimate_comm_channel(H_c_noisy, snr_db=snr_db, R_hh=C_hh, mu_h=mu_h)
        nmse_db = LMMSEEstimator.compute_nmse(H_lmmse, H_c)
        empirical_nmse_db.append(nmse_db)
        
        print(f"SNR: {snr_db:2d} dB | Empirical LMMSE NMSE: {nmse_db:6.2f} dB")
        
    # Check Phase 1 exit criterion: LMMSE NMSE decreases monotonically as SNR increases
    assert np.all(np.diff(empirical_nmse_db) < 0), "LMMSE NMSE must decrease monotonically with SNR!"
    print("\nPhase 1 Exit Criterion PASSED: Monotonic LMMSE NMSE improvement confirmed!")
    return snr_db_list, empirical_nmse_db

if __name__ == "__main__":
    run_lmmse_snr_sweep()
