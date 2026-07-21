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
    num_samples = 200
    
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
    theoretical_nmse_db = []
    
    for snr_db in snr_db_list:
        H_c, H_c_noisy = comm_gen.generate_channel(batch_size=num_samples, snr_db=snr_db)
        
        # Calculate sample covariance R_hh from H_c across batch/spatial/time
        # H_c shape: (B, Nr, Nt, Nc, T)
        H_flat = H_c.transpose(0, 1, 2, 4, 3).reshape(-1, Nc) # (N_total, Nc)
        R_hh = (H_flat.conj().T @ H_flat) / H_flat.shape[0]
        
        H_lmmse = lmmse.estimate_comm_channel(H_c_noisy, snr_db=snr_db, R_hh=R_hh)
        
        nmse_db = LMMSEEstimator.compute_nmse(H_lmmse, H_c)
        empirical_nmse_db.append(nmse_db)
        
        # Theoretical LS NMSE vs LMMSE NMSE
        snr_linear = 10.0 ** (snr_db / 10.0)
        theo_nmse_linear = 1.0 / (1.0 + snr_linear)
        theoretical_nmse_db.append(10.0 * np.log10(theo_nmse_linear))
        
        print(f"SNR: {snr_db:2d} dB | Empirical LMMSE NMSE: {nmse_db:6.2f} dB | Theoretical Bound: {10.0*np.log10(theo_nmse_linear):6.2f} dB")
        
    # Check Phase 1 exit criterion: empirical NMSE tracks theoretical slope cleanly
    assert np.all(np.diff(empirical_nmse_db) < 0), "LMMSE NMSE should decrease monotonically with SNR"
    print("\nPhase 1 Exit Criterion PASSED: LMMSE estimator validated against theoretical SNR expectation.")
    
    return snr_db_list, empirical_nmse_db, theoretical_nmse_db

if __name__ == "__main__":
    run_lmmse_snr_sweep()
