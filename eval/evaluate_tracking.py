import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yaml
import torch
import numpy as np

from data.dataset import generate_isac_samples
from baselines.lmmse import LMMSEEstimator
from baselines.kalman_tracker import KalmanFilterTracker
from utils.seed import set_seed
from eval.metrics import compute_rmse

def run_tracking_evaluation(config_path: str = "configs/default_config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    set_seed(42)
    # Force sequential data generation
    config['dataset']['sequential_trajectory'] = True
    config['dataset']['trajectory_dt'] = 0.1
    
    num_samples = 100
    test_dict = generate_isac_samples(config, num_samples=num_samples, seed=42)
    
    # 1. Base LMMSE estimates (single frame)
    lmmse = LMMSEEstimator(
        num_subcarriers=config['dataset']['num_subcarriers'],
        num_time_slots=config['dataset']['num_time_slots'],
        pilot_spacing=config['pilots']['pilot_spacing']
    )
    
    Y_obs = test_dict['Y_obs']
    H_lmmse = lmmse.estimate_comm_channel(Y_obs, snr_db=15.0)
    R_lmmse, nu_lmmse = lmmse.estimate_sensing_parameters(Y_obs, H_c_est=H_lmmse, snr_db=15.0)
    
    # Estimate velocity from Doppler
    fc = config['dataset']['carrier_frequency']
    c = 3.0e8
    V_lmmse = (nu_lmmse * c) / (2.0 * fc)
    
    # Ground truth
    R_true = test_dict['range']
    V_true = test_dict['velocity']
    
    # 2. Kalman Filter on top of LMMSE
    kf = KalmanFilterTracker(dt=0.1, process_noise_std=0.5, meas_noise_std=2.0)
    
    R_tracked = np.zeros(num_samples)
    V_tracked = np.zeros(num_samples)
    
    for i in range(num_samples):
        z = np.array([R_lmmse[i], V_lmmse[i]])
        x_filtered = kf.step(z)
        R_tracked[i] = x_filtered[0]
        V_tracked[i] = x_filtered[1]
        
    raw_range_rmse = compute_rmse(torch.tensor(R_lmmse), torch.tensor(R_true))
    raw_vel_rmse = compute_rmse(torch.tensor(V_lmmse), torch.tensor(V_true))
    
    tracked_range_rmse = compute_rmse(torch.tensor(R_tracked), torch.tensor(R_true))
    tracked_vel_rmse = compute_rmse(torch.tensor(V_tracked), torch.tensor(V_true))
    
    print("\n=======================================================")
    print("KALMAN FILTER TRACKING EVALUATION")
    print("=======================================================")
    print(f"Raw LMMSE Range RMSE:    {raw_range_rmse:.3f} m")
    print(f"Tracked Range RMSE:      {tracked_range_rmse:.3f} m")
    print(f"Raw LMMSE Velocity RMSE: {raw_vel_rmse:.3f} m/s")
    print(f"Tracked Velocity RMSE:   {tracked_vel_rmse:.3f} m/s")
    print("=======================================================\n")
    
if __name__ == "__main__":
    run_tracking_evaluation()
