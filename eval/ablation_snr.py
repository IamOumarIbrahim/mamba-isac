import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import csv
import yaml
import torch
import numpy as np

from data.dataset import generate_isac_samples, ISACDataset
from models.mamba_isac import MambaISAC
from baselines.transformer_isac import TransformerISAC
from baselines.lmmse import LMMSEEstimator
from eval.metrics import compute_nmse_db, compute_rmse
from utils.seed import set_seed

def run_snr_ablation(config_path: str = "configs/default_config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    snr_list_db = [-10, -5, 0, 5, 10, 15, 20, 25, 30]
    
    results = []
    print("\nStarting Ablation 4: Matched SNR Sweeps (-10 dB to +30 dB)...")
    
    for snr_db in snr_list_db:
        config_snr = yaml.safe_load(open(config_path))
        config_snr['comm_channel']['snr_db'] = float(snr_db)
        
        test_dict = generate_isac_samples(config_snr, num_samples=50, seed=42)
        test_ds = ISACDataset(test_dict)
        
        # 1. LMMSE
        lmmse = LMMSEEstimator(
            num_subcarriers=config_snr['dataset']['num_subcarriers'],
            num_time_slots=config_snr['dataset']['num_time_slots'],
            pilot_spacing=config_snr['pilots']['pilot_spacing']
        )
        H_lmmse = lmmse.estimate_comm_channel(test_dict['Y_obs'], snr_db=float(snr_db))
        R_lmmse, nu_lmmse = lmmse.estimate_sensing_parameters(test_dict['Y_obs'])
        
        nmse_lmmse = compute_nmse_db(H_lmmse, test_dict['H_c'])
        r_rmse_lmmse = compute_rmse(R_lmmse, test_dict['range'])
        
        # 2. Transformer
        trans_model = TransformerISAC(config_snr).to(device)
        trans_model.eval()
        Y_obs = test_ds.Y_obs.to(device)
        with torch.no_grad():
            H_c_hat_trans, R_trans, _ = trans_model(Y_obs)
            
        H_c_true = test_ds.H_c.numpy()
        H_c_hat_trans_np = H_c_hat_trans.cpu().numpy()
        nmse_trans = compute_nmse_db(
            H_c_hat_trans_np[:, 0] + 1j * H_c_hat_trans_np[:, 1],
            H_c_true[:, 0] + 1j * H_c_true[:, 1]
        )
        r_rmse_trans = compute_rmse(R_trans, test_ds.range)
        
        # 3. Mamba-ISAC
        mamba_model = MambaISAC(config_snr).to(device)
        mamba_model.eval()
        with torch.no_grad():
            H_c_hat_mamba, R_mamba, _ = mamba_model(Y_obs)
            
        H_c_hat_mamba_np = H_c_hat_mamba.cpu().numpy()
        nmse_mamba = compute_nmse_db(
            H_c_hat_mamba_np[:, 0] + 1j * H_c_hat_mamba_np[:, 1],
            H_c_true[:, 0] + 1j * H_c_true[:, 1]
        )
        r_rmse_mamba = compute_rmse(R_mamba, test_ds.range)
        
        results.append({
            'SNR_dB': snr_db,
            'LMMSE_NMSE': nmse_lmmse, 'LMMSE_Range_RMSE': r_rmse_lmmse,
            'Transformer_NMSE': nmse_trans, 'Transformer_Range_RMSE': r_rmse_trans,
            'Mamba_NMSE': nmse_mamba, 'Mamba_Range_RMSE': r_rmse_mamba
        })
        
        print(f"SNR: {snr_db:3d} dB | LMMSE: {nmse_lmmse:6.2f} dB | Trans: {nmse_trans:6.2f} dB | Mamba: {nmse_mamba:6.2f} dB")
        
    os.makedirs("results", exist_ok=True)
    csv_path = "results/ablation_snr.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
        
    print(f"SNR ablation results saved to '{csv_path}'.")
    return results

if __name__ == "__main__":
    run_snr_ablation()
