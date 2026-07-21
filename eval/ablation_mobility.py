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
from eval.metrics import compute_nmse_db
from utils.seed import set_seed

def run_mobility_ablation(config_path: str = "configs/default_config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    mamba_ckpt = "checkpoints/best_mamba_isac.pt"
    trans_ckpt = "checkpoints/best_transformer_isac.pt"
    
    velocities_ms = [5.0, 15.0, 30.0, 60.0, 120.0]
    
    results = []
    print("\nStarting Ablation 2: Mobility / Velocity Sweeps (v in [5, 15, 30, 60, 120] m/s)...")
    
    for v in velocities_ms:
        config_v = yaml.safe_load(open(config_path))
        config_v['comm_channel']['user_velocity'] = v
        
        test_dict = generate_isac_samples(config_v, num_samples=30, seed=42)
        test_ds = ISACDataset(test_dict)
        
        # 1. LMMSE
        lmmse = LMMSEEstimator(
            num_subcarriers=config_v['dataset']['num_subcarriers'],
            num_time_slots=config_v['dataset']['num_time_slots'],
            pilot_spacing=config_v['pilots']['pilot_spacing']
        )
        H_lmmse = lmmse.estimate_comm_channel(test_dict['Y_obs'], snr_db=15.0)
        nmse_lmmse = compute_nmse_db(H_lmmse, test_dict['H_c'])
        
        # 2. Transformer
        trans_model = TransformerISAC(config_v).to(device)
        if os.path.exists(trans_ckpt):
            ckpt = torch.load(trans_ckpt, map_location=device)
            trans_model.load_state_dict(ckpt['model_state_dict'], strict=False)
        trans_model.eval()
        Y_obs = test_ds.Y_obs.to(device)
        with torch.no_grad():
            H_c_hat_trans, _, _ = trans_model(Y_obs)
            
        H_c_true = test_ds.H_c.numpy()
        H_c_hat_trans_np = H_c_hat_trans.cpu().numpy()
        nmse_trans = compute_nmse_db(
            H_c_hat_trans_np[:, 0] + 1j * H_c_hat_trans_np[:, 1],
            H_c_true[:, 0] + 1j * H_c_true[:, 1]
        )
        
        # 3. Mamba-ISAC
        mamba_model = MambaISAC(config_v).to(device)
        if os.path.exists(mamba_ckpt):
            ckpt = torch.load(mamba_ckpt, map_location=device)
            mamba_model.load_state_dict(ckpt['model_state_dict'], strict=False)
        mamba_model.eval()
        with torch.no_grad():
            H_c_hat_mamba, _, _ = mamba_model(Y_obs)
            
        H_c_hat_mamba_np = H_c_hat_mamba.cpu().numpy()
        nmse_mamba = compute_nmse_db(
            H_c_hat_mamba_np[:, 0] + 1j * H_c_hat_mamba_np[:, 1],
            H_c_true[:, 0] + 1j * H_c_true[:, 1]
        )
        
        results.append({
            'velocity_ms': v,
            'velocity_kmh': v * 3.6,
            'LMMSE_NMSE': nmse_lmmse,
            'Transformer_NMSE': nmse_trans,
            'Mamba_NMSE': nmse_mamba
        })
        
        print(f"Velocity: {v:5.1f} m/s ({v*3.6:5.1f} km/h) | LMMSE NMSE: {nmse_lmmse:.2f} dB | Trans NMSE: {nmse_trans:.2f} dB | Mamba NMSE: {nmse_mamba:.2f} dB")
        
    os.makedirs("results", exist_ok=True)
    csv_path = "results/ablation_mobility.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
        
    print(f"Mobility ablation results saved to '{csv_path}'.")
    return results

if __name__ == "__main__":
    run_mobility_ablation()
