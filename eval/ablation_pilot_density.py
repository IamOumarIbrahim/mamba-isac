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

def run_pilot_density_ablation(config_path: str = "configs/default_config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    seeds = [42, 43, 44]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    mamba_ckpt = "checkpoints/best_mamba_isac.pt"
    trans_ckpt = "checkpoints/best_transformer_isac.pt"
    
    pilot_spacings = [2, 4, 8, 16]
    
    results = []
    print("\nStarting Ablation 3: Pilot Density Sweeps (spacing in [2, 4, 8, 16])...")
    
    for spacing in pilot_spacings:
        config_p = yaml.safe_load(open(config_path))
        config_p['pilots']['pilot_spacing'] = spacing
        
        seed_results = {
            'LMMSE_NMSE': [],
            'Trans_NMSE': [],
            'Mamba_NMSE': []
        }
        
        for seed in seeds:
            set_seed(seed)
            test_dict = generate_isac_samples(config_p, num_samples=30, seed=seed)
            test_ds = ISACDataset(test_dict)
            
            # 1. LMMSE
            lmmse = LMMSEEstimator(
                num_subcarriers=config_p['dataset']['num_subcarriers'],
                num_time_slots=config_p['dataset']['num_time_slots'],
                pilot_spacing=spacing
            )
            H_lmmse = lmmse.estimate_comm_channel(test_dict['Y_obs'], snr_db=15.0)
            nmse_lmmse = compute_nmse_db(H_lmmse, test_dict['H_c'])
            
            # 2. Transformer
            trans_model = TransformerISAC(config_p).to(device)
            if os.path.exists(trans_ckpt):
                ckpt = torch.load(trans_ckpt, map_location=device, weights_only=False)
                trans_model.load_state_dict(ckpt['model_state_dict'], strict=False)
            trans_model.eval()
            
            Y_obs = test_ds.Y_obs.to(device)
            pilot_mask = test_ds.pilot_mask.to(device)
            
            pilot_mask_expanded = pilot_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
            Y_obs_masked = Y_obs.clone()
            Y_obs_masked[~pilot_mask_expanded.expand_as(Y_obs_masked)] = 0.0

            with torch.no_grad():
                H_c_hat_trans, _, _ = trans_model(Y_obs_masked)
                
            H_c_true = test_ds.H_c.numpy()
            H_c_hat_trans_np = H_c_hat_trans.cpu().numpy()
            nmse_trans = compute_nmse_db(
                H_c_hat_trans_np[:, 0] + 1j * H_c_hat_trans_np[:, 1],
                H_c_true[:, 0] + 1j * H_c_true[:, 1]
            )
            
            # 3. Mamba-ISAC
            mamba_model = MambaISAC(config_p).to(device)
            if os.path.exists(mamba_ckpt):
                ckpt = torch.load(mamba_ckpt, map_location=device, weights_only=False)
                mamba_model.load_state_dict(ckpt['model_state_dict'], strict=False)
            mamba_model.eval()
            with torch.no_grad():
                H_c_hat_mamba, _, _ = mamba_model(Y_obs_masked)
                
            H_c_hat_mamba_np = H_c_hat_mamba.cpu().numpy()
            nmse_mamba = compute_nmse_db(
                H_c_hat_mamba_np[:, 0] + 1j * H_c_hat_mamba_np[:, 1],
                H_c_true[:, 0] + 1j * H_c_true[:, 1]
            )
            
            seed_results['LMMSE_NMSE'].append(nmse_lmmse)
            seed_results['Trans_NMSE'].append(nmse_trans)
            seed_results['Mamba_NMSE'].append(nmse_mamba)
            
        results.append({
            'pilot_spacing': spacing,
            'pilot_overhead_pct': pilot_overhead_pct,
            'LMMSE_NMSE_mean': np.mean(seed_results['LMMSE_NMSE']),
            'LMMSE_NMSE_std': np.std(seed_results['LMMSE_NMSE']),
            'Transformer_NMSE_mean': np.mean(seed_results['Trans_NMSE']),
            'Transformer_NMSE_std': np.std(seed_results['Trans_NMSE']),
            'Mamba_NMSE_mean': np.mean(seed_results['Mamba_NMSE']),
            'Mamba_NMSE_std': np.std(seed_results['Mamba_NMSE'])
        })
        
        print(f"Spacing: 1/{spacing:<2d} ({pilot_overhead_pct:5.1f}% overhead) | LMMSE: {np.mean(seed_results['LMMSE_NMSE']):.2f} dB | Trans: {np.mean(seed_results['Trans_NMSE']):.2f} dB | Mamba: {np.mean(seed_results['Mamba_NMSE']):.2f} dB")
        
    os.makedirs("results", exist_ok=True)
    csv_path = "results/ablation_pilot_density.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
        
    print(f"Pilot density ablation results saved to '{csv_path}'.")
    return results

if __name__ == "__main__":
    run_pilot_density_ablation()
