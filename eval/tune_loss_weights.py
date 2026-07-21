import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import csv
import yaml
import torch
import numpy as np

from data.dataset import generate_isac_samples, ISACDataset
from models.mamba_isac import MambaISAC
from models.loss import JointISACLoss
from utils.seed import set_seed

def tune_loss_weights(config_path: str = "configs/default_config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    val_dict = generate_isac_samples(config, num_samples=30, seed=999)
    val_ds = ISACDataset(val_dict)
    
    weight_candidates = [
        (1.0, 0.1, 0.1),
        (1.0, 0.5, 0.5),
        (1.0, 1.0, 1.0),
        (0.5, 1.0, 1.0)
    ]
    
    model = MambaISAC(config).to(device)
    model.eval()
    
    results = []
    print("\nStarting Validation Loss Weight Tuning Grid Search...")
    
    best_weights = None
    best_val_score = float("inf")
    
    with torch.no_grad():
        for (lc, ls, ld) in weight_candidates:
            loss_fn = JointISACLoss(lambda_c=lc, lambda_s=ls, lambda_d=ld).to(device)
            
            Y_obs = val_ds.Y_obs.to(device)
            H_c_true = val_ds.H_c.to(device)
            R_true = val_ds.range.to(device)
            nu_s_true = val_ds.doppler_s.to(device)
            
            H_c_hat, R_hat, nu_s_hat = model(Y_obs)
            total_loss, loss_dict = loss_fn(H_c_hat, H_c_true, R_hat, R_true, nu_s_hat, nu_s_true)
            
            score = loss_dict['loss_comm_nmse'] + loss_dict['loss_range_norm_mse'] + loss_dict['loss_doppler_norm_mse']
            
            results.append({
                'lambda_c': lc, 'lambda_s': ls, 'lambda_d': ld,
                'val_total_loss': total_loss.item(),
                'val_comm_nmse': loss_dict['loss_comm_nmse'],
                'val_range_norm_mse': loss_dict['loss_range_norm_mse'],
                'val_doppler_norm_mse': loss_dict['loss_doppler_norm_mse'],
                'composite_score': score
            })
            
            if score < best_val_score:
                best_val_score = score
                best_weights = (lc, ls, ld)
                
            print(f"Weights ({lc:.1f}, {ls:.1f}, {ld:.1f}) | Val Total Loss: {total_loss.item():.4f} | Composite Score: {score:.4f}")
            
    print(f"\nOptimal Loss Weights Selected via Validation Set: lambda_c={best_weights[0]}, lambda_s={best_weights[1]}, lambda_d={best_weights[2]}")
    
    os.makedirs("results", exist_ok=True)
    csv_path = "results/loss_weight_tuning.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
        
    print(f"Loss weight tuning results saved to '{csv_path}'.")
    return best_weights

if __name__ == "__main__":
    tune_loss_weights()
