import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import csv
import yaml
import torch
import numpy as np

from data.comm_channel import CommunicationChannelGenerator
from data.dataset import generate_isac_samples, ISACDataset
from models.mamba_isac import MambaISAC
from baselines.transformer_isac import TransformerISAC
from baselines.lmmse import LMMSEEstimator
from eval.metrics import compute_nmse_db, compute_rmse
from utils.seed import set_seed

def run_clutter_ablation(config_path: str = "configs/default_config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    seeds = [42, 43, 44]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    mamba_ckpt = "checkpoints/best_mamba_isac.pt"
    trans_ckpt = "checkpoints/best_transformer_isac.pt"
    
    clutter_modes = [False, True]
    results = []
    
    print("\nStarting Ablation 4: Clutter Robustness Sweeps (Clutter: [False, True])...")
    
    # Pre-generate channel samples to compute empirical covariance for Centered LMMSE
    comm_gen = CommunicationChannelGenerator(
        num_subcarriers=config['dataset']['num_subcarriers'],
        num_time_slots=config['dataset']['num_time_slots'],
        num_tx_antennas=config['dataset']['num_tx_antennas'],
        num_rx_antennas=config['dataset']['num_rx_antennas'],
        k_factor_db=config['comm_channel']['k_factor_db'],
        user_velocity=config['comm_channel']['user_velocity']
    )
    H_c_all, _ = comm_gen.generate_channel(batch_size=1000, snr_db=None)
    mu_h = np.mean(H_c_all, axis=0)
    Nc = config['dataset']['num_subcarriers']
    H_flat = (H_c_all - mu_h).transpose(0, 1, 2, 4, 3).reshape(-1, Nc)
    C_hh = (H_flat.conj().T @ H_flat) / H_flat.shape[0]

    for clutter_flag in clutter_modes:
        config_c = yaml.safe_load(open(config_path))
        config_c['sensing_channel']['include_clutter'] = clutter_flag
        
        seed_results = {
            'LMMSE_NMSE': [], 'LMMSE_RMSE': [],
            'Trans_NMSE': [], 'Trans_RMSE': [],
            'Mamba_NMSE': [], 'Mamba_RMSE': []
        }
        
        for seed in seeds:
            set_seed(seed)
            test_dict = generate_isac_samples(config_c, num_samples=50, seed=seed)
            test_ds = ISACDataset(test_dict)
            
            # 1. LMMSE
            lmmse = LMMSEEstimator(
                num_subcarriers=config_c['dataset']['num_subcarriers'],
                num_time_slots=config_c['dataset']['num_time_slots'],
                pilot_spacing=config_c['pilots']['pilot_spacing']
            )
            H_lmmse = lmmse.estimate_comm_channel(test_dict['Y_obs'], snr_db=15.0, R_hh=C_hh, mu_h=mu_h)
            R_lmmse, _ = lmmse.estimate_sensing_parameters(test_dict['Y_obs'], H_c_est=H_lmmse, snr_db=15.0)
            
            nmse_lmmse = compute_nmse_db(H_lmmse, test_dict['H_c'])
            r_rmse_lmmse = compute_rmse(R_lmmse, test_dict['range'])
            
            # 2. Transformer
            trans_model = TransformerISAC(config_c).to(device)
            if os.path.exists(trans_ckpt):
                ckpt = torch.load(trans_ckpt, map_location=device, weights_only=False)
                trans_model.load_state_dict(ckpt['model_state_dict'], strict=False)
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
            mamba_model = MambaISAC(config_c).to(device)
            if os.path.exists(mamba_ckpt):
                ckpt = torch.load(mamba_ckpt, map_location=device, weights_only=False)
                mamba_model.load_state_dict(ckpt['model_state_dict'], strict=False)
            mamba_model.eval()
            
            with torch.no_grad():
                H_c_hat_mamba, R_mamba, _ = mamba_model(Y_obs)
                
            H_c_hat_mamba_np = H_c_hat_mamba.cpu().numpy()
            nmse_mamba = compute_nmse_db(
                H_c_hat_mamba_np[:, 0] + 1j * H_c_hat_mamba_np[:, 1],
                H_c_true[:, 0] + 1j * H_c_true[:, 1]
            )
            r_rmse_mamba = compute_rmse(R_mamba, test_ds.range)
            
            seed_results['LMMSE_NMSE'].append(nmse_lmmse)
            seed_results['LMMSE_RMSE'].append(r_rmse_lmmse)
            seed_results['Trans_NMSE'].append(nmse_trans)
            seed_results['Trans_RMSE'].append(r_rmse_trans)
            seed_results['Mamba_NMSE'].append(nmse_mamba)
            seed_results['Mamba_RMSE'].append(r_rmse_mamba)
            
        results.append({
            'include_clutter': clutter_flag,
            'LMMSE_NMSE_mean': np.mean(seed_results['LMMSE_NMSE']),
            'LMMSE_NMSE_std': np.std(seed_results['LMMSE_NMSE']),
            'LMMSE_Range_RMSE_mean': np.mean(seed_results['LMMSE_RMSE']),
            'LMMSE_Range_RMSE_std': np.std(seed_results['LMMSE_RMSE']),
            'Transformer_NMSE_mean': np.mean(seed_results['Trans_NMSE']),
            'Transformer_NMSE_std': np.std(seed_results['Trans_NMSE']),
            'Transformer_Range_RMSE_mean': np.mean(seed_results['Trans_RMSE']),
            'Transformer_Range_RMSE_std': np.std(seed_results['Trans_RMSE']),
            'Mamba_NMSE_mean': np.mean(seed_results['Mamba_NMSE']),
            'Mamba_NMSE_std': np.std(seed_results['Mamba_NMSE']),
            'Mamba_Range_RMSE_mean': np.mean(seed_results['Mamba_RMSE']),
            'Mamba_Range_RMSE_std': np.std(seed_results['Mamba_RMSE'])
        })
        
        mode_str = "Clutter ACTIVE" if clutter_flag else "Clean"
        print(f"[{mode_str:<14}] LMMSE Range RMSE: {np.mean(seed_results['LMMSE_RMSE']):5.2f}m | Trans Range RMSE: {np.mean(seed_results['Trans_RMSE']):5.2f}m | Mamba Range RMSE: {np.mean(seed_results['Mamba_RMSE']):5.2f}m")
        
    os.makedirs("results", exist_ok=True)
    csv_path = "results/ablation_clutter.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
        
    print(f"Clutter ablation results saved to '{csv_path}'.")
    return results

if __name__ == "__main__":
    run_clutter_ablation()
