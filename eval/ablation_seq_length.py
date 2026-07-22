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
from eval.metrics import compute_nmse_db, measure_inference_latency
from utils.seed import set_seed

def run_sequence_length_ablation(config_path: str = "configs/default_config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    seeds = [42, 43, 44]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    mamba_ckpt = "checkpoints/best_mamba_isac.pt"
    trans_ckpt = "checkpoints/best_transformer_isac.pt"
    
    seq_lengths = [4, 8, 16, 32, 64, 128, 256]
    
    results = []
    print("\nStarting Ablation 1: Sequence-Length Scaling Sweeps (T in [4, 8, 16, 32, 64, 128, 256])...")
    
    for T_len in seq_lengths:
        config_t = yaml.safe_load(open(config_path))
        config_t['dataset']['num_time_slots'] = T_len
        
        seed_results = {
            'LMMSE_NMSE': [], 'LMMSE_Lat': [],
            'Trans_NMSE': [], 'Trans_Lat': [],
            'Mamba_NMSE': [], 'Mamba_Lat': []
        }
        
        for seed in seeds:
            set_seed(seed)
            test_dict = generate_isac_samples(config_t, num_samples=30, seed=seed)
            test_ds = ISACDataset(test_dict)
            
            # 1. LMMSE
            lmmse = LMMSEEstimator(
                num_subcarriers=config_t['dataset']['num_subcarriers'],
                num_time_slots=T_len,
                pilot_spacing=config_t['pilots']['pilot_spacing']
            )
            H_lmmse = lmmse.estimate_comm_channel(test_dict['Y_obs'], snr_db=15.0)
            nmse_lmmse = compute_nmse_db(H_lmmse, test_dict['H_c'])
            lat_lmmse, _ = measure_inference_latency(lambda x: lmmse.estimate_comm_channel(x, snr_db=15.0), test_dict['Y_obs'][:1])
            
            # 2. Transformer
            trans_model = TransformerISAC(config_t).to(device)
            if os.path.exists(trans_ckpt):
                ckpt = torch.load(trans_ckpt, map_location=device, weights_only=False)
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
            lat_trans, _ = measure_inference_latency(lambda x: trans_model(x), Y_obs[:1])
            
            # 3. Mamba-ISAC
            mamba_model = MambaISAC(config_t).to(device)
            if os.path.exists(mamba_ckpt):
                ckpt = torch.load(mamba_ckpt, map_location=device, weights_only=False)
                mamba_model.load_state_dict(ckpt['model_state_dict'], strict=False)
            mamba_model.eval()
            with torch.no_grad():
                H_c_hat_mamba, _, _ = mamba_model(Y_obs)
                
            H_c_hat_mamba_np = H_c_hat_mamba.cpu().numpy()
            nmse_mamba = compute_nmse_db(
                H_c_hat_mamba_np[:, 0] + 1j * H_c_hat_mamba_np[:, 1],
                H_c_true[:, 0] + 1j * H_c_true[:, 1]
            )
            lat_mamba, _ = measure_inference_latency(lambda x: mamba_model(x), Y_obs[:1])
            
            seed_results['LMMSE_NMSE'].append(nmse_lmmse)
            seed_results['LMMSE_Lat'].append(lat_lmmse)
            seed_results['Trans_NMSE'].append(nmse_trans)
            seed_results['Trans_Lat'].append(lat_trans)
            seed_results['Mamba_NMSE'].append(nmse_mamba)
            seed_results['Mamba_Lat'].append(lat_mamba)
            
        results.append({
            'T': T_len,
            'LMMSE_NMSE_mean': np.mean(seed_results['LMMSE_NMSE']), 'LMMSE_NMSE_std': np.std(seed_results['LMMSE_NMSE']),
            'LMMSE_Lat_ms_mean': np.mean(seed_results['LMMSE_Lat']), 'LMMSE_Lat_ms_std': np.std(seed_results['LMMSE_Lat']),
            'Transformer_NMSE_mean': np.mean(seed_results['Trans_NMSE']), 'Transformer_NMSE_std': np.std(seed_results['Trans_NMSE']),
            'Transformer_Lat_ms_mean': np.mean(seed_results['Trans_Lat']), 'Transformer_Lat_ms_std': np.std(seed_results['Trans_Lat']),
            'Mamba_NMSE_mean': np.mean(seed_results['Mamba_NMSE']), 'Mamba_NMSE_std': np.std(seed_results['Mamba_NMSE']),
            'Mamba_Lat_ms_mean': np.mean(seed_results['Mamba_Lat']), 'Mamba_Lat_ms_std': np.std(seed_results['Mamba_Lat'])
        })
        
        print(f"T={T_len:3d} | LMMSE: {np.mean(seed_results['LMMSE_Lat']):.2f} ms | Trans: {np.mean(seed_results['Trans_Lat']):.2f} ms | Mamba: {np.mean(seed_results['Mamba_Lat']):.2f} ms")
        
    os.makedirs("results", exist_ok=True)
    csv_path = "results/ablation_seq_length.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
        
    print(f"Sequence-length ablation results saved to '{csv_path}'.")
    return results

if __name__ == "__main__":
    run_sequence_length_ablation()
