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
        
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    mamba_ckpt = "checkpoints/best_mamba_isac.pt"
    trans_ckpt = "checkpoints/best_transformer_isac.pt"
    
    seq_lengths = [4, 8, 16, 32, 64]
    
    results = []
    print("\nStarting Ablation 1: Sequence-Length Scaling Sweeps (T in [4, 8, 16, 32, 64])...")
    
    for T_len in seq_lengths:
        config_t = yaml.safe_load(open(config_path))
        config_t['dataset']['num_time_slots'] = T_len
        
        test_dict = generate_isac_samples(config_t, num_samples=30, seed=42)
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
        lat_trans, _ = measure_inference_latency(lambda x: trans_model(x), Y_obs[:1])
        
        # 3. Mamba-ISAC
        mamba_model = MambaISAC(config_t).to(device)
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
        lat_mamba, _ = measure_inference_latency(lambda x: mamba_model(x), Y_obs[:1])
        
        results.append({
            'T': T_len,
            'LMMSE_NMSE': nmse_lmmse, 'LMMSE_Lat_ms': lat_lmmse,
            'Transformer_NMSE': nmse_trans, 'Transformer_Lat_ms': lat_trans,
            'Mamba_NMSE': nmse_mamba, 'Mamba_Lat_ms': lat_mamba
        })
        
        print(f"T={T_len:2d} | LMMSE NMSE: {nmse_lmmse:.2f} dB | Trans NMSE: {nmse_trans:.2f} dB | Mamba NMSE: {nmse_mamba:.2f} dB")
        
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
