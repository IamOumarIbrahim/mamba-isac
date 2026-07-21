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
from eval.metrics import compute_nmse_db, compute_rmse, measure_inference_latency
from utils.flops import estimate_flops, count_parameters
from utils.seed import set_seed

def run_main_evaluation(config_path: str = "configs/default_config.yaml", seeds: list = [42, 43, 44]):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    mamba_ckpt = "checkpoints/best_mamba_isac.pt"
    trans_ckpt = "checkpoints/best_transformer_isac.pt"
    
    # Pre-generate channel samples to compute empirical mean and covariance for Centered LMMSE
    comm_gen = CommunicationChannelGenerator(
        num_subcarriers=config['dataset']['num_subcarriers'],
        num_time_slots=config['dataset']['num_time_slots'],
        num_tx_antennas=config['dataset']['num_tx_antennas'],
        num_rx_antennas=config['dataset']['num_rx_antennas'],
        k_factor_db=config['comm_channel']['k_factor_db'],
        user_velocity=config['comm_channel']['user_velocity']
    )
    H_c_all, _ = comm_gen.generate_channel(batch_size=1000, snr_db=None) # (1000, Nr, Nt, Nc, T)
    mu_h = np.mean(H_c_all, axis=0) # (Nr, Nt, Nc, T)
    Nc = config['dataset']['num_subcarriers']
    H_flat = (H_c_all - mu_h).transpose(0, 1, 2, 4, 3).reshape(-1, Nc)
    C_hh = (H_flat.conj().T @ H_flat) / H_flat.shape[0]

    all_seed_results = {
        'LMMSE': {'nmse': [], 'r_rmse': []},
        'Transformer': {'nmse': [], 'r_rmse': []},
        'Mamba-ISAC': {'nmse': [], 'r_rmse': []}
    }

    for seed in seeds:
        set_seed(seed)
        test_dict = generate_isac_samples(config, num_samples=100, seed=seed)
        test_ds = ISACDataset(test_dict)
        
        # 1. Centered LMMSE
        lmmse = LMMSEEstimator(
            num_subcarriers=config['dataset']['num_subcarriers'],
            num_time_slots=config['dataset']['num_time_slots'],
            pilot_spacing=config['pilots']['pilot_spacing']
        )
        
        Y_obs_complex = test_dict['Y_obs'] # (100, Nr, Nt, Nc, T)
        H_lmmse = lmmse.estimate_comm_channel(Y_obs_complex, snr_db=15.0, R_hh=C_hh, mu_h=mu_h)
        R_lmmse, nu_lmmse = lmmse.estimate_sensing_parameters(Y_obs_complex, H_c_est=H_lmmse, snr_db=15.0)
        
        nmse_lmmse = compute_nmse_db(H_lmmse, test_dict['H_c'])
        r_rmse_lmmse = compute_rmse(R_lmmse, test_dict['range'])
        
        all_seed_results['LMMSE']['nmse'].append(nmse_lmmse)
        all_seed_results['LMMSE']['r_rmse'].append(r_rmse_lmmse)
        
        # 2. Transformer
        trans_model = TransformerISAC(config).to(device)
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
        
        all_seed_results['Transformer']['nmse'].append(nmse_trans)
        all_seed_results['Transformer']['r_rmse'].append(r_rmse_trans)
        
        # 3. Mamba-ISAC
        mamba_model = MambaISAC(config).to(device)
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
        
        all_seed_results['Mamba-ISAC']['nmse'].append(nmse_mamba)
        all_seed_results['Mamba-ISAC']['r_rmse'].append(r_rmse_mamba)

    # Compute FLOPs and parameters
    dummy_input = torch.randn(1, 2, config['dataset']['num_rx_antennas'], config['dataset']['num_tx_antennas'], config['dataset']['num_subcarriers'], config['dataset']['num_time_slots']).to(device)
    
    flops_trans = estimate_flops(trans_model, dummy_input)
    flops_mamba = estimate_flops(mamba_model, dummy_input)
    flops_lmmse = 2 * (config['dataset']['num_subcarriers'] ** 3)

    summary = []
    print("\n" + "="*80)
    print("TABLE I: MAIN BENCHMARK ACCURACY & COMPUTE (15 dB SNR, Mean ± Std, 3 Seeds)")
    print("="*80)
    print(f"{'Method':<20} | {'Comm NMSE (dB)':<18} | {'Range RMSE (m)':<18} | {'FLOPs':<12} | {'Params':<10}")
    print("-" * 80)

    for method in ['LMMSE', 'Transformer', 'Mamba-ISAC']:
        nmse_m, nmse_s = np.mean(all_seed_results[method]['nmse']), np.std(all_seed_results[method]['nmse'])
        r_m, r_s = np.mean(all_seed_results[method]['r_rmse']), np.std(all_seed_results[method]['r_rmse'])
        
        flops = flops_lmmse if method == 'LMMSE' else (flops_trans if method == 'Transformer' else flops_mamba)
        params = 0 if method == 'LMMSE' else (count_parameters(trans_model) if method == 'Transformer' else count_parameters(mamba_model))
        
        summary.append({
            'Method': method,
            'NMSE_mean': nmse_m, 'NMSE_std': nmse_s,
            'Range_RMSE_mean': r_m, 'Range_RMSE_std': r_s,
            'FLOPs': flops,
            'Parameters': params
        })
        
        print(f"{method:<20} | {nmse_m:6.2f} ± {nmse_s:4.2f}     | {r_m:6.2f} ± {r_s:4.2f}     | {flops:<12d} | {params:<10d}")

    print("="*80 + "\n")
    
    os.makedirs("results", exist_ok=True)
    with open("results/main_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
        
    print("Main results saved to 'results/main_results.csv'.")
    return summary

if __name__ == "__main__":
    run_main_evaluation()
