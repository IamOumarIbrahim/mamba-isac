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
from eval.metrics import compute_nmse_db, compute_rmse, measure_inference_latency
from utils.flops import count_parameters, estimate_flops
from utils.seed import set_seed

def evaluate_all(config_path: str = "configs/default_config.yaml", seeds: list = [42, 43, 44]):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Checkpoints if trained, otherwise initialized
    mamba_ckpt = "checkpoints/best_mamba_isac.pt"
    
    results = {}
    
    for method_name in ["LMMSE", "Transformer", "Mamba-ISAC"]:
        nmse_list = []
        range_rmse_list = []
        doppler_rmse_list = []
        latency_list = []
        
        for seed in seeds:
            set_seed(seed)
            test_dict = generate_isac_samples(config, num_samples=100, seed=seed+100)
            test_ds = ISACDataset(test_dict)
            
            if method_name == "LMMSE":
                lmmse = LMMSEEstimator(
                    num_subcarriers=config['dataset']['num_subcarriers'],
                    num_time_slots=config['dataset']['num_time_slots'],
                    pilot_spacing=config['pilots']['pilot_spacing']
                )
                
                Y_obs_complex = test_dict['Y_obs']
                H_c_true = test_dict['H_c']
                
                H_lmmse = lmmse.estimate_comm_channel(Y_obs_complex, snr_db=config['comm_channel']['snr_db'])
                R_hat, nu_s_hat = lmmse.estimate_sensing_parameters(Y_obs_complex)
                
                nmse = compute_nmse_db(H_lmmse, H_c_true)
                r_rmse = compute_rmse(R_hat, test_dict['range'])
                d_rmse = compute_rmse(nu_s_hat, test_dict['doppler_s'])
                
                mean_lat, _ = measure_inference_latency(
                    lambda x: lmmse.estimate_comm_channel(x, snr_db=15.0),
                    Y_obs_complex[:1]
                )
                
                params = 0
                flops = config['dataset']['num_subcarriers'] ** 3
                
            elif method_name == "Transformer":
                model = TransformerISAC(config).to(device)
                model.eval()
                
                Y_obs = test_ds.Y_obs.to(device)
                with torch.no_grad():
                    H_c_hat, R_hat, nu_s_hat = model(Y_obs)
                    
                H_c_true = test_ds.H_c.numpy()
                H_c_hat_np = H_c_hat.cpu().numpy()
                # Reconstruct complex matrix
                H_c_hat_complex = H_c_hat_np[:, 0] + 1j * H_c_hat_np[:, 1]
                H_c_true_complex = H_c_true[:, 0] + 1j * H_c_true[:, 1]
                
                nmse = compute_nmse_db(H_c_hat_complex, H_c_true_complex)
                r_rmse = compute_rmse(R_hat, test_ds.range)
                d_rmse = compute_rmse(nu_s_hat, test_ds.doppler_s)
                
                mean_lat, _ = measure_inference_latency(lambda x: model(x), Y_obs[:1])
                params = count_parameters(model)
                flops = estimate_flops(model, Y_obs[:1])
                
            elif method_name == "Mamba-ISAC":
                model = MambaISAC(config).to(device)
                if os.path.exists(mamba_ckpt):
                    ckpt = torch.load(mamba_ckpt, map_location=device)
                    model.load_state_dict(ckpt['model_state_dict'])
                model.eval()
                
                Y_obs = test_ds.Y_obs.to(device)
                with torch.no_grad():
                    H_c_hat, R_hat, nu_s_hat = model(Y_obs)
                    
                H_c_true = test_ds.H_c.numpy()
                H_c_hat_np = H_c_hat.cpu().numpy()
                H_c_hat_complex = H_c_hat_np[:, 0] + 1j * H_c_hat_np[:, 1]
                H_c_true_complex = H_c_true[:, 0] + 1j * H_c_true[:, 1]
                
                nmse = compute_nmse_db(H_c_hat_complex, H_c_true_complex)
                r_rmse = compute_rmse(R_hat, test_ds.range)
                d_rmse = compute_rmse(nu_s_hat, test_ds.doppler_s)
                
                mean_lat, _ = measure_inference_latency(lambda x: model(x), Y_obs[:1])
                params = count_parameters(model)
                flops = estimate_flops(model, Y_obs[:1])
                
            nmse_list.append(nmse)
            range_rmse_list.append(r_rmse)
            doppler_rmse_list.append(d_rmse)
            latency_list.append(mean_lat)
            
        results[method_name] = {
            'nmse_mean': np.mean(nmse_list),
            'nmse_std': np.std(nmse_list),
            'range_rmse_mean': np.mean(range_rmse_list),
            'range_rmse_std': np.std(range_rmse_list),
            'doppler_rmse_mean': np.mean(doppler_rmse_list),
            'doppler_rmse_std': np.std(doppler_rmse_list),
            'latency_ms_mean': np.mean(latency_list),
            'latency_ms_std': np.std(latency_list),
            'params': params,
            'flops': flops
        }
        
    print("\n" + "=" * 85)
    print("TABLE I: MAIN BENCHMARK EVALUATION RESULTS (Mean ± Std over 3 Seeds)")
    print("=" * 85)
    print(f"{'Method':<20} | {'NMSE (dB)':<18} | {'Range RMSE (m)':<18} | {'FLOPs':<12} | {'Latency (ms)':<15}")
    print("-" * 85)
    for method, res in results.items():
        nmse_str = f"{res['nmse_mean']:.2f} ± {res['nmse_std']:.2f}"
        range_str = f"{res['range_rmse_mean']:.2f} ± {res['range_rmse_std']:.2f}"
        lat_str = f"{res['latency_ms_mean']:.2f} ± {res['latency_ms_std']:.2f}"
        print(f"{method:<20} | {nmse_str:<18} | {range_str:<18} | {res['flops']:<12d} | {lat_str:<15}")
    print("=" * 85 + "\n")
    
    # Save CSV
    os.makedirs("results", exist_ok=True)
    csv_path = "results/main_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "NMSE_mean_dB", "NMSE_std", "Range_RMSE_mean_m", "Range_RMSE_std", "Doppler_RMSE_mean_Hz", "Doppler_RMSE_std", "Params", "FLOPs", "Latency_ms_mean", "Latency_ms_std"])
        for method, res in results.items():
            writer.writerow([
                method, res['nmse_mean'], res['nmse_std'],
                res['range_rmse_mean'], res['range_rmse_std'],
                res['doppler_rmse_mean'], res['doppler_rmse_std'],
                res['params'], res['flops'],
                res['latency_ms_mean'], res['latency_ms_std']
            ])
    print(f"Main results saved to '{csv_path}'.")
    return results

if __name__ == "__main__":
    evaluate_all()
