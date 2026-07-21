import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yaml
import torch
import numpy as np

from data.comm_channel import CommunicationChannelGenerator
from data.dataset import generate_isac_samples, ISACDataset
from models.mamba_isac import MambaISAC
from baselines.transformer_isac import TransformerISAC
from baselines.lmmse import LMMSEEstimator
from eval.metrics import compute_nmse_db, compute_rmse
from utils.flops import estimate_flops, count_parameters
from utils.seed import set_seed

def run_raw_diagnostics():
    print("=== RAW DIAGNOSTIC OUTPUT ===")
    
    with open("configs/default_config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("PyTorch CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU Device Name:", torch.cuda.get_device_name(0))
        
    scs = float(config['dataset']['subcarrier_spacing'])
    nc = int(config['dataset']['num_subcarriers'])
    t_len = int(config['dataset']['num_time_slots'])
    
    print("\n--- CONFIG PARAMETERS ---")
    print(f"Subcarriers Nc: {nc}")
    print(f"Time slots T: {t_len}")
    print(f"Subcarrier spacing df: {scs} Hz")
    bandwidth_mhz = (nc * scs) / 1e6
    print(f"Total Bandwidth: {bandwidth_mhz:.2f} MHz")
    rayleigh_res_m = 3e8 / (2 * bandwidth_mhz * 1e6)
    print(f"Theoretical Rayleigh Range Resolution (c / 2B): {rayleigh_res_m:.2f} meters")

    # 1. Test LMMSE across SNRs with exact dataset samples
    print("\n--- LMMSE NMSE & RANGE RMSE RAW SWEEP (Seed 42, 100 samples) ---")
    
    comm_gen = CommunicationChannelGenerator(
        num_subcarriers=nc,
        num_time_slots=t_len,
        num_tx_antennas=int(config['dataset']['num_tx_antennas']),
        num_rx_antennas=int(config['dataset']['num_rx_antennas']),
        k_factor_db=float(config['comm_channel']['k_factor_db']),
        user_velocity=float(config['comm_channel']['user_velocity'])
    )
    H_c_all, _ = comm_gen.generate_channel(batch_size=1000, snr_db=None)
    mu_h = np.mean(H_c_all, axis=0)
    H_flat = (H_c_all - mu_h).transpose(0, 1, 2, 4, 3).reshape(-1, nc)
    C_hh = (H_flat.conj().T @ H_flat) / H_flat.shape[0]

    lmmse = LMMSEEstimator(
        num_subcarriers=nc,
        num_time_slots=t_len,
        pilot_spacing=int(config['pilots']['pilot_spacing'])
    )

    for snr_db in [-10, -5, 0, 5, 10, 15, 20, 25, 30]:
        config_snr = yaml.safe_load(open("configs/default_config.yaml"))
        config_snr['comm_channel']['snr_db'] = float(snr_db)
        test_dict = generate_isac_samples(config_snr, num_samples=100, seed=42)
        
        Y_obs_complex = test_dict['Y_obs']
        H_lmmse = lmmse.estimate_comm_channel(Y_obs_complex, snr_db=float(snr_db), R_hh=C_hh, mu_h=mu_h)
        R_lmmse, _ = lmmse.estimate_sensing_parameters(Y_obs_complex, H_c_est=H_lmmse, snr_db=float(snr_db))
        
        nmse = compute_nmse_db(H_lmmse, test_dict['H_c'])
        r_rmse = compute_rmse(R_lmmse, test_dict['range'])
        print(f"SNR: {snr_db:3d} dB | LMMSE NMSE: {nmse:7.2f} dB | LMMSE Range RMSE: {r_rmse:6.2f} m")

    # 2. Check Deep Models at 15 dB SNR across 3 Seeds
    print("\n--- MODEL EVALUATION AT 15 dB SNR (3 Seeds) ---")
    mamba_ckpt = "checkpoints/best_mamba_isac.pt"
    trans_ckpt = "checkpoints/best_transformer_isac.pt"

    trans_model = TransformerISAC(config).to(device)
    if os.path.exists(trans_ckpt):
        ckpt = torch.load(trans_ckpt, map_location=device, weights_only=False)
        trans_model.load_state_dict(ckpt['model_state_dict'], strict=False)
    trans_model.eval()

    mamba_model = MambaISAC(config).to(device)
    if os.path.exists(mamba_ckpt):
        ckpt = torch.load(mamba_ckpt, map_location=device, weights_only=False)
        mamba_model.load_state_dict(ckpt['model_state_dict'], strict=False)
    mamba_model.eval()

    for seed in [42, 43, 44]:
        set_seed(seed)
        test_dict = generate_isac_samples(config, num_samples=100, seed=seed)
        test_ds = ISACDataset(test_dict)
        
        # Centered LMMSE
        Y_obs_complex = test_dict['Y_obs']
        H_lmmse = lmmse.estimate_comm_channel(Y_obs_complex, snr_db=15.0, R_hh=C_hh, mu_h=mu_h)
        R_lmmse, _ = lmmse.estimate_sensing_parameters(Y_obs_complex, H_c_est=H_lmmse, snr_db=15.0)
        nmse_lmmse = compute_nmse_db(H_lmmse, test_dict['H_c'])
        r_rmse_lmmse = compute_rmse(R_lmmse, test_dict['range'])

        # Transformer
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

        # Mamba-ISAC
        with torch.no_grad():
            H_c_hat_mamba, R_mamba, _ = mamba_model(Y_obs)
        H_c_hat_mamba_np = H_c_hat_mamba.cpu().numpy()
        nmse_mamba = compute_nmse_db(
            H_c_hat_mamba_np[:, 0] + 1j * H_c_hat_mamba_np[:, 1],
            H_c_true[:, 0] + 1j * H_c_true[:, 1]
        )
        r_rmse_mamba = compute_rmse(R_mamba, test_ds.range)

        print(f"Seed {seed} | LMMSE NMSE: {nmse_lmmse:6.2f} dB, RMSE: {r_rmse_lmmse:5.2f}m | Trans NMSE: {nmse_trans:6.2f} dB, RMSE: {r_rmse_trans:5.2f}m | Mamba NMSE: {nmse_mamba:6.2f} dB, RMSE: {r_rmse_mamba:5.2f}m")

    # 3. Print FLOPs and Parameters Breakdown
    print("\n--- COMPUTE BREAKDOWN ---")
    dummy_input = torch.randn(1, 2, config['dataset']['num_rx_antennas'], config['dataset']['num_tx_antennas'], config['dataset']['num_subcarriers'], config['dataset']['num_time_slots']).to(device)
    flops_trans = estimate_flops(trans_model, dummy_input)
    flops_mamba = estimate_flops(mamba_model, dummy_input)
    params_trans = count_parameters(trans_model)
    params_mamba = count_parameters(mamba_model)
    
    print(f"Transformer  | FLOPs: {flops_trans:<12d} | Params: {params_trans:<10d}")
    print(f"Mamba-ISAC   | FLOPs: {flops_mamba:<12d} | Params: {params_mamba:<10d}")

if __name__ == "__main__":
    run_raw_diagnostics()
