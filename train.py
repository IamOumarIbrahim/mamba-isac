import os
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset import generate_isac_samples, ISACDataset
from models.mamba_isac import MambaISAC
from models.loss import JointISACLoss
from utils.seed import set_seed

def train_model(config_path: str = "configs/default_config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    seed = config['system']['seed']
    set_seed(seed)

    device = torch.device(config['system']['device'] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset splits
    data_dir = "data/processed"
    train_path = os.path.join(data_dir, "train.npz")
    val_path = os.path.join(data_dir, "val.npz")

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        print("Dataset files not found. Generating datasets...")
        os.makedirs(data_dir, exist_ok=True)
        train_dict = generate_isac_samples(config, num_samples=config['data_splits']['num_train'], seed=seed+1)
        val_dict = generate_isac_samples(config, num_samples=config['data_splits']['num_val'], seed=seed+2)
    else:
        import numpy as np
        train_dict = dict(np.load(train_path))
        val_dict = dict(np.load(val_path))

    train_ds = ISACDataset(train_dict)
    val_ds = ISACDataset(val_dict)

    batch_size = config['training']['batch_size']
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = MambaISAC(config).to(device)
    loss_cfg = config['loss']
    loss_fn = JointISACLoss(
        lambda_c=float(loss_cfg['lambda_c']),
        lambda_s=float(loss_cfg['lambda_s']),
        lambda_d=float(loss_cfg['lambda_d'])
    ).to(device)

    lr = float(config['training']['learning_rate'])
    weight_decay = float(config['training']['weight_decay'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    epochs = config['training']['epochs']
    patience = config['training']['patience']
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    ckpt_dir = "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    best_ckpt_path = os.path.join(ckpt_dir, "best_mamba_isac.pt")

    best_val_nmse = float("inf")
    patience_counter = 0

    print("\nStarting Mamba-ISAC Training Loop...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_nmse_sum = 0.0

        for batch in train_loader:
            Y_obs = batch['Y_obs'].to(device)
            H_c_true = batch['H_c'].to(device)
            R_true = batch['range'].to(device)
            nu_s_true = batch['doppler_s'].to(device)

            optimizer.zero_grad()
            H_c_hat, R_hat, nu_s_hat = model(Y_obs)
            loss, loss_dict = loss_fn(H_c_hat, H_c_true, R_hat, R_true, nu_s_hat, nu_s_true)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss.item()
            train_nmse_sum += loss_dict['loss_comm_nmse']

        scheduler.step()

        # Validation loop
        model.eval()
        val_loss_sum = 0.0
        val_nmse_sum = 0.0
        val_range_rmse_sum = 0.0
        val_doppler_rmse_sum = 0.0

        with torch.no_grad():
            for batch in val_loader:
                Y_obs = batch['Y_obs'].to(device)
                H_c_true = batch['H_c'].to(device)
                R_true = batch['range'].to(device)
                nu_s_true = batch['doppler_s'].to(device)

                H_c_hat, R_hat, nu_s_hat = model(Y_obs)
                loss, loss_dict = loss_fn(H_c_hat, H_c_true, R_hat, R_true, nu_s_hat, nu_s_true)

                val_loss_sum += loss.item()
                val_nmse_sum += loss_dict['loss_comm_nmse']
                val_range_rmse_sum += loss_dict['raw_range_rmse_m']
                val_doppler_rmse_sum += loss_dict['raw_doppler_rmse_hz']

        num_val_batches = len(val_loader)
        avg_val_loss = val_loss_sum / num_val_batches
        avg_val_nmse = val_nmse_sum / num_val_batches
        avg_val_range_rmse = val_range_rmse_sum / num_val_batches
        avg_val_doppler_rmse = val_doppler_rmse_sum / num_val_batches

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:2d}/{epochs:2d} | "
                f"Train Loss: {train_loss_sum/len(train_loader):.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | "
                f"Val Comm NMSE: {10*torch.log10(torch.tensor(avg_val_nmse)):.2f} dB | "
                f"Val Range RMSE: {avg_val_range_rmse:.2f} m | "
                f"Val Doppler RMSE: {avg_val_doppler_rmse:.2f} Hz"
            )

        # Early stopping and best checkpoint selection based on validation NMSE
        if avg_val_nmse < best_val_nmse:
            best_val_nmse = avg_val_nmse
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_nmse': avg_val_nmse,
                'config': config
            }, best_ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}. Best Val NMSE: {10*torch.log10(torch.tensor(best_val_nmse)):.2f} dB")
                break

    print(f"Training finished. Best checkpoint saved to '{best_ckpt_path}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    args = parser.parse_args()
    train_model(args.config)
