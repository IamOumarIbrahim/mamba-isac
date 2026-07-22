import os
import hashlib
import torch
import warnings

# Suppress PyTorch 2.x security warnings for torch.load
warnings.filterwarnings("ignore", category=FutureWarning)

def get_sha256(path):
    sha256_hash = hashlib.sha256()
    with open(path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def count_params(state_dict):
    return sum(p.numel() for p in state_dict.values())

mamba_path = "checkpoints/best_mamba_isac.pt"
trans_path = "checkpoints/best_transformer_isac.pt"

mamba_size = os.path.getsize(mamba_path)
trans_size = os.path.getsize(trans_path)

mamba_hash = get_sha256(mamba_path)
trans_hash = get_sha256(trans_path)

mamba_ckpt = torch.load(mamba_path, map_location="cpu", weights_only=False)
trans_ckpt = torch.load(trans_path, map_location="cpu", weights_only=False)

mamba_params = count_params(mamba_ckpt['model_state_dict'])
trans_params = count_params(trans_ckpt['model_state_dict'])

with open("checkpoints/manifest.txt", "w") as f:
    f.write("=== Mamba-ISAC Checkpoint Manifest ===\n")
    f.write(f"File: best_mamba_isac.pt\n")
    f.write(f"SHA-256: {mamba_hash}\n")
    f.write(f"Size: {mamba_size:,} bytes ({mamba_size / 1024 / 1024:.2f} MB)\n")
    f.write(f"Parameters: {mamba_params:,} ({mamba_params / 1e6:.2f}M)\n")
    
    theoretical = mamba_params * 4 * 3
    f.write(f"Expected Size Arithmetic: {mamba_params:,} params * 4 bytes/param * 3 (Weights + Adam exp_avg + Adam exp_avg_sq) = {theoretical:,} bytes\n")
    f.write(f"Serialization Overhead: {mamba_size - theoretical:,} bytes\n\n")

    f.write("=== Transformer-ISAC Checkpoint Manifest ===\n")
    f.write(f"File: best_transformer_isac.pt\n")
    f.write(f"SHA-256: {trans_hash}\n")
    f.write(f"Size: {trans_size:,} bytes ({trans_size / 1024 / 1024:.2f} MB)\n")
    f.write(f"Parameters: {trans_params:,} ({trans_params / 1e6:.2f}M)\n")
    
    theoretical_trans = trans_params * 4 * 3
    f.write(f"Expected Size Arithmetic: {trans_params:,} params * 4 bytes/param * 3 (Weights + Adam exp_avg + Adam exp_avg_sq) = {theoretical_trans:,} bytes\n")
    f.write(f"Serialization Overhead: {trans_size - theoretical_trans:,} bytes\n")

print("checkpoints/manifest.txt generated successfully.")
