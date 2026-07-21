import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yaml
import torch
import numpy as np

from models.mamba_isac import MambaISAC
from baselines.transformer_isac import TransformerISAC
from eval.metrics import measure_inference_latency

def profile_sequence_scaling():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nProfiling sequence length latency on device: {device}...")
    
    seq_lengths = [16, 64, 128, 256, 512, 1024]
    
    print("="*75)
    print(f"{'T (Time Slots)':<15} | {'Transformer (ms)':<20} | {'Mamba PyTorch (ms)':<20} | {'Status':<15}")
    print("-" * 75)
    
    for T in seq_lengths:
        config_t = yaml.safe_load(open("configs/default_config.yaml"))
        config_t['dataset']['num_time_slots'] = T
        
        trans_model = TransformerISAC(config_t).to(device)
        trans_model.eval()
        
        mamba_model = MambaISAC(config_t).to(device)
        mamba_model.eval()
        
        dummy_input = torch.randn(1, 2, config_t['dataset']['num_rx_antennas'], config_t['dataset']['num_tx_antennas'], config_t['dataset']['num_subcarriers'], T).to(device)
        
        try:
            trans_lat, _ = measure_inference_latency(lambda x: trans_model(x), dummy_input, num_runs=10)
            trans_str = f"{trans_lat:6.2f} ms"
            status_str = "OK"
        except torch.cuda.OutOfMemoryError:
            trans_str = "OOM"
            status_str = "CUDA OOM"
            
        mamba_lat, _ = measure_inference_latency(lambda x: mamba_model(x), dummy_input, num_runs=10)
        
        print(f"{T:<15d} | {trans_str:<20} | {mamba_lat:6.2f} ms             | {status_str:<15}")
        
    print("="*75 + "\n")

if __name__ == "__main__":
    profile_sequence_scaling()
