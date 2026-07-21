import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yaml
import torch
import numpy as np

from models.mamba_isac import MambaISAC
from baselines.transformer_isac import TransformerISAC
from baselines.lmmse import LMMSEEstimator
from data.dataset import generate_isac_samples, ISACDataset
from utils.flops import count_parameters, estimate_flops
from utils.seed import set_seed

def run_fairness_check(config_path: str = "configs/default_config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    set_seed(42)
    print("Executing Fairness Control Audit across all 3 Estimators...\n")
    
    # 1. Dataset split locking: generate single held-out test set
    test_dict = generate_isac_samples(config, num_samples=50, seed=12345)
    test_ds = ISACDataset(test_dict)
    
    # 2. Instantiate models
    mamba_model = MambaISAC(config)
    trans_model = TransformerISAC(config)
    lmmse_estimator = LMMSEEstimator(
        num_subcarriers=config['dataset']['num_subcarriers'],
        num_time_slots=config['dataset']['num_time_slots'],
        pilot_spacing=config['pilots']['pilot_spacing']
    )
    
    # 3. Model parameter count audit
    mamba_params = count_parameters(mamba_model)
    trans_params = count_parameters(trans_model)
    lmmse_params = 0 # Closed form
    
    # 4. FLOPs estimation
    sample_input = test_ds[0]['Y_obs'].unsqueeze(0) # (1, 2, Nr, Nt, Nc, T)
    mamba_flops = estimate_flops(mamba_model, sample_input)
    trans_flops = estimate_flops(trans_model, sample_input)
    lmmse_flops = config['dataset']['num_subcarriers'] ** 3 # Standard O(Nc^3) matrix inversion
    
    print("-" * 60)
    print(f"{'Method':<20} | {'Params':<12} | {'FLOPs':<15} | {'Learned':<8}")
    print("-" * 60)
    print(f"{'LMMSE Baseline':<20} | {lmmse_params:<12d} | {lmmse_flops:<15d} | {'No':<8}")
    print(f"{'Transformer Baseline':<20} | {trans_params:<12d} | {trans_flops:<15d} | {'Yes':<8}")
    print(f"{'Mamba-ISAC (Proposed)':<20} | {mamba_params:<12d} | {mamba_flops:<15d} | {'Yes':<8}")
    print("-" * 60)
    
    param_diff_pct = abs(mamba_params - trans_params) / mamba_params * 100.0
    print(f"\nParameter Count Discrepancy: {param_diff_pct:.2f}% (Target: < 25% for fair comparison)")
    assert param_diff_pct < 25.0, "Transformer baseline parameter count must match Mamba-ISAC."
    
    print("Fairness Control Audit PASSED successfully!\n")

if __name__ == "__main__":
    run_fairness_check()
