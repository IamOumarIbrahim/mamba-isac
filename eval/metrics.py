import time
import numpy as np
import torch
from typing import Tuple, Dict, Any

def compute_nmse_db(H_est: np.ndarray, H_true: np.ndarray) -> float:
    """Computes Normalized Mean-Squared Error in dB."""
    if hasattr(H_est, "cpu"):
        H_est = H_est.cpu().numpy()
    if hasattr(H_true, "cpu"):
        H_true = H_true.cpu().numpy()
        
    error = np.sum(np.abs(H_est - H_true) ** 2)
    power = np.sum(np.abs(H_true) ** 2) + 1e-12
    return float(10.0 * np.log10(error / power))

def compute_rmse(pred: np.ndarray, target: np.ndarray) -> float:
    """Computes Root-Mean-Squared Error."""
    if hasattr(pred, "cpu"):
        pred = pred.cpu().numpy()
    if hasattr(target, "cpu"):
        target = target.cpu().numpy()
        
    return float(np.sqrt(np.mean((pred - target) ** 2)))

def measure_inference_latency(
    model_func,
    sample_input,
    num_runs: int = 50,
    warmup: int = 10
) -> Tuple[float, float]:
    """
    Measures wall-clock inference latency (mean ± std in milliseconds).
    """
    with torch.no_grad():
        for _ in range(warmup):
            _ = model_func(sample_input)
            
        latencies_ms = []
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model_func(sample_input)
            end = time.perf_counter()
            latencies_ms.append((end - start) * 1000.0)
            
    return float(np.mean(latencies_ms)), float(np.std(latencies_ms))
