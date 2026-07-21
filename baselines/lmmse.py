import numpy as np
from typing import Tuple, Optional

class LMMSEEstimator:
    """
    Closed-form Linear Minimum Mean-Squared Error (LMMSE) channel estimator.
    Performs per-snapshot LS estimation on pilot subcarriers followed by LMMSE filtering.
    """
    def __init__(self, num_subcarriers: int = 64, num_time_slots: int = 16, pilot_spacing: int = 4):
        self.Nc = num_subcarriers
        self.T = num_time_slots
        self.pilot_spacing = pilot_spacing
        self.pilot_subcarriers = np.arange(0, self.Nc, self.pilot_spacing)
        self.num_pilots = len(self.pilot_subcarriers)

    def estimate_comm_channel(
        self,
        Y_obs: np.ndarray,
        snr_db: float,
        R_hh: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Estimates full communication channel matrix H_c from noisy observations Y_obs.
        
        Args:
            Y_obs: (..., Nr, Nt, Nc, T) complex observation matrix
            snr_db: Operating SNR in dB
            R_hh: (Nc, Nc) frequency-domain covariance matrix (optional)
            
        Returns:
            H_lmmse: (..., Nr, Nt, Nc, T) estimated channel matrix
        """
        snr_linear = 10.0 ** (snr_db / 10.0)
        sigma2 = 1.0 / snr_linear
        
        # LS estimate on pilot subcarriers
        Y_pilots = Y_obs[..., self.pilot_subcarriers, :] # (..., Nr, Nt, num_pilots, T)
        H_ls = Y_pilots.copy()
        
        if R_hh is None:
            # Estimate empirical covariance or construct sinc delay correlation profile
            # k1, k2 subcarrier differences
            k = np.arange(self.Nc)
            dk = k[:, None] - k[None, :]
            # Normalized sinc correlation matrix
            R_hh = np.sinc(dk * 0.05) + 0.01 * np.eye(self.Nc)
            
        # Extract sub-covariance matrix for pilot locations
        R_p = R_hh[self.pilot_subcarriers, :][:, self.pilot_subcarriers] # (num_pilots, num_pilots)
        R_hp = R_hh[:, self.pilot_subcarriers] # (Nc, num_pilots)
        
        # W_lmmse = R_hp * (R_p + (sigma2 / beta) * I)^(-1)
        # Beta factor accounts for signal energy / pilot symbol scaling
        inv_matrix = np.linalg.inv(R_p + (sigma2 / (self.Nc / self.num_pilots)) * np.eye(self.num_pilots))
        W_lmmse = R_hp @ inv_matrix # (Nc, num_pilots)
        
        # Interpolate across subcarrier dimension
        H_lmmse = np.einsum('kp, ...npt->...nkt', W_lmmse, H_ls)
        return H_lmmse

    @staticmethod
    def compute_nmse(H_est: np.ndarray, H_true: np.ndarray) -> float:
        """
        Computes Normalized Mean-Squared Error (NMSE) in dB.
        NMSE = ||H_est - H_true||^2 / ||H_true||^2
        """
        error = np.sum(np.abs(H_est - H_true) ** 2)
        power = np.sum(np.abs(H_true) ** 2)
        nmse_linear = error / (power + 1e-12)
        return float(10.0 * np.log10(nmse_linear))
