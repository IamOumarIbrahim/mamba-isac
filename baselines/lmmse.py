import numpy as np
from typing import Tuple, Optional

class LMMSEEstimator:
    """
    Closed-form Linear Minimum Mean-Squared Error (LMMSE) channel estimator
    extended with communication channel cancellation + 2D FFT periodogram for target range and Doppler recovery.
    Pure closed-form estimator with zero learned parameters.
    """
    def __init__(
        self,
        num_subcarriers: int = 64,
        num_time_slots: int = 16,
        pilot_spacing: int = 4,
        carrier_frequency: float = 28.0e9,
        subcarrier_spacing: float = 120.0e3,
        symbol_duration: float = 8.33e-6,
        c: float = 3.0e8
    ):
        self.Nc = num_subcarriers
        self.T = num_time_slots
        self.pilot_spacing = pilot_spacing
        self.pilot_subcarriers = np.arange(0, self.Nc, self.pilot_spacing)
        self.num_pilots = len(self.pilot_subcarriers)
        
        self.fc = carrier_frequency
        self.df = subcarrier_spacing
        self.Ts = symbol_duration
        self.c = c

    def estimate_comm_channel(
        self,
        Y_obs: np.ndarray,
        snr_db: float,
        R_hh: Optional[np.ndarray] = None
    ) -> np.ndarray:
        if hasattr(Y_obs, "cpu"):
            Y_obs = Y_obs.cpu().numpy()
            
        if Y_obs.ndim == 6 and Y_obs.shape[1] == 2:
            Y_obs = Y_obs[:, 0] + 1j * Y_obs[:, 1]

        snr_linear = 10.0 ** (snr_db / 10.0)
        sigma2 = 1.0 / snr_linear
        
        Y_pilots = Y_obs[..., self.pilot_subcarriers, :]
        H_ls = Y_pilots.copy()
        
        if R_hh is None:
            k = np.arange(self.Nc)
            dk = k[:, None] - k[None, :]
            R_hh = np.sinc(dk * 0.05) + 0.01 * np.eye(self.Nc)
            
        R_p = R_hh[self.pilot_subcarriers, :][:, self.pilot_subcarriers]
        R_hp = R_hh[:, self.pilot_subcarriers]
        
        inv_matrix = np.linalg.inv(R_p + (sigma2 / (self.Nc / self.num_pilots)) * np.eye(self.num_pilots))
        W_lmmse = R_hp @ inv_matrix
        
        H_lmmse = np.einsum('kp, ...npt->...nkt', W_lmmse, H_ls)
        return H_lmmse

    def estimate_sensing_parameters(
        self,
        Y_obs: np.ndarray,
        H_c_est: Optional[np.ndarray] = None,
        snr_db: float = 15.0,
        max_search_range: float = 120.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimates target range R (meters) and target Doppler shift nu_s (Hz)
        using 2D FFT periodogram within valid range window [0, max_search_range].
        """
        if hasattr(Y_obs, "cpu"):
            Y_obs = Y_obs.cpu().numpy()
            
        if Y_obs.ndim == 6 and Y_obs.shape[1] == 2:
            Y_obs = Y_obs[:, 0] + 1j * Y_obs[:, 1]
            
        if H_c_est is None:
            H_c_est = self.estimate_comm_channel(Y_obs, snr_db=snr_db)
        elif hasattr(H_c_est, "cpu"):
            H_c_est = H_c_est.cpu().numpy()
            if H_c_est.ndim == 6 and H_c_est.shape[1] == 2:
                H_c_est = H_c_est[:, 0] + 1j * H_c_est[:, 1]

        Y_residual = Y_obs - H_c_est
        
        B = Y_obs.shape[0]
        R_hat_list = []
        nu_s_hat_list = []
        
        n_fft_range = max(self.Nc * 8, 512)
        n_fft_doppler = max(self.T * 8, 512)
        doppler_freqs = np.fft.fftshift(np.fft.fftfreq(n_fft_doppler, d=self.Ts))
        
        # Max range bin corresponding to max_search_range (120 meters)
        max_bin = int(np.ceil((2.0 * max_search_range * n_fft_range * self.df) / self.c))
        max_bin = min(max_bin, n_fft_range)
        
        for b in range(B):
            snapshot = Y_residual[b, 0, 0, :, :] # (Nc, T)
            
            delay_profile = np.fft.ifft(snapshot, n=n_fft_range, axis=0) # (n_fft_range, T)
            rd_map = np.fft.fft(delay_profile, n=n_fft_doppler, axis=1)   # (n_fft_range, n_fft_doppler)
            rd_map = np.fft.fftshift(rd_map, axes=1)
            
            mag = np.abs(rd_map[:max_bin, :]) # Constrain peak search to valid range window
            range_bin, doppler_bin = np.unravel_index(np.argmax(mag), mag.shape)
            
            est_range = (self.c * range_bin) / (2.0 * n_fft_range * self.df)
            est_nu_s = doppler_freqs[doppler_bin]
            
            R_hat_list.append(est_range)
            nu_s_hat_list.append(est_nu_s)
            
        return np.array(R_hat_list), np.array(nu_s_hat_list)

    @staticmethod
    def compute_nmse(H_est: np.ndarray, H_true: np.ndarray) -> float:
        if hasattr(H_est, "cpu"):
            H_est = H_est.cpu().numpy()
        if hasattr(H_true, "cpu"):
            H_true = H_true.cpu().numpy()
            
        if H_est.ndim == 6 and H_est.shape[1] == 2:
            H_est = H_est[:, 0] + 1j * H_est[:, 1]
        if H_true.ndim == 6 and H_true.shape[1] == 2:
            H_true = H_true[:, 0] + 1j * H_true[:, 1]

        error = np.sum(np.abs(H_est - H_true) ** 2)
        power = np.sum(np.abs(H_true) ** 2)
        nmse_linear = error / (power + 1e-12)
        return float(10.0 * np.log10(nmse_linear))
