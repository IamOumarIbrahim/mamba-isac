import numpy as np
from typing import Tuple, Optional

class LMMSEEstimator:
    """
    Closed-form Linear Minimum Mean-Squared Error (LMMSE) channel estimator
    extended with 2D FFT matched-filtering for sensing range and Doppler recovery.
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
            k = np.arange(self.Nc)
            dk = k[:, None] - k[None, :]
            R_hh = np.sinc(dk * 0.05) + 0.01 * np.eye(self.Nc)
            
        R_p = R_hh[self.pilot_subcarriers, :][:, self.pilot_subcarriers]
        R_hp = R_hh[:, self.pilot_subcarriers]
        
        inv_matrix = np.linalg.inv(R_p + (sigma2 / (self.Nc / self.num_pilots)) * np.eye(self.num_pilots))
        W_lmmse = R_hp @ inv_matrix
        
        H_lmmse = np.einsum('kp, ...npt->...nkt', W_lmmse, H_ls)
        return H_lmmse

    def estimate_sensing_parameters(self, Y_obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimates target range R (meters) and target Doppler shift nu_s (Hz)
        using 2D FFT matched-filtering periodogram on observation matrix Y_obs.
        
        Args:
            Y_obs: (B, Nr, Nt, Nc, T) complex observation tensor or numpy array
            
        Returns:
            R_hat: (B,) array of range estimates (meters)
            nu_s_hat: (B,) array of Doppler estimates (Hz)
        """
        if hasattr(Y_obs, "cpu"):
            Y_obs = Y_obs.cpu().numpy()
            
        # Handle real/imag split shape (B, 2, Nr, Nt, Nc, T)
        if Y_obs.ndim == 6 and Y_obs.shape[1] == 2:
            Y_obs = Y_obs[:, 0] + 1j * Y_obs[:, 1]
            
        B = Y_obs.shape[0]
        R_hat_list = []
        nu_s_hat_list = []
        
        n_fft_range = max(self.Nc * 8, 512)
        n_fft_doppler = max(self.T * 8, 512)
        doppler_freqs = np.fft.fftshift(np.fft.fftfreq(n_fft_doppler, d=self.Ts))
        
        for b in range(B):
            # Take average across antenna pairs
            snapshot = Y_obs[b, 0, 0, :, :] # (Nc, T)
            
            delay_profile = np.fft.ifft(snapshot, n=n_fft_range, axis=0) # (n_fft_range, T)
            rd_map = np.fft.fft(delay_profile, n=n_fft_doppler, axis=1) # (n_fft_range, n_fft_doppler)
            rd_map = np.fft.fftshift(rd_map, axes=1)
            
            mag = np.abs(rd_map)
            range_bin, doppler_bin = np.unravel_index(np.argmax(mag), mag.shape)
            
            est_range = (self.c * range_bin) / (2.0 * n_fft_range * self.df)
            est_nu_s = doppler_freqs[doppler_bin]
            
            R_hat_list.append(est_range)
            nu_s_hat_list.append(est_nu_s)
            
        return np.array(R_hat_list), np.array(nu_s_hat_list)

    @staticmethod
    def compute_nmse(H_est: np.ndarray, H_true: np.ndarray) -> float:
        error = np.sum(np.abs(H_est - H_true) ** 2)
        power = np.sum(np.abs(H_true) ** 2)
        nmse_linear = error / (power + 1e-12)
        return float(10.0 * np.log10(nmse_linear))
