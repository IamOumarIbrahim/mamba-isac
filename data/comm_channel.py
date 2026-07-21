import numpy as np
import scipy.special as sp
from typing import Tuple, Optional

class CommunicationChannelGenerator:
    """
    Synthetic OFDM communication channel generator supporting time-correlated
    Rician fading (Jakes Doppler spectrum) and configurable AWGN noise injection.
    """
    def __init__(
        self,
        num_subcarriers: int = 64,
        num_time_slots: int = 16,
        num_tx_antennas: int = 4,
        num_rx_antennas: int = 4,
        carrier_frequency: float = 28.0e9,
        subcarrier_spacing: float = 120.0e3,
        symbol_duration: float = 8.33e-6,
        k_factor_db: float = 10.0,
        user_velocity: float = 15.0, # m/s
        c: float = 3.0e8
    ):
        self.Nc = num_subcarriers
        self.T = num_time_slots
        self.Nt = num_tx_antennas
        self.Nr = num_rx_antennas
        self.fc = carrier_frequency
        self.df = subcarrier_spacing
        self.Ts = symbol_duration
        self.k_factor_db = k_factor_db
        self.K_linear = 10.0 ** (k_factor_db / 10.0)
        self.v = user_velocity
        self.c = c
        
        # Max Doppler shift: nu_c = v * fc / c
        self.doppler_max = (self.v * self.fc) / self.c

    def generate_jakes_correlation_matrix(self) -> np.ndarray:
        """
        Computes NxN covariance matrix according to Jakes Doppler spectrum R(dt) = J_0(2 pi fd dt).
        """
        t_indices = np.arange(self.T)
        dt = np.abs(t_indices[:, None] - t_indices[None, :]) * self.Ts
        cov = sp.jv(0, 2 * np.pi * self.doppler_max * dt)
        # Ensure positive semi-definite for Cholesky/eigen decomposition
        cov += 1e-8 * np.eye(self.T)
        return cov

    def generate_nlos_fading(self, batch_size: int = 1) -> np.ndarray:
        """
        Generates NLoS Rayleigh fading coefficients across time slots with Jakes correlation.
        Returns tensor of shape (batch_size, Nr, Nt, Nc, T) complex128.
        """
        cov = self.generate_jakes_correlation_matrix()
        L = np.linalg.cholesky(cov) # (T, T)
        
        # Independent complex Gaussian i.i.d. random noise
        # shape: (batch_size, Nr, Nt, Nc, T)
        w = (np.random.randn(batch_size, self.Nr, self.Nt, self.Nc, self.T) +
             1j * np.random.randn(batch_size, self.Nr, self.Nt, self.Nc, self.T)) / np.sqrt(2.0)
        
        # Correlate across time dimension (axis -1)
        h_nlos = np.einsum('ij, b...j->b...i', L, w)
        return h_nlos

    def generate_los_component(self, batch_size: int = 1, aoa: float = 0.0, aod: float = 0.0) -> np.ndarray:
        """
        Generates deterministic LoS component with array steering vectors and Doppler phase shift.
        Returns tensor of shape (batch_size, Nr, Nt, Nc, T) complex128.
        """
        # Steering vectors for ULA
        r_idx = np.arange(self.Nr)[:, None] # (Nr, 1)
        t_idx = np.arange(self.Nt)[None, :] # (1, Nt)
        
        a_r = np.exp(1j * np.pi * r_idx * np.sin(aoa)) # (Nr, 1)
        a_t = np.exp(1j * np.pi * t_idx * np.sin(aod)) # (1, Nt)
        array_factor = a_r @ a_t # (Nr, Nt)
        
        # Delay profile across subcarriers
        k_idx = np.arange(self.Nc)
        delay_factor = np.exp(-1j * 2 * np.pi * k_idx * 1e-9 * self.df) # (Nc,)
        
        # Doppler shift phase accumulation across time slots
        t_idx = np.arange(self.T)
        doppler_factor = np.exp(1j * 2 * np.pi * self.doppler_max * t_idx * self.Ts) # (T,)
        
        # Combine LoS component
        # (Nr, Nt, Nc, T)
        los = array_factor[:, :, None, None] * delay_factor[None, None, :, None] * doppler_factor[None, None, None, :]
        
        # Tile across batch dimension
        los_batch = np.tile(los[None, ...], (batch_size, 1, 1, 1, 1))
        return los_batch

    def generate_channel(
        self,
        batch_size: int = 1,
        snr_db: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates clean Rician channel matrix H_c and optional noisy observation.
        Returns:
            H_c: (batch_size, Nr, Nt, Nc, T) complex128 clean channel matrix
            H_c_noisy: (batch_size, Nr, Nt, Nc, T) complex128 noisy channel matrix
        """
        h_los = self.generate_los_component(batch_size=batch_size)
        h_nlos = self.generate_nlos_fading(batch_size=batch_size)
        
        # Rician combination
        coeff_los = np.sqrt(self.K_linear / (self.K_linear + 1.0))
        coeff_nlos = np.sqrt(1.0 / (self.K_linear + 1.0))
        
        H_c = coeff_los * h_los + coeff_nlos * h_nlos
        
        if snr_db is not None:
            snr_linear = 10.0 ** (snr_db / 10.0)
            noise_std = np.sqrt(1.0 / (2.0 * snr_linear))
            noise = (np.random.randn(*H_c.shape) + 1j * np.random.randn(*H_c.shape)) * noise_std
            H_c_noisy = H_c + noise
        else:
            H_c_noisy = H_c.copy()
            
        return H_c, H_c_noisy
