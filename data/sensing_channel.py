import numpy as np
from typing import Tuple, Optional

class SensingChannelGenerator:
    """
    Synthetic OFDM sensing (radar point-target echo) channel generator.
    Implements Eq. (1) from mamba_isac_briefing.tex:
    y_s[k,t] = alpha * exp(-j 2 pi k df tau) * exp(j 2 pi t Ts nu_s) * a_r(theta) a_t^T(theta) * x[k,t] + n[k,t]
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
        c: float = 3.0e8
    ):
        self.Nc = num_subcarriers
        self.T = num_time_slots
        self.Nt = num_tx_antennas
        self.Nr = num_rx_antennas
        self.fc = carrier_frequency
        self.df = subcarrier_spacing
        self.Ts = symbol_duration
        self.c = c

    def steering_vector(self, num_antennas: int, angle_rad: float) -> np.ndarray:
        """
        Computes ULA steering vector of dimension (num_antennas, 1) with d = lambda/2.
        a(theta) = exp(j * pi * m * sin(theta)), m = 0, ..., M-1
        """
        m = np.arange(num_antennas)[:, None]
        return np.exp(1j * np.pi * m * np.sin(angle_rad))

    def generate_echo(
        self,
        range_m: float = 50.0,
        velocity_ms: float = 20.0,
        reflectivity: complex = 1.0 + 0.0j,
        azimuth_rad: float = 0.0,
        tx_symbols: Optional[np.ndarray] = None,
        snr_db: Optional[float] = None,
        include_clutter: bool = False
    ) -> Tuple[np.ndarray, float, float]:
        """
        Generates sensing echo signal matrix y_s and ground truth targets.
        
        Args:
            range_m: Target range in meters
            velocity_ms: Target radial velocity in m/s
            reflectivity: Complex target radar cross-section / reflectivity
            azimuth_rad: Target azimuth angle in radians
            tx_symbols: Transmitted symbols of shape (Nc, T), defaults to unit pilot matrix
            snr_db: Configurable SNR in dB for noise injection
            include_clutter: Multi-path / clutter toggle (off by default)
            
        Returns:
            y_s: (Nr, Nt, Nc, T) complex echo observation matrix
            tau: Round-trip delay in seconds (2 * R / c)
            nu_s: Target Doppler shift in Hz (2 * v * fc / c)
        """
        # Round trip delay tau and Doppler shift nu_s
        tau = (2.0 * range_m) / self.c
        nu_s = (2.0 * velocity_ms * self.fc) / self.c
        
        # Steering vectors
        a_r = self.steering_vector(self.Nr, azimuth_rad) # (Nr, 1)
        a_t = self.steering_vector(self.Nt, azimuth_rad) # (Nt, 1)
        spatial_matrix = a_r @ a_t.T # (Nr, Nt)
        
        # Subcarrier delay phase shift
        k_idx = np.arange(self.Nc)
        delay_phase = np.exp(-1j * 2.0 * np.pi * k_idx * self.df * tau) # (Nc,)
        
        # Time Doppler phase shift
        t_idx = np.arange(self.T)
        doppler_phase = np.exp(1j * 2.0 * np.pi * t_idx * self.Ts * nu_s) # (T,)
        
        if tx_symbols is None:
            tx_symbols = np.ones((self.Nc, self.T), dtype=np.complex128)
            
        # Combine echo components according to Eq. (1)
        # y_s[r, t_ant, k, t] = reflectivity * delay_phase[k] * doppler_phase[t] * spatial[r, t_ant] * tx[k, t]
        y_s = (
            reflectivity *
            spatial_matrix[:, :, None, None] *
            delay_phase[None, None, :, None] *
            doppler_phase[None, None, None, :] *
            tx_symbols[None, None, :, :]
        )
        
        if include_clutter:
            # Simple stationary clutter component at random range & small RCS
            clutter_tau = (2.0 * 20.0) / self.c
            clutter_phase = np.exp(-1j * 2.0 * np.pi * k_idx * self.df * clutter_tau)
            clutter = 0.1 * spatial_matrix[:, :, None, None] * clutter_phase[None, None, :, None]
            y_s += clutter
            
        if snr_db is not None:
            snr_linear = 10.0 ** (snr_db / 10.0)
            signal_power = np.mean(np.abs(y_s) ** 2)
            noise_std = np.sqrt(signal_power / (2.0 * snr_linear))
            noise = (np.random.randn(*y_s.shape) + 1j * np.random.randn(*y_s.shape)) * noise_std
            y_s = y_s + noise
            
        return y_s, tau, nu_s

    def recover_range_doppler_2d_fft(self, y_s: np.ndarray) -> Tuple[float, float]:
        """
        Helper method to estimate range and Doppler shift via 2D FFT (periodogram)
        on the zero-th antenna pair. Used for unit-testing and closed-form baseline.
        """
        # Take subcarrier (delay) and time (Doppler) dimensions
        # Shape of y_s: (Nr, Nt, Nc, T)
        snapshot = y_s[0, 0, :, :] # (Nc, T)
        
        # 2D FFT: IFFT across subcarriers (delay domain), FFT across time (Doppler domain)
        # Zero padding for higher FFT resolution
        n_fft_range = max(self.Nc * 8, 512)
        n_fft_doppler = max(self.T * 8, 512)
        
        delay_profile = np.fft.ifft(snapshot, n=n_fft_range, axis=0) # (n_fft_range, T)
        range_doppler_map = np.fft.fft(delay_profile, n=n_fft_doppler, axis=1) # (n_fft_range, n_fft_doppler)
        range_doppler_map = np.fft.fftshift(range_doppler_map, axes=1)
        
        mag = np.abs(range_doppler_map)
        max_idx = np.unravel_index(np.argmax(mag), mag.shape)
        range_bin, doppler_bin = max_idx
        
        # Convert range bin to range (meters)
        # delta_tau = 1 / (n_fft_range * df)
        # R = c * tau / 2 = c * range_bin / (2 * n_fft_range * df)
        est_range = (self.c * range_bin) / (2.0 * n_fft_range * self.df)
        
        # Convert doppler bin to velocity (m/s)
        # Doppler axis ranges from -1/(2 Ts) to 1/(2 Ts)
        doppler_freqs = np.fft.fftshift(np.fft.fftfreq(n_fft_doppler, d=self.Ts))
        est_nu_s = doppler_freqs[doppler_bin]
        est_velocity = (est_nu_s * self.c) / (2.0 * self.fc)
        
        return est_range, est_velocity
