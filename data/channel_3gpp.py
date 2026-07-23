import numpy as np

class Channel3GPPGenerator:
    """
    Simplified TDL-style multi-path fading channel model generator.
    Generates multi-path fading channels following a statistical delay profile.
    """
    def __init__(
        self,
        num_subcarriers: int = 64,
        num_time_slots: int = 16,
        num_tx_antennas: int = 4,
        num_rx_antennas: int = 4,
        subcarrier_spacing: float = 120.0e3,
        profile: str = 'TDL-A'
    ):
        self.Nc = num_subcarriers
        self.T = num_time_slots
        self.Nt = num_tx_antennas
        self.Nr = num_rx_antennas
        self.df = subcarrier_spacing
        self.profile = profile

    def generate_channel(self, batch_size: int = 1, snr_db: float = None):
        """
        Returns:
            H_c: (batch_size, Nr, Nt, Nc, T) clean channel
            H_c_noisy: (batch_size, Nr, Nt, Nc, T) noisy channel
        """
        # Simplified statistical model: random taps with exponential decay
        num_taps = 12
        delays = np.linspace(0, 1e-6, num_taps)
        powers = np.exp(-delays / 1e-7)
        powers /= np.sum(powers)
        
        # Tap gains
        shape = (batch_size, self.Nr, self.Nt, num_taps, self.T)
        h_taps = (np.random.randn(*shape) + 1j * np.random.randn(*shape)) * np.sqrt(powers[None, None, None, :, None] / 2)
        
        # Transform taps to frequency domain
        k_idx = np.arange(self.Nc)
        phase_shift = np.exp(-1j * 2 * np.pi * k_idx[:, None] * self.df * delays[None, :]) # (Nc, num_taps)
        
        # H_c shape expected: (batch_size, Nr, Nt, Nc, T)
        # h_taps: (batch_size, Nr, Nt, num_taps, T)
        # phase_shift: (Nc, num_taps)
        H_c = np.einsum('brntT,kt->brnkT', h_taps, phase_shift)
        
        H_c_noisy = H_c.copy()
        if snr_db is not None:
            snr_linear = 10.0 ** (snr_db / 10.0)
            signal_power = np.mean(np.abs(H_c) ** 2)
            noise_std = np.sqrt(signal_power / (2.0 * snr_linear))
            noise = (np.random.randn(*H_c.shape) + 1j * np.random.randn(*H_c.shape)) * noise_std
            H_c_noisy = H_c + noise
            
        return H_c, H_c_noisy
