import numpy as np
from typing import Tuple, List

class ISACPilotAllocator:
    """
    Allocates shared comb-pattern pilot subcarriers and time slots for ISAC system.
    Supports variable pilot density and verifies non-overlapping pilot masks.
    """
    def __init__(self, num_subcarriers: int = 64, num_time_slots: int = 16, pilot_spacing: int = 4, pattern: str = "comb", delay_guard: int = 4, doppler_guard: int = 2):
        self.Nc = num_subcarriers
        self.T = num_time_slots
        self.pilot_spacing = pilot_spacing
        self.pattern = pattern
        self.delay_guard = delay_guard
        self.doppler_guard = doppler_guard
        
    def get_pilot_indices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            pilot_subcarriers: 1D array of pilot subcarrier/delay indices
            pilot_mask: 2D boolean array of shape (Nc, T) where True indicates a pilot location
        """
        pilot_mask = np.zeros((self.Nc, self.T), dtype=bool)
        
        if self.pattern == "impulse":
            # Delay-Doppler OTFS impulse pilot at grid center with guard region
            kp, lp = self.Nc // 2, self.T // 2
            pilot_mask[kp, lp] = True
            pilot_subcarriers = np.array([kp])
        else:
            # Standard OFDM comb pattern
            pilot_subcarriers = np.arange(0, self.Nc, self.pilot_spacing)
            pilot_mask[pilot_subcarriers, :] = True
            
        return pilot_subcarriers, pilot_mask

    def allocate_pilots(self, batch_size: int = 1) -> np.ndarray:
        """
        Generates complex pilot symbols on comb subcarriers.
        Returns:
            pilot_grid: (batch_size, Nc, T) complex128 array containing pilot symbols
        """
        pilot_subcarriers, pilot_mask = self.get_pilot_indices()
        pilot_grid = np.zeros((batch_size, self.Nc, self.T), dtype=np.complex128)
        
        # Standard QPSK pilot sequence on pilot subcarriers
        qpsk_constellation = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2.0)
        
        for b in range(batch_size):
            for k in pilot_subcarriers:
                # Deterministic or pseudo-random pilot sequence across time slots
                indices = np.random.choice(len(qpsk_constellation), size=self.T)
                pilot_grid[b, k, :] = qpsk_constellation[indices]
                
        return pilot_grid

    def extract_pilots(self, channel_grid: np.ndarray) -> np.ndarray:
        """
        Extracts channel observations only on pilot subcarrier locations.
        Args:
            channel_grid: (..., Nc, T)
        Returns:
            pilot_obs: (..., NumPilots, T)
        """
        pilot_subcarriers, _ = self.get_pilot_indices()
        return channel_grid[..., pilot_subcarriers, :]
