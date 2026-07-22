import numpy as np
import torch

def isfft(x_dd: np.ndarray, axes=(-2, -1)) -> np.ndarray:
    """
    Inverse Symplectic Finite Fourier Transform (ISFFT).
    Converts from Delay-Doppler (DD) domain to Time-Frequency (TF) domain.
    Usually applied as: IFFT along Doppler (time-slots) and FFT along delay (subcarriers).
    
    Args:
        x_dd: Input Delay-Doppler matrix. Default shape assumes (..., Nc, T)
              where Nc is delay (subcarriers) and T is Doppler (time-slots).
        axes: Axes corresponding to (delay, doppler).
              
    Returns:
        x_tf: Time-Frequency matrix.
    """
    delay_axis, doppler_axis = axes
    x_tf = np.fft.fft(x_dd, axis=delay_axis, norm="ortho")
    x_tf = np.fft.ifft(x_tf, axis=doppler_axis, norm="ortho")
    return x_tf

def sfft(x_tf: np.ndarray, axes=(-2, -1)) -> np.ndarray:
    """
    Symplectic Finite Fourier Transform (SFFT).
    Converts from Time-Frequency (TF) domain to Delay-Doppler (DD) domain.
    
    Args:
        x_tf: Input Time-Frequency matrix. (..., Nc, T)
        axes: Axes corresponding to (subcarriers, time-slots).
        
    Returns:
        x_dd: Delay-Doppler matrix.
    """
    delay_axis, doppler_axis = axes
    x_dd = np.fft.ifft(x_tf, axis=delay_axis, norm="ortho")
    x_dd = np.fft.fft(x_dd, axis=doppler_axis, norm="ortho")
    return x_dd

def isfft_torch(x_dd: torch.Tensor, axes=(-2, -1)) -> torch.Tensor:
    delay_axis, doppler_axis = axes
    x_tf = torch.fft.fft(x_dd, dim=delay_axis, norm="ortho")
    x_tf = torch.fft.ifft(x_tf, dim=doppler_axis, norm="ortho")
    return x_tf

def sfft_torch(x_tf: torch.Tensor, axes=(-2, -1)) -> torch.Tensor:
    delay_axis, doppler_axis = axes
    x_dd = torch.fft.ifft(x_tf, dim=delay_axis, norm="ortho")
    x_dd = torch.fft.fft(x_dd, dim=doppler_axis, norm="ortho")
    return x_dd
