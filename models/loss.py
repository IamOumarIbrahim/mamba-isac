import torch
import torch.nn as nn
import torch.nn.functional as F

class JointISACLoss(nn.Module):
    """
    Joint Multi-Task Loss for Mamba-ISAC.
    Eq. (4) in mamba_isac_briefing.tex:
    L = lambda_c * NMSE(H_c_hat, H_c) + lambda_s * MSE_norm(R_hat, R) + lambda_d * MSE_norm(nu_s_hat, nu_s)
    """
    def __init__(
        self,
        lambda_c: float = 1.0,
        lambda_s: float = 1.0,
        lambda_d: float = 1.0,
        max_range: float = 100.0,
        max_doppler: float = 5000.0
    ):
        super().__init__()
        self.lambda_c = lambda_c
        self.lambda_s = lambda_s
        self.lambda_d = lambda_d
        self.max_range = max_range
        self.max_doppler = max_doppler

    @staticmethod
    def compute_nmse_loss(H_hat: torch.Tensor, H_true: torch.Tensor) -> torch.Tensor:
        """
        Computes linear NMSE loss = ||H_hat - H_true||^2 / ||H_true||^2
        """
        error = torch.sum((H_hat - H_true) ** 2, dim=(1, 2, 3, 4, 5))
        power = torch.sum(H_true ** 2, dim=(1, 2, 3, 4, 5)) + 1e-8
        nmse = torch.mean(error / power)
        return nmse

    def forward(
        self,
        H_c_hat: torch.Tensor,
        H_c_true: torch.Tensor,
        R_hat: torch.Tensor,
        R_true: torch.Tensor,
        nu_s_hat: torch.Tensor,
        nu_s_true: torch.Tensor
    ):
        loss_comm = self.compute_nmse_loss(H_c_hat, H_c_true)
        
        # Normalized MSE for range and Doppler
        loss_range_norm = F.mse_loss(R_hat / self.max_range, R_true / self.max_range)
        loss_doppler_norm = F.mse_loss(nu_s_hat / self.max_doppler, nu_s_true / self.max_doppler)
        
        total_loss = (
            self.lambda_c * loss_comm +
            self.lambda_s * loss_range_norm +
            self.lambda_d * loss_doppler_norm
        )
        
        # Calculate raw physical unnormalized MSE for reporting
        raw_range_rmse = torch.sqrt(F.mse_loss(R_hat, R_true) + 1e-8).item()
        raw_doppler_rmse = torch.sqrt(F.mse_loss(nu_s_hat, nu_s_true) + 1e-8).item()
        
        return total_loss, {
            'loss_comm_nmse': loss_comm.item(),
            'loss_range_norm_mse': loss_range_norm.item(),
            'loss_doppler_norm_mse': loss_doppler_norm.item(),
            'raw_range_rmse_m': raw_range_rmse,
            'raw_doppler_rmse_hz': raw_doppler_rmse,
            'total_loss': total_loss.item()
        }
