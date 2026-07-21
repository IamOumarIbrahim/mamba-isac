import torch
import torch.nn as nn
import torch.nn.functional as F

class JointISACLoss(nn.Module):
    """
    Joint Multi-Task Loss for Mamba-ISAC.
    Eq. (4) in mamba_isac_briefing.tex:
    L = lambda_c * NMSE(H_c_hat, H_c) + lambda_s * MSE(R_hat, R) + lambda_d * MSE(nu_s_hat, nu_s)
    """
    def __init__(self, lambda_c: float = 1.0, lambda_s: float = 0.1, lambda_d: float = 0.1):
        super().__init__()
        self.lambda_c = lambda_c
        self.lambda_s = lambda_s
        self.lambda_d = lambda_d

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
        loss_range = F.mse_loss(R_hat, R_true)
        loss_doppler = F.mse_loss(nu_s_hat, nu_s_true)
        
        total_loss = (
            self.lambda_c * loss_comm +
            self.lambda_s * loss_range +
            self.lambda_d * loss_doppler
        )
        
        return total_loss, {
            'loss_comm_nmse': loss_comm.item(),
            'loss_range_mse': loss_range.item(),
            'loss_doppler_mse': loss_doppler.item(),
            'total_loss': total_loss.item()
        }
