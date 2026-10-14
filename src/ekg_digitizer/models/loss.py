# ==============================================================================
# File Name : loss.py
# Purpose   : Defines the loss function
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.ekg_digitizer.preprocessing.signal_processor import SignalProcessor


class EnhancedSNRLoss(nn.Module):
    """Advanced loss function with dynamic weighting"""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.smooth_l1 = nn.SmoothL1Loss(beta=0.5)
        self.signal_processor = SignalProcessor()

    def pearson_correlation_loss(self, pred, target):
        """Pearson correlation coefficient loss"""
        pred_mean = pred.mean(dim=1, keepdim=True)
        target_mean = target.mean(dim=1, keepdim=True)

        pred_centered = pred - pred_mean
        target_centered = target - target_mean

        numerator = (pred_centered * target_centered).sum(dim=1)
        denominator = torch.sqrt((pred_centered ** 2).sum(dim=1) * (target_centered ** 2).sum(dim=1))

        correlation = numerator / (denominator + 1e-6)

        return (1 - correlation).mean()

    def forward(self, pred_dict, target_signals):
        total_loss = 0

        lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
                      'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

        for idx, lead_name in enumerate(lead_names):
            pred = pred_dict[lead_name]
            target = target_signals[:, idx, :pred.size(1)]

            # Core losses
            mse_loss = self.mse(pred, target)
            smooth_loss = self.smooth_l1(pred, target)

            # Correlation loss
            corr_loss = self.pearson_correlation_loss(pred, target)

            # Frequency domain loss
            try:
                pred_fft = torch.fft.rfft(pred, dim=1)
                target_fft = torch.fft.rfft(target, dim=1)

                # Magnitude and phase losses
                mag_loss = F.l1_loss(
                    torch.abs(pred_fft),
                    torch.abs(target_fft)
                )

                phase_loss = F.l1_loss(
                    torch.angle(pred_fft),
                    torch.angle(target_fft)
                )

                freq_loss = 0.7 * mag_loss + 0.3 * phase_loss

            except:
                freq_loss = 0

            # Gradient loss
            pred_grad = pred[:, 1:] - pred[:, :-1]
            target_grad = target[:, 1:] - target[:, :-1]
            grad_loss = F.l1_loss(pred_grad, target_grad)

            # Second derivative loss (for smoothness)
            pred_grad2 = pred_grad[:, 1:] - pred_grad[:, :-1]
            target_grad2 = target_grad[:, 1:] - target_grad[:, :-1]
            smooth_loss2 = F.l1_loss(pred_grad2, target_grad2)

            # Combine with adaptive weights
            lead_loss = (
                0.2 * mse_loss +
                0.2 * smooth_loss +
                0.2 * corr_loss +
                0.15 * freq_loss +
                0.15 * grad_loss +
                0.1 * smooth_loss2
            )

            # Dynamic weighting for lead II
            weight = 2.5 if lead_name == 'II' else 1.0

            total_loss += weight * lead_loss

        return total_loss / (len(lead_names) + 1.5)