import torch

from src.ekg_digitizer.models.loss import EnhancedSNRLoss


def test_loss_forward_pass():
    criterion = EnhancedSNRLoss()

    pred_dict = {
        'I': torch.randn(1, 1250),
        'II': torch.randn(1, 5000),
        'III': torch.randn(1, 1250),
        'aVR': torch.randn(1, 1250),
        'aVL': torch.randn(1, 1250),
        'aVF': torch.randn(1, 1250),
        'V1': torch.randn(1, 1250),
        'V2': torch.randn(1, 1250),
        'V3': torch.randn(1, 1250),
        'V4': torch.randn(1, 1250),
        'V5': torch.randn(1, 1250),
        'V6': torch.randn(1, 1250),
    }

    targets = torch.randn(1, 12, 5000)

    loss = criterion(pred_dict, targets)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_loss_is_small_for_perfect_predictions():
    criterion = EnhancedSNRLoss()

    targets = torch.randn(1, 12, 5000)

    pred_dict = {
        'I': targets[:, 0, :1250].clone(),
        'II': targets[:, 1, :5000].clone(),
        'III': targets[:, 2, :1250].clone(),
        'aVR': targets[:, 3, :1250].clone(),
        'aVL': targets[:, 4, :1250].clone(),
        'aVF': targets[:, 5, :1250].clone(),
        'V1': targets[:, 6, :1250].clone(),
        'V2': targets[:, 7, :1250].clone(),
        'V3': targets[:, 8, :1250].clone(),
        'V4': targets[:, 9, :1250].clone(),
        'V5': targets[:, 10, :1250].clone(),
        'V6': targets[:, 11, :1250].clone(),
    }

    loss = criterion(pred_dict, targets)

    assert loss < 1e-4
