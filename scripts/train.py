# ==============================================================================
# File Name : train.py
# Purpose   : Main training entry point script
# ==============================================================================

import cv2
import pandas as pd
import torch
import torch.nn as nn
import albumentations as A

from albumentations.pytorch import ToTensorV2
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from src.ekg_digitizer.config import load_config
from src.ekg_digitizer.models.dataset import EKGDataset
from src.ekg_digitizer.models.model import EnhancedEKGNet
from src.ekg_digitizer.models.loss import EnhancedSNRLoss

# ================================
# Training Functions
# ================================

def train_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    epoch,
    config
):
    model.train()
    total_loss = 0.0

    for batch_idx, (images, targets, _) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        use_amp = (
            config.optimization.use_amp
            and device.type == "cuda"
        )

        if use_amp:
            with autocast(device_type="cuda"):
                outputs = model(images)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.optimization.gradient_clip
            )

            scaler.step(optimizer)
            scaler.update()

        else:
            outputs = model(images)
            loss = criterion(outputs, targets)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.optimization.gradient_clip
            )

            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# ================================
# Main Training
# ================================

def main():
    # Load config
    config = load_config("configs/config.yaml")

    # Device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")
    print(f"Available GPUs: {torch.cuda.device_count()}")

    # Load data
    train_df = pd.read_csv(
        "/kaggle/input/physionet-ecg-image-digitization/train.csv"
    )

    print(f"Training samples: {len(train_df)}")

    # Training augmentations
    train_transform = A.Compose([
        A.Resize(
            config.image_properties.resize_height,
            config.image_properties.resize_width
        ),

        A.OneOf([
            A.GaussNoise(
                var_limit=(5, 40),
                p=1.0
            ),

            A.ISONoise(
                color_shift=(0.01, 0.04),
                intensity=(0.05, 0.4),
                p=1.0
            ),

            A.MultiplicativeNoise(
                multiplier=(0.95, 1.05),
                p=1.0
            ),
        ], p=0.35),

        A.OneOf([
            A.MotionBlur(
                blur_limit=3,
                p=1.0
            ),

            A.GaussianBlur(
                blur_limit=(3, 5),
                p=1.0
            ),

            A.MedianBlur(
                blur_limit=3,
                p=1.0
            ),
        ], p=0.25),

        A.ShiftScaleRotate(
            shift_limit=0.02,
            scale_limit=0.03,
            rotate_limit=1,
            border_mode=cv2.BORDER_CONSTANT,
            value=255,
            p=0.3
        ),

        A.RandomBrightnessContrast(
            brightness_limit=0.12,
            contrast_limit=0.12,
            p=0.35
        ),

        A.OneOf([
            A.GridDistortion(
                num_steps=5,
                distort_limit=0.05,
                p=1.0
            ),

            A.ElasticTransform(
                alpha=1,
                sigma=10,
                p=1.0
            ),
        ], p=0.15),

        A.CoarseDropout(
            max_holes=6,
            max_height=10,
            max_width=10,
            fill_value=255,
            p=0.2
        ),

        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),

        ToTensorV2()
    ])

    # Dataset
    train_dataset = EKGDataset(
        images=train_df,
        data_dir="/kaggle/input/physionet-ecg-image-digitization/train",
        config=config,
        transform=train_transform,
        is_train=True
    )

    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(config.training.num_workers > 0),
        drop_last=True
    )

    # Model
    model = EnhancedEKGNet(config)

    if config.hardware.use_multi_gpu:
        print(
            f"Using DataParallel with "
            f"{len(config.hardware.device_ids)} GPUs"
        )

        model = nn.DataParallel(
            model,
            device_ids=config.hardware.device_ids
        )

    model = model.to(device)

    # Loss
    criterion = EnhancedSNRLoss()

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.lr,
        weight_decay=5e-5,
        betas=(0.9, 0.999)
    )

    # Learning-rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=6,
        T_mult=2,
        eta_min=config.training.min_lr
    )

    # AMP scaler
    use_amp = (
        config.optimization.use_amp
        and device.type == "cuda"
    )

    scaler = GradScaler("cuda") if use_amp else None

    # Training loop
    best_loss = float("inf")

    patience = 8
    patience_counter = 0

    for epoch in range(config.training.epochs):

        # Warmup
        if epoch < config.optimization.warmup_epochs:

            lr = (
                config.training.lr
                * (epoch + 1)
                / config.optimization.warmup_epochs
            )

            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        loss = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            epoch,
            config
        )

        if epoch >= config.optimization.warmup_epochs:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch + 1}/{config.training.epochs}, "
            f"Loss: {loss:.4f}, "
            f"LR: {current_lr:.6f}"
        )

        # Save best model
        if loss < best_loss:

            best_loss = loss
            patience_counter = 0

            save_dict = {
                "epoch": epoch,

                "model_state_dict":
                    model.module.state_dict()
                    if config.hardware.use_multi_gpu
                    else model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "loss":
                    best_loss,
            }

            torch.save(
                save_dict,
                "best_model.pth"
            )

            print(
                f"  -> Saved "
                f"(loss: {best_loss:.4f})"
            )

        else:
            patience_counter += 1

            if patience_counter >= patience:
                print(
                    f"Early stopping at epoch "
                    f"{epoch + 1}"
                )

                break

    return model


# ================================
# Execute
# ================================

if __name__ == "__main__":
    print("=" * 80)
    print("Enhanced EKG Digitization - Training Started")
    print("=" * 80)

    main()

    print("=" * 80)
    print("Training Complete!")
    print("=" * 80)