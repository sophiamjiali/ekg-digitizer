# ==============================================================================
# File Name : config.py
# Purpose   : Defines the configuration class and initializes it from .yaml
# ==============================================================================

import yaml

from dataclasses import dataclass
from torch.cuda import device_count

@dataclass
class HardwareConfig:
    use_multi_gpu: int = device_count() > 1
    device_ids: list[int] = list(range(device_count())) if use_multi_gpu else [0]

@dataclass
class TrainingConfig:
    batch_size : int
    epochs     : int
    lr         : float
    min_lr     : float
    num_workers: int

@dataclass
class OptimizationConfig:
    warmup_epochs  : int
    gradient_clip  : float
    use_amp        : bool
    label_smoothing: float

@dataclass
class ImagePropertyConfig:
    img_width    : int
    img_height   : int
    img_channels : int
    resize_width : int
    resize_height: int

@dataclass
class SignalPropertyConfig:
    base_fs             : int
    lead_ii_duration    : float
    other_leads_duration: float

@dataclass
class AugmentationConfig:
    use_tta  : bool
    tta_count: int

@dataclass
class SignalProcessingConfig:
    use_peak_detection   : bool
    use_wavelet_denoising: bool


@dataclass
class Config:
    """Initializes the configurations class from the loaded yaml."""

    hardware         : HardwareConfig
    training         : TrainingConfig
    optimization     : OptimizationConfig
    image_properties : ImagePropertyConfig
    signal_properties: SignalPropertyConfig
    augmentations    : AugmentationConfig
    signal_processing: SignalProcessingConfig


def load_config(config_path: str) -> Config:
    """Loads the .yaml configurations as a Config class object."""

    with open(config_path, 'r') as f:
        raw = yaml.safe_load(f)

    return Config(
        hardware          = HardwareConfig(),
        training          = TrainingConfig(        **raw['training']),
        optimization      = OptimizationConfig(    **raw['optimization']),
        image_properties  = ImagePropertyConfig(   **raw['image_properties']),
        signal_properties = SignalPropertyConfig(  **raw['signal_properties']),
        augmentations     = AugmentationConfig(    **raw['augmentations']),
        signal_processing = SignalProcessingConfig(**raw['signal_processing'])
    )