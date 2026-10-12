# ==============================================================================
# File Name : config.py
# Purpose   : Defines the configuration class and initializes it from .yaml
# ==============================================================================

import torch


class Config:
    def __init__(self, configurations):
        """Initializes the configurations class from the loaded yaml."""

        self.use_multi_gpu = torch.cuda.device_count() > 1

        if not self.use_multi_gpu: self.device_ids = [0]
        else: self.device_ids = list(range(torch.cuda.device_count()))

        ...
        


def load_config(config_path: str) -> Config:
    """Loads the .yaml configurations as a Config class object."""

    ...