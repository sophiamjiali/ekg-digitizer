# ==============================================================================
# File Name : dataset.py
# Purpose   : Defines the EKG image dataset
# ==============================================================================

import os
import cv2
import torch

import numpy as np
import pandas as pd

from torch.utils.data import Dataset

from src.ekg_digitizer.preprocessing.image_processor import EKGImageProcessor
from src.ekg_digitizer.preprocessing.signal_processor import SignalProcessor
from src.ekg_digitizer.config import Config


SEGMENTS = ['0001', '0003', '0004', '0005', '0006', 
            '0009', '0010', '0011', '0012']


class EKGDataset(Dataset):
    """Defines a dataset of EKG images for model training and inference."""

    def __init__(self, 
                 images  : pd.DataFrame,
                 data_dir: str,
                 config  : Config,
                 transform = None,
                 is_train : bool     = True,
                 use_mixup: bool     = False):
        
        self.images      = images.reset_index(drop = True)
        self.data_dir    = data_dir
        self.config      = config
        self.transform   = transform
        self.is_train    = is_train
        self.use_mixup   = use_mixup and is_train
        self.processor   = EKGImageProcessor()
        self.signal_proc = SignalProcessor()
        self.leads       = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 
                            'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    def __len__(self):
        return len(self.images)


    def load_image(self, base_id):
        """Load an image, providing comprehensive feedback."""

        if self.is_train:
            for seg in SEGMENTS:
                img_path = os.path.join(self.data_dir, str(base_id), 
                                        f"{base_id}-{seg}.png")
                return self._convert_image(img_path)
            
        else:
            img_path = os.path.join(self.data_dir, f"{base_id}.png")
            return self._convert_image(img_path)


    def _convert_image(self, img_path: str) -> np.ndarray:
        """Standardizes the image colour space. Helper for load_image()."""

        if os.path.exists(img_path):
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

            if img is not None:

                # Convert the colour space based on the input channels
                if   img.ndim == 2    : code = cv2.COLOR_GRAY2RGB
                elif img.shape[2] == 4: code = cv2.COLOR_BGRA2RGB
                else                  : code = cv2.COLOR_BGR2RGB

                return cv2.cvtColor(img, code)

        # If the image was not found or could not be loaded, return zeroes
        h = self.config.image_properties.img_height
        w = self.config.image_properties.img_width
        c = self.config.image_properties.img_channels

        return np.ones((h, w, c), dtype = np.uint8) * 255


    def _prepare_image(self, image, base_id):
        """Standardizes the image size."""

        if image is None or image.size == 0:
            raise ValueError(f"Invalid image for ID {base_id}")
        if self.transform:
            return self.transform(image=image)['image']

        width = self.config.image_properties.resize_width
        height = self.config.image_properties.resize_height
        
        image = cv2.resize(image, (width, height))
        return torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0


    def __getitem__(self, idx: int):

        # Extract relevant configurations
        base_fs          = self.config.signal_properties.base_fs
        lead_ii_duration = self.config.signal_properties.lead_ii_duration
        other_duration   = self.config.signal_properties.other_leads_duration

        # Load the image and standardize its colour-scheme
        row          = self.images.iloc[idx]
        base_id      = row['id']
        image        = self.load_image(base_id)
        fs           = row.get('fs', base_fs)
        image_tensor = self._prepare_image(image, base_id)

        if not self.is_train: return image_tensor, base_id, fs

        csv_path = os.path.join(self.data_dir, str(base_id), f"{base_id}.csv")
        max_len = int(fs * lead_ii_duration)

        try:
            signals_df = pd.read_csv(csv_path)
            targets = np.zeros((len(self.leads), max_len), dtype = np.float32)
            mask = np.zeros_like(targets)

            # Ingest and process each signal, standardizing their length
            for i, lead in enumerate(self.leads):
                signal = signals_df[lead].to_numpy(dtype=np.float64)
                signal = (pd.Series(signal)
                        .interpolate(method='linear', limit_direction='both')
                        .fillna(0).to_numpy())

                duration = lead_ii_duration if lead == 'II' else other_duration
                expected_len = int(fs * duration)

                # Pad with zeroes if the length is not equal to the ii duration
                if len(signal) > expected_len:
                    signal = signal[:expected_len]
                elif len(signal) < expected_len:
                    signal = np.pad(signal, (0, expected_len - len(signal)), 
                                    mode = 'edge')

                # Preprocess the signal for training
                signal = self.signal_proc.denoise_signal(signal)
                signal = self.signal_proc.remove_baseline_wander(signal, fs)

                # Initialize a target and signal mask for loss masking
                targets[i, :expected_len] = signal
                mask[i, :expected_len]    = 1.0

        except (FileNotFoundError, KeyError, pd.errors.ParserError) as e:
            raise RuntimeError(
                f"Failed to load signals for ID {base_id}: {e!r}"
            ) from e

        targets, mask = torch.from_numpy(targets), torch.from_numpy(mask)

        return image_tensor, targets, mask, base_id

# [END]