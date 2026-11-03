# ==============================================================================
# File Name : dataset.py
# Purpose   : Defines the EKG image dataset
# ==============================================================================

import os
import cv2
import torch

import numpy as np
import pandas as pd

from math import gcd
from scipy.signal import resample_poly
from torch.utils.data import Dataset

from src.ekg_digitizer.preprocessing.image_processor import EKGImageProcessor
from src.ekg_digitizer.preprocessing.signal_processor import SignalProcessor
from src.ekg_digitizer.config import Config


SEGMENTS = ['0001', '0003', '0004', '0005', '0006',
            '0009', '0010', '0011', '0012']


class EKGDataset(Dataset):
    """Defines a dataset of EKG images for model training and inference."""

    def __init__(self,
                 images,
                 data_dir,
                 config: Config,
                 transform=None,
                 is_train: bool=True,
                 use_mixup: bool=False):

        self.images      = images.reset_index(drop=True)
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
        """Load the first available EKG image."""

        if self.is_train:
            for seg in SEGMENTS:
                img_path = os.path.join(
                    self.data_dir,
                    str(base_id),
                    f"{base_id}-{seg}.png"
                )

                if os.path.exists(img_path):
                    return self._convert_image(img_path)

            raise FileNotFoundError(
                f"No training image found for ID {base_id}"
            )

        img_path = os.path.join(
            self.data_dir,
            f"{base_id}.png"
        )

        if not os.path.exists(img_path):
            raise FileNotFoundError(
                f"No test image found for ID {base_id}"
            )

        return self._convert_image(img_path)


    def _convert_image(self, img_path: str) -> np.ndarray:
        """Standardizes the image colour space. Helper for load_image()."""

        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        if img is None:
            raise ValueError(
                f"Unable to load image: {img_path}"
            )

        # Convert the colour space based on the input channels
        if img.ndim == 2:
            code = cv2.COLOR_GRAY2RGB
        elif img.shape[2] == 4:
            code = cv2.COLOR_BGRA2RGB
        else:
            code = cv2.COLOR_BGR2RGB

        return cv2.cvtColor(img, code)


    def _prepare_image(self, image, base_id):
        """Standardizes the image size."""

        if image is None or image.size == 0:
            raise ValueError(
                f"Invalid image for ID {base_id}"
            )

        if self.transform:
            transformed = self.transform(image=image)
            return transformed['image']

        width = self.config.image_properties.resize_width
        height = self.config.image_properties.resize_height

        image_resized = cv2.resize(
            image,
            (width, height)
        )

        return torch.from_numpy(
            image_resized.transpose(2, 0, 1)
        ).float() / 255.0


    def __getitem__(self, idx: int):

        base_fs = self.config.signal_properties.base_fs
        lead_ii_duration = self.config.signal_properties.lead_ii_duration
        other_leads_duration = (
            self.config.signal_properties.other_leads_duration
        )

        # Load the image and standardize its colour-scheme
        row = self.images.iloc[idx]
        base_id = row['id']
        image = self.load_image(base_id)

        fs = int(row.get('fs', base_fs))

        image_tensor = self._prepare_image(
            image,
            base_id
        )

        if not self.is_train:
            return image_tensor, base_id, fs

        csv_path = os.path.join(
            self.data_dir,
            str(base_id),
            f"{base_id}.csv"
        )

        max_len = int(
            base_fs * lead_ii_duration
        )

        try:
            signals_df = pd.read_csv(csv_path)

            target_signals = np.zeros(
                (len(self.leads), max_len),
                dtype=np.float32
            )

            # Extract the time-series signal data from each lead
            for i, lead in enumerate(self.leads):

                signal = (
                    signals_df[lead]
                    .dropna()
                    .to_numpy(dtype=np.float64)
                )

                # Resample each signal to the configured base frequency
                if fs != base_fs:
                    divisor = gcd(fs, base_fs)

                    up = base_fs // divisor
                    down = fs // divisor

                    signal = resample_poly(
                        signal,
                        up=up,
                        down=down
                    )

                if lead == 'II':
                    expected_l = int(
                        base_fs * lead_ii_duration
                    )
                else:
                    expected_l = int(
                        base_fs * other_leads_duration
                    )

                if len(signal) > expected_l:
                    signal = signal[:expected_l]

                elif len(signal) < expected_l:
                    signal = np.pad(
                        signal,
                        (0, expected_l - len(signal)),
                        mode='edge'
                    )

                # Apply signal processing
                signal = self.signal_proc.denoise_signal(
                    signal
                )

                signal = self.signal_proc.remove_baseline_wander(
                    signal,
                    base_fs
                )

                # Store each lead in the shared target array
                target_signals[i, :expected_l] = (
                    signal.astype(np.float32)
                )

        except (
            FileNotFoundError,
            KeyError,
            pd.errors.ParserError
        ) as e:
            raise RuntimeError(
                f"Failed to load signals for ID {base_id}: {e!r}"
            ) from e

        return (
            image_tensor,
            torch.from_numpy(target_signals),
            base_id
        )