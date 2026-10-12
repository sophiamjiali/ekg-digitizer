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
                 images, 
                 data_dir, 
                 config: Config,
                 transform = None, 
                 is_train: bool = True, 
                 use_mixup: bool = False):
        
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
                if   img.ndim == 2: code = cv2.COLOR_GRAY2RGB
                elif img.ndim == 4: code = cv2.COLOR_BGRA2RGB
                else              : code = cv2.COLOR_BGR2RGB

                return cv2.cvtColor(img, code)

        # If the image was not found or could not be loaded, return zeroes
        return np.ones((self.config.img_height, self.config.img_width, 
                        self.config.img_channels), dtype = np.uint8) * 255


    def __getitem__(self, idx: int):

        # Load the image and standardize its colour-scheme
        row     = self.images.iloc[idx]
        base_id = row['id']
        image = self.load_image(base_id)

        if self.is_train:
            csv_path = os.path.join(self.data_dir, str(base_id), 
                                    f"{base_id}.csv")
            
            try:
                signals_df = pd.read_csv(csv_path)
                fs         = row.get('fs', self.config.base_fs)

                # Extract the time-series signal data from each lead
                all_signals = []
                for lead in self.leads:
                    signal = signals_df[lead].values
                    
                    # Handle NaN values
                    signal = pd.Series(signal).interpolate(
                        method          = 'linear', 
                        limit_direction = 'both'
                    ).fillna(0).values
                    
                    # Extract the lead's expected length and resize it
                    if lead == 'II':
                        expected_l = int(fs * self.config.lead_ii_duration)
                    else:
                        expected_l = int(fs * self.config.other_leads_duration)
                    
                    if len(signal) > expected_l:
                        signal = signal[:expected_l]
                    elif len(signal) < expected_l:
                        signal = np.pad(signal, (0, expected_l - len(signal)), 
                                        mode = 'edge')
                    
                    # Apply signal processing
                    signal = self.signal_proc.denoise_signal(signal)
                    signal = self.signal_proc.remove_baseline_wander(signal, fs)
                    
                    all_signals.append(signal.astype(np.float32))
                
                target_signals = np.stack(all_signals, axis = 0)
                
            except Exception as e:
                target_signals = np.zeros(
                    (12, int(self.config.base_fs * self.config.lead_ii_duration)
                ), dtype=np.float32)
            
            # Apply transformations if provided
            if self.transform:
                transformed  = self.transform(image=image)
                image_tensor = transformed['image']
            else:
                image_resized = cv2.resize(image, (self.config.resize_width, 
                                                   self.config.resize_height))
                image_tensor = torch.from_numpy(
                    image_resized.transpose(2, 0, 1)
                ).float() / 255.0
            
            return image_tensor, torch.FloatTensor(target_signals), base_id
        
        else:
            fs = row.get('fs', self.config.base_fs)
            
            if self.transform:
                transformed  = self.transform(image=image)
                image_tensor = transformed['image']
            else:
                image_resized = cv2.resize(image, (self.config.resize_width, 
                                                   self.config.resize_height))
                image_tensor = torch.from_numpy(
                    image_resized.transpose(2, 0, 1)
                ).float() / 255.0
            
            return image_tensor, base_id, fs

# [END]