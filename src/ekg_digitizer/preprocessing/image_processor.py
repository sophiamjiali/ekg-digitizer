# ==============================================================================
# File Name : image_processor.py
# Purpose   : Defines the EKG image processing class and utilities
# ==============================================================================

import cv2

import numpy as np

from cv2.typing import MatLike
from src.ekg_digitizer.preprocessing.signal_processor import SignalProcessor


class EKGImageProcessor:
    """Defines utilities for preprocessing raw EKG images for digitization."""

    def __init__(self):
        self.signal_processor = SignalProcessor()

        self.lead_row_positions = {
            'I'  : (0, 0.15),    'II' : (0.15, 0.30), 'III': (0.30, 0.45),
            'aVR': (0.45, 0.60), 'aVL': (0.60, 0.75), 'aVF': (0.75, 0.90),
            'V1' : (0, 0.15),    'V2' : (0.15, 0.30), 'V3' : (0.30, 0.45),
            'V4' : (0.45, 0.60), 'V5' : (0.60, 0.75), 'V6' : (0.75, 0.90)
        }
        
        self.lead_columns = {
            'I'  : (0.05, 0.48), 'II' : (0.05, 0.48), 'III': (0.05, 0.48),
            'aVR': (0.05, 0.48), 'aVL': (0.05, 0.48), 'aVF': (0.05, 0.48),
            'V1' : (0.52, 0.95), 'V2' : (0.52, 0.95), 'V3' : (0.52, 0.95),
            'V4' : (0.52, 0.95), 'V5' : (0.52, 0.95), 'V6' : (0.52, 0.95)
        }
        

    def preprocess_image(self, image: np.ndarray) -> tuple[MatLike, MatLike]:
        """
        Preprocesses multiple images by converting to gray-scale and 
        applying multi-scale CLAHE, bilateral filtering, grid removal,
        denoising, and adaptive thresholding.
        """

        if len(image.shape) == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        # Apply gray-scale, multi-scale CLAHE, and bilateral filtering
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit = 2.5, tileGridSize = (8, 8))
        enhanced = clahe.apply(gray)
        filtered = cv2.bilateralFilter(enhanced, 7, 75, 75)

        # Perform grid removal of horizontal and vertical lines
        grid_removed = filtered.copy()

        # Remove horizontal grid lines
        for kernel_width in [30, 40, 50]:
            h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, 
                                                 (kernel_width, 1))
            h_lines = cv2.morphologyEx(grid_removed, cv2.MORPH_OPEN, 
                                       h_kernel, iterations = 1)
            grid_removed = cv2.subtract(grid_removed, h_lines)

        # Remove vertical grid lines
        for kernel_height in [30, 40, 50]:
            v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, 
                                                 (1, kernel_height))
            v_lines = cv2.morphologyEx(grid_removed, cv2.MORPH_OPEN, 
                                       v_kernel, iterations = 1)
            grid_removed = cv2.subtract(grid_removed, v_lines)

        # Apply denoising and adaptive thresholding
        denoised = cv2.fastNlMeansDenoising(grid_removed, h = 8)

        binary = cv2.adaptiveThreshold(denoised, 255, 
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 15, 3)

        return binary, enhanced

# [END]