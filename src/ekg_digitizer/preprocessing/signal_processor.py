# ==============================================================================
# File Name : signal_processor.py
# Purpose   : Defines the signal processing class and utilities
# ==============================================================================

import numpy as np

from scipy import signal as scipy_signal
from scipy.signal import medfilt


class SignalProcessor:
    """
    Defines utilities for preprocessing the time-series signal extracted
    from the digitized EKG images.
    """

    @staticmethod
    def remove_baseline_wander(signal: np.ndarray, fs: int = 500) -> np.ndarray:
        """Remove baseline wander using a high-pass filter."""

        if len(signal) < 4: return signal

        # Initialize a Butterworth digital and analog filter
        sos = scipy_signal.butter(
            N      = 3,
            Wn     = 0.5,
            btype  = 'highpass',
            fs     = fs,
            output = 'sos'
        )

        return scipy_signal.sosfiltfilt(sos, x = signal)


    @staticmethod
    def denoise_signal(signal       : np.ndarray, 
                       window_length: int = 5) -> np.ndarray:
        """Denoise the signal using a median filter."""

        if len(signal) < window_length: return signal

        return medfilt(volume = signal, kernel_size = window_length)


    @staticmethod
    def normalize_signal(signal : np.ndarray,
                          method: str = 'robust', 
                          eps   : float = 1e-6) -> np.ndarray:
        """
        Normalize the signal using the defined method, including robust, 
        standard, and minmax.
        """

        if method == 'robust':

            # Perform robust normalization using percentiles
            q25, q75 = np.percentile(signal, [25, 75])
            iqr      = q75 - q25

            if iqr > eps: signal = (signal - np.median(signal)) / iqr
            
        elif method == 'standard':

            # Normalize using standard deviation
            if signal.std() > eps: 
                signal = (signal - signal.mean()) / signal.std()

        elif method == 'minmax':

            # Normalize via min-max
            signal_min, signal_max = signal.min(), signal.max()
            if signal_max - signal_min > eps:
                signal = (signal - signal_min) / (signal_max - signal_min)

                # Scale the range to [-1, 1]
                signal = signal * 2 - 1

        return signal


    @staticmethod
    def align_signals(pred     : np.ndarray, 
                      target   : np.ndarray, 
                      max_shift: int = 100) -> np.ndarray:
        """Align predicted signal with target using cross-correlation."""

        if len(pred) != len(target): return pred

        correlation = np.correlate(target, pred, mode = "same")
        shift       = np.argmax(correlation) - len(pred) // 2
        shift       = int(np.clip(shift, -max_shift, max_shift))

        if shift > 0:   aligned = np.pad(pred[shift:], (0, shift),  mode="edge")
        elif shift < 0: aligned = np.pad(pred[:shift], (-shift, 0), mode="edge")
        else:           aligned = pred

        return aligned

# [END]