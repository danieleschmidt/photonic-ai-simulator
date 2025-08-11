"""
Advanced data processing utilities for photonic neural networks.

Implements data preprocessing, augmentation, and optimization specifically
tailored for optical computing systems with hardware constraints.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
import time

try:
    from ..core import WavelengthConfig
except ImportError:
    from core import WavelengthConfig

logger = logging.getLogger(__name__)


@dataclass
class OpticalDataConfig:
    """Configuration for optical data encoding."""
    intensity_range: Tuple[float, float] = (0.0, 1.0)
    phase_encoding: bool = True
    wavelength_channels: int = 8
    dynamic_range_db: float = 30.0
    snr_target_db: float = 20.0
    

class OpticalDataProcessor:
    """
    Advanced data processor optimized for photonic neural networks.
    
    Handles data encoding, normalization, and augmentation specifically
    designed for optical computing constraints and characteristics.
    """
    
    def __init__(self, config: OpticalDataConfig):
        """Initialize optical data processor."""
        self.config = config
        self.preprocessing_history = []
        self.encoding_stats = {}
        
    def encode_optical_intensity(self, data: np.ndarray) -> np.ndarray:
        """
        Encode data as optical intensities suitable for photonic processing.
        
        Args:
            data: Input data array
            
        Returns:
            Intensity-encoded data in range [0, 1]
        """
        # Normalize to optical intensity range
        data_min = np.min(data)
        data_max = np.max(data)
        
        # Avoid division by zero
        if data_max == data_min:
            return np.full_like(data, 0.5)
        
        # Map to intensity range with proper scaling
        normalized = (data - data_min) / (data_max - data_min)
        intensity_min, intensity_max = self.config.intensity_range
        
        encoded = normalized * (intensity_max - intensity_min) + intensity_min
        
        # Store encoding statistics
        self.encoding_stats["min_input"] = data_min
        self.encoding_stats["max_input"] = data_max
        self.encoding_stats["dynamic_range"] = data_max - data_min
        
        return encoded
    
    def add_optical_noise(self, data: np.ndarray, 
                         noise_type: str = "gaussian",
                         snr_db: float = None) -> np.ndarray:
        """
        Add realistic optical noise to data.
        
        Args:
            data: Input optical data
            noise_type: Type of noise ("gaussian", "poisson", "thermal")
            snr_db: Signal-to-noise ratio in dB
            
        Returns:
            Noisy optical data
        """
        if snr_db is None:
            snr_db = self.config.snr_target_db
        
        # Calculate noise power based on SNR
        signal_power = np.mean(data ** 2)
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        
        if noise_type == "gaussian":
            noise = np.random.normal(0, np.sqrt(noise_power), data.shape)
        elif noise_type == "poisson":
            # Poisson noise for shot noise simulation
            # Scale data to photon counts, add noise, scale back
            photon_counts = data * 1000  # Assume 1000 photons per unit intensity
            noisy_counts = np.random.poisson(photon_counts)
            return noisy_counts / 1000
        elif noise_type == "thermal":
            # Thermal noise with 1/f characteristics
            noise = np.random.normal(0, np.sqrt(noise_power), data.shape)
            # Add low-frequency component
            lf_noise = np.random.normal(0, np.sqrt(noise_power * 0.1), data.shape)
            noise = noise + lf_noise
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
        
        noisy_data = data + noise
        
        # Clip to valid optical range
        return np.clip(noisy_data, self.config.intensity_range[0], 
                      self.config.intensity_range[1])
    
    def wavelength_multiplex_encode(self, data: np.ndarray, 
                                   wavelength_config: WavelengthConfig) -> np.ndarray:
        """
        Encode data across multiple wavelength channels.
        
        Args:
            data: Input data (batch_size, features)
            wavelength_config: Wavelength configuration
            
        Returns:
            Wavelength-multiplexed data (batch_size, features, num_channels)
        """
        batch_size, features = data.shape
        num_channels = wavelength_config.num_channels
        
        # Initialize wavelength-multiplexed array
        wdm_data = np.zeros((batch_size, features, num_channels), dtype=complex)
        
        for channel in range(num_channels):
            wavelength = wavelength_config.wavelengths[channel]
            
            # Apply wavelength-dependent phase encoding
            phase_shift = 2 * np.pi * (wavelength - wavelength_config.center_wavelength) / 1000
            
            # Encode as complex optical signal
            amplitude = self.encode_optical_intensity(data)
            phase = np.full_like(amplitude, phase_shift)
            
            wdm_data[:, :, channel] = amplitude * np.exp(1j * phase)
        
        return wdm_data
    
    def apply_hardware_constraints(self, data: np.ndarray) -> np.ndarray:
        """
        Apply realistic hardware constraints to data.
        
        Args:
            data: Input optical data
            
        Returns:
            Hardware-constrained data
        """
        constrained_data = data.copy()
        
        # Apply dynamic range limitations
        max_signal = 10 ** (self.config.dynamic_range_db / 20)
        constrained_data = np.clip(constrained_data, -max_signal, max_signal)
        
        # Apply quantization effects (simulate ADC/DAC)
        if hasattr(self.config, 'quantization_bits'):
            n_levels = 2 ** self.config.quantization_bits
            constrained_data = np.round(constrained_data * n_levels) / n_levels
        
        # Apply bandwidth limitations (low-pass filter)
        # This is a simplified model - real systems would use proper filtering
        if len(constrained_data.shape) > 1 and constrained_data.shape[-1] > 10:
            # Simple moving average as bandwidth limitation
            kernel_size = 3
            kernel = np.ones(kernel_size) / kernel_size
            
            # Apply to last dimension
            for i in range(constrained_data.shape[0]):
                constrained_data[i] = np.convolve(
                    constrained_data[i], kernel, mode='same'
                )
        
        return constrained_data
    
    def preprocess_for_training(self, X: np.ndarray, y: np.ndarray,
                               add_noise: bool = True,
                               apply_constraints: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Complete preprocessing pipeline for training data.
        
        Args:
            X: Input features
            y: Target labels
            add_noise: Whether to add optical noise
            apply_constraints: Whether to apply hardware constraints
            
        Returns:
            Preprocessed (X, y) data
        """
        logger.info(f"Preprocessing training data: {X.shape}")
        
        # Encode as optical intensities
        X_optical = self.encode_optical_intensity(X)
        
        # Add realistic optical noise if requested
        if add_noise:
            X_optical = self.add_optical_noise(X_optical, "gaussian")
        
        # Apply hardware constraints
        if apply_constraints:
            X_optical = self.apply_hardware_constraints(X_optical)
        
        # Store preprocessing metadata
        preprocessing_info = {
            "timestamp": time.time(),
            "input_shape": X.shape,
            "encoding_stats": self.encoding_stats.copy(),
            "noise_added": add_noise,
            "constraints_applied": apply_constraints
        }
        self.preprocessing_history.append(preprocessing_info)
        
        return X_optical, y
    
    def augment_optical_data(self, data: np.ndarray, 
                           augmentation_factor: int = 2) -> np.ndarray:
        """
        Augment optical data with realistic variations.
        
        Args:
            data: Input optical data
            augmentation_factor: Number of augmented copies per sample
            
        Returns:
            Augmented data
        """
        augmented_samples = [data]
        
        for aug in range(augmentation_factor - 1):
            # Create augmented copy
            aug_data = data.copy()
            
            # Apply random intensity scaling (simulating power fluctuations)
            power_variation = np.random.uniform(0.9, 1.1, aug_data.shape)
            aug_data = aug_data * power_variation
            
            # Add phase noise
            if np.iscomplexobj(aug_data):
                phase_noise = np.random.normal(0, 0.1, aug_data.shape)
                aug_data = aug_data * np.exp(1j * phase_noise)
            
            # Add small amount of thermal noise
            thermal_noise = np.random.normal(0, 0.01, aug_data.shape)
            aug_data = aug_data + thermal_noise
            
            # Ensure valid optical range
            if np.iscomplexobj(aug_data):
                # Clip amplitude, preserve phase
                amplitude = np.abs(aug_data)
                phase = np.angle(aug_data)
                amplitude = np.clip(amplitude, self.config.intensity_range[0],
                                  self.config.intensity_range[1])
                aug_data = amplitude * np.exp(1j * phase)
            else:
                aug_data = np.clip(aug_data, self.config.intensity_range[0],
                                 self.config.intensity_range[1])
            
            augmented_samples.append(aug_data)
        
        return np.concatenate(augmented_samples, axis=0)
    
    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """Get summary of preprocessing operations."""
        if not self.preprocessing_history:
            return {"message": "No preprocessing operations recorded"}
        
        recent_ops = self.preprocessing_history[-10:]
        
        return {
            "total_operations": len(self.preprocessing_history),
            "recent_input_shapes": [op["input_shape"] for op in recent_ops],
            "encoding_consistency": len(set(
                str(op["encoding_stats"]) for op in recent_ops
            )) == 1,
            "latest_encoding_stats": recent_ops[-1]["encoding_stats"] if recent_ops else {}
        }


class HardwareCalibrationSystem:
    """
    System for hardware calibration and drift compensation.
    
    Implements adaptive calibration procedures for real photonic hardware
    deployment with automated drift detection and correction.
    """
    
    def __init__(self):
        """Initialize calibration system."""
        self.calibration_history = []
        self.drift_measurements = []
        self.compensation_parameters = {}
        
    def perform_wavelength_calibration(self, wavelength_config: WavelengthConfig,
                                     target_accuracy_pm: float = 5.0) -> Dict[str, float]:
        """
        Perform wavelength calibration procedure.
        
        Args:
            wavelength_config: Target wavelength configuration
            target_accuracy_pm: Required wavelength accuracy in picometers
            
        Returns:
            Calibration results and corrections
        """
        logger.info("Performing wavelength calibration")
        
        calibration_results = {}
        
        for i, target_wavelength in enumerate(wavelength_config.wavelengths):
            # Simulate wavelength measurement with realistic noise
            measured_wavelength = target_wavelength + np.random.normal(0, 2.0)  # ±2pm noise
            
            wavelength_error = measured_wavelength - target_wavelength
            
            # Calculate correction needed
            correction_pm = -wavelength_error
            
            calibration_results[f"channel_{i}"] = {
                "target_nm": target_wavelength,
                "measured_nm": measured_wavelength,
                "error_pm": wavelength_error * 1000,  # Convert to pm
                "correction_pm": correction_pm * 1000,
                "within_spec": abs(wavelength_error * 1000) <= target_accuracy_pm
            }
        
        # Store calibration results
        self.calibration_history.append({
            "timestamp": time.time(),
            "calibration_type": "wavelength",
            "results": calibration_results,
            "target_accuracy_pm": target_accuracy_pm
        })
        
        return calibration_results
    
    def monitor_thermal_drift(self, temperature_k: float,
                            reference_temperature_k: float = 300.0) -> Dict[str, float]:
        """
        Monitor and compensate for thermal drift.
        
        Args:
            temperature_k: Current temperature in Kelvin
            reference_temperature_k: Reference calibration temperature
            
        Returns:
            Drift measurement and compensation parameters
        """
        temp_difference = temperature_k - reference_temperature_k
        
        # Calculate expected drift based on thermal coefficients
        # Typical values for silicon photonics
        wavelength_drift_pm_per_k = 10.0  # pm/K
        phase_drift_rad_per_k = 0.1       # rad/K
        
        wavelength_drift_pm = temp_difference * wavelength_drift_pm_per_k
        phase_drift_rad = temp_difference * phase_drift_rad_per_k
        
        # Calculate compensation parameters
        compensation = {
            "temperature_k": temperature_k,
            "temp_difference_k": temp_difference,
            "wavelength_drift_pm": wavelength_drift_pm,
            "phase_drift_rad": phase_drift_rad,
            "wavelength_correction_pm": -wavelength_drift_pm,
            "phase_correction_rad": -phase_drift_rad
        }
        
        self.drift_measurements.append({
            "timestamp": time.time(),
            **compensation
        })
        
        return compensation
    
    def adaptive_power_calibration(self, measured_powers: np.ndarray,
                                 target_powers: np.ndarray) -> Dict[str, float]:
        """
        Perform adaptive power calibration.
        
        Args:
            measured_powers: Measured optical powers
            target_powers: Target optical powers
            
        Returns:
            Power calibration corrections
        """
        power_errors = measured_powers - target_powers
        relative_errors = power_errors / (target_powers + 1e-12)
        
        # Calculate correction factors
        correction_factors = target_powers / (measured_powers + 1e-12)
        
        calibration_result = {
            "mean_power_error": np.mean(power_errors),
            "max_power_error": np.max(np.abs(power_errors)),
            "mean_relative_error": np.mean(np.abs(relative_errors)),
            "correction_factors": correction_factors.tolist(),
            "calibration_quality": np.mean(np.abs(relative_errors)) < 0.05  # <5% error
        }
        
        return calibration_result
    
    def get_calibration_status(self) -> Dict[str, Any]:
        """Get overall calibration system status."""
        if not self.calibration_history:
            return {"status": "No calibrations performed"}
        
        recent_calibrations = self.calibration_history[-5:]
        recent_drift = self.drift_measurements[-10:] if self.drift_measurements else []
        
        return {
            "total_calibrations": len(self.calibration_history),
            "recent_calibration_types": [cal["calibration_type"] for cal in recent_calibrations],
            "thermal_drift_monitoring": len(recent_drift) > 0,
            "avg_temperature_stability": (
                np.std([d["temperature_k"] for d in recent_drift]) 
                if recent_drift else 0.0
            ),
            "last_calibration": recent_calibrations[-1]["timestamp"] if recent_calibrations else None
        }


def create_optical_data_pipeline(wavelength_config: WavelengthConfig,
                               add_augmentation: bool = True,
                               noise_level_db: float = 20.0) -> Callable:
    """
    Create complete optical data processing pipeline.
    
    Args:
        wavelength_config: Wavelength configuration
        add_augmentation: Whether to include data augmentation
        noise_level_db: SNR level for noise addition
        
    Returns:
        Configured data processing pipeline function
    """
    optical_config = OpticalDataConfig(
        wavelength_channels=wavelength_config.num_channels,
        snr_target_db=noise_level_db
    )
    
    processor = OpticalDataProcessor(optical_config)
    
    def pipeline(X: np.ndarray, y: np.ndarray, training: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Process data through optical pipeline."""
        
        # Basic optical preprocessing
        X_processed, y_processed = processor.preprocess_for_training(
            X, y, add_noise=training, apply_constraints=True
        )
        
        # Wavelength multiplexing
        X_wdm = processor.wavelength_multiplex_encode(X_processed, wavelength_config)
        
        # Data augmentation for training
        if training and add_augmentation:
            X_wdm = processor.augment_optical_data(X_wdm, augmentation_factor=2)
            # Duplicate labels for augmented data
            y_processed = np.tile(y_processed, (2, 1))
        
        return X_wdm, y_processed
    
    return pipeline