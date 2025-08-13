"""
Core photonic computing simulation components.

This module implements the fundamental building blocks for photonic neural network
simulation based on recent MIT breakthroughs in integrated photonic processors.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import time
import logging

logger = logging.getLogger(__name__)


class NoiseType(Enum):
    """Types of noise in photonic systems."""
    THERMAL = "thermal"
    SHOT = "shot" 
    FABRICATION = "fabrication"
    ENVIRONMENTAL = "environmental"


@dataclass
class WavelengthConfig:
    """Configuration for wavelength-division multiplexing in photonic systems."""
    center_wavelength: float = 1550.0  # nm (C-band for telecom compatibility)
    wavelength_spacing: float = 0.8    # nm (standard DWDM spacing)
    num_channels: int = 8              # Number of WDM channels
    bandwidth: float = 50.0            # GHz per channel
    
    @property
    def wavelengths(self) -> np.ndarray:
        """Get array of all wavelengths in the WDM grid."""
        start = self.center_wavelength - (self.num_channels - 1) * self.wavelength_spacing / 2
        return np.linspace(start, start + (self.num_channels - 1) * self.wavelength_spacing, 
                          self.num_channels)


@dataclass
class ThermalConfig:
    """Thermal management configuration for photonic devices."""
    operating_temperature: float = 300.0  # K (room temperature)
    thermal_drift_rate: float = 10.0      # pm/K (resonance drift per Kelvin)
    power_per_heater: float = 15.0        # mW (power consumption per phase shifter)
    thermal_time_constant: float = 1e-3   # s (thermal response time)
    max_temperature_variation: float = 5.0 # K (maximum allowed temperature drift)


@dataclass
class FabricationConfig:
    """Fabrication tolerance and process variation parameters."""
    etch_tolerance: float = 10.0          # nm (±10nm etch variation)
    refractive_index_variation: float = 0.001  # Typical Si3N4 variation
    coupling_efficiency: float = 0.85     # Typical fiber-chip coupling
    propagation_loss: float = 0.1        # dB/cm waveguide loss
    
    def apply_fabrication_noise(self, values: np.ndarray) -> np.ndarray:
        """Apply fabrication-induced variations to device parameters."""
        noise = np.random.normal(0, self.etch_tolerance / 3, values.shape)
        return values * (1 + noise / 1000)  # Convert nm to relative variation


class PhotonicProcessor:
    """
    Core photonic computing processor simulation.
    
    Implements the fundamental photonic neural network operations based on
    Mach-Zehnder interferometer architectures with wavelength multiplexing.
    """
    
    def __init__(self, 
                 wavelength_config: WavelengthConfig,
                 thermal_config: ThermalConfig,
                 fabrication_config: FabricationConfig,
                 enable_noise: bool = True):
        """
        Initialize photonic processor with specified configurations.
        
        Args:
            wavelength_config: WDM wavelength configuration
            thermal_config: Thermal management settings
            fabrication_config: Fabrication process parameters
            enable_noise: Whether to simulate realistic noise effects
        """
        self.wavelength_config = wavelength_config
        self.thermal_config = thermal_config
        self.fabrication_config = fabrication_config
        self.enable_noise = enable_noise
        
        # Initialize processor state
        self.current_temperature = thermal_config.operating_temperature
        self.phase_shifts = {}  # Track phase shifter states
        self.power_consumption = 0.0
        
        # Performance tracking
        self.inference_times = []
        self.accuracy_history = []
        
    def mach_zehnder_operation(self, 
                              inputs: np.ndarray,
                              phase_shift: float,
                              wavelength: float = 1550.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate Mach-Zehnder interferometer operation.
        
        Based on MIT's demonstrated coherent optical neural network architecture.
        
        Args:
            inputs: Input optical signals [amplitude, phase]
            phase_shift: Applied phase shift (radians)  
            wavelength: Operating wavelength (nm)
            
        Returns:
            Tuple of (output_1, output_2) representing the two MZI outputs
        """
        # Input validation
        if not isinstance(inputs, np.ndarray) or inputs.size == 0:
            raise ValueError("Invalid input array: must be non-empty numpy array")
        if not isinstance(phase_shift, (int, float)) or not np.isfinite(phase_shift):
            raise ValueError("Phase shift must be a finite number")
        if wavelength <= 0 or wavelength > 10000:
            raise ValueError("Wavelength must be positive and realistic (0-10000nm)")
            
        if self.enable_noise:
            # Add thermal noise to phase shift
            thermal_noise = np.random.normal(0, 0.1)  # Typical thermal fluctuation
            phase_shift += thermal_noise
            
            # Add fabrication variations
            phase_shift = self.fabrication_config.apply_fabrication_noise(
                np.array([phase_shift]))[0]
        
        # Calculate MZI transfer functions with bounds checking
        try:
            cos_term = np.cos(phase_shift / 2)
            sin_term = np.sin(phase_shift / 2) * 1j
            
            # Ensure inputs have at least 2 elements for MZI operation
            if len(inputs) < 2:
                padded_inputs = np.pad(inputs, (0, 2-len(inputs)), 'constant')
            else:
                padded_inputs = inputs
            
            # Split ratio and interference
            output_1 = cos_term * padded_inputs[0] + sin_term * padded_inputs[1]
            output_2 = sin_term * padded_inputs[0] + cos_term * padded_inputs[1]
            
            # Validate outputs
            if not np.all(np.isfinite(output_1)) or not np.all(np.isfinite(output_2)):
                raise ValueError("MZI operation produced non-finite values")
            
            return output_1, output_2
        except Exception as e:
            logger.error(f"MZI operation failed: {e}")
            raise ValueError(f"MZI operation error: {e}")
    
    def wavelength_multiplexed_operation(self, 
                                       inputs: np.ndarray,
                                       weights: np.ndarray) -> np.ndarray:
        """
        Perform parallel operations across multiple wavelength channels.
        
        Implements the massive parallelism advantage of photonic systems
        using wavelength-division multiplexing.
        
        Args:
            inputs: Input data shaped (batch_size, input_dim, num_wavelengths)
            weights: Weight matrix shaped (input_dim, output_dim, num_wavelengths)
            
        Returns:
            Output data shaped (batch_size, output_dim, num_wavelengths)
        """
        batch_size, input_dim, num_wavelengths = inputs.shape
        output_dim = weights.shape[1]
        
        # Initialize output
        outputs = np.zeros((batch_size, output_dim, num_wavelengths), dtype=complex)
        
        # Process each wavelength channel in parallel (simulated sequentially)
        for w in range(num_wavelengths):
            wavelength = self.wavelength_config.wavelengths[w]
            
            # Matrix multiplication in optical domain
            for i in range(input_dim):
                for j in range(output_dim):
                    # Implement optical matrix multiplication using MZI network
                    phase_shift = np.angle(weights[i, j, w])  # Phase encoding of weights
                    amplitude = np.abs(weights[i, j, w])      # Amplitude encoding
                    
                    # Store phase shifter state for power tracking
                    self.phase_shifts[f"w{w}_i{i}_j{j}"] = phase_shift
                    
                    # Optical multiplication via interference
                    input_signal = inputs[:, i, w] * amplitude
                    outputs[:, j, w] += input_signal * np.exp(1j * phase_shift)
        
        return outputs
    
    def nonlinear_optical_function_unit(self, 
                                      inputs: np.ndarray,
                                      activation_type: str = "relu") -> np.ndarray:
        """
        Simulate Nonlinear Optical Function Unit (NOFU).
        
        Based on MIT's breakthrough in implementing nonlinear activation functions
        in the optical domain using combined electronics and optics.
        
        Args:
            inputs: Complex optical signals
            activation_type: Type of activation ("relu", "sigmoid", "tanh")
            
        Returns:
            Processed optical signals with nonlinear activation applied
        """
        # Convert to power domain for nonlinear processing
        power = np.abs(inputs) ** 2
        
        # Apply activation function
        if activation_type == "relu":
            activated = np.maximum(0, power)
        elif activation_type == "sigmoid":
            activated = 1 / (1 + np.exp(-power))
        elif activation_type == "tanh":
            activated = np.tanh(power)
        else:
            raise ValueError(f"Unsupported activation type: {activation_type}")
        
        # Convert back to optical domain maintaining phase information
        phase = np.angle(inputs)
        return np.sqrt(activated) * np.exp(1j * phase)
    
    def thermal_drift_compensation(self, current_outputs: np.ndarray) -> np.ndarray:
        """
        Simulate thermal drift effects and active compensation.
        
        Models the thermal management required for stable operation
        based on demonstrated systems with micro-heater tuning.
        
        Args:
            current_outputs: Current optical outputs
            
        Returns:
            Temperature-compensated outputs
        """
        if not self.enable_noise:
            return current_outputs
        
        # Simulate temperature fluctuation
        temp_variation = np.random.normal(0, 1.0)  # ±1K variation
        self.current_temperature += temp_variation
        
        # Calculate resonance drift
        drift_pm = self.thermal_config.thermal_drift_rate * temp_variation
        phase_drift = 2 * np.pi * drift_pm / (self.wavelength_config.center_wavelength * 1000)
        
        # Apply thermal compensation (active feedback control)
        compensation_power = abs(temp_variation) * self.thermal_config.power_per_heater
        self.power_consumption += compensation_power
        
        # Apply phase correction
        corrected_outputs = current_outputs * np.exp(1j * phase_drift)
        
        return corrected_outputs
    
    def measure_inference_latency(self, operation_func, *args, **kwargs):
        """
        Measure and record inference latency.
        
        Targets sub-nanosecond latency as demonstrated in recent MIT work.
        """
        start_time = time.perf_counter_ns()
        result = operation_func(*args, **kwargs)
        end_time = time.perf_counter_ns()
        
        latency_ns = end_time - start_time
        self.inference_times.append(latency_ns)
        
        return result, latency_ns
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get comprehensive performance metrics for the processor."""
        if not self.inference_times:
            return {}
            
        return {
            "avg_latency_ns": np.mean(self.inference_times),
            "min_latency_ns": np.min(self.inference_times),
            "max_latency_ns": np.max(self.inference_times),
            "power_consumption_mw": self.power_consumption,
            "thermal_stability": 1.0 / np.std(self.accuracy_history) if self.accuracy_history else 0.0,
            "current_temperature_k": self.current_temperature,
            "num_phase_shifters": len(self.phase_shifts),
        }
    
    def reset_performance_tracking(self):
        """Reset all performance tracking metrics."""
        self.inference_times.clear()
        self.accuracy_history.clear()
        self.power_consumption = 0.0


class OpticalSignal:
    """Represent optical signals with amplitude and phase information."""
    
    def __init__(self, amplitude: np.ndarray, phase: np.ndarray):
        """
        Initialize optical signal.
        
        Args:
            amplitude: Signal amplitude
            phase: Signal phase (radians)
        """
        self.amplitude = amplitude
        self.phase = phase
    
    @property
    def complex_representation(self) -> np.ndarray:
        """Get complex representation of the optical signal."""
        return self.amplitude * np.exp(1j * self.phase)
    
    @classmethod
    def from_complex(cls, complex_signal: np.ndarray) -> 'OpticalSignal':
        """Create OpticalSignal from complex representation."""
        return cls(
            amplitude=np.abs(complex_signal),
            phase=np.angle(complex_signal)
        )
    
    def power(self) -> np.ndarray:
        """Calculate optical power."""
        return self.amplitude ** 2