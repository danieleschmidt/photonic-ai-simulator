"""
Model helper functions for improved code organization and reduced complexity.

This module contains helper functions extracted from complex model implementations
to improve maintainability and reduce cyclomatic complexity.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging

logger = logging.getLogger(__name__)


def quantize_weights(weights: np.ndarray, precision: int) -> np.ndarray:
    """
    Quantize weights to specified bit precision.
    
    Args:
        weights: Input weights array
        precision: Number of bits for quantization
        
    Returns:
        Quantized weights array
    """
    max_val = np.max(np.abs(weights))
    if max_val == 0:
        return weights
        
    scale = (2 ** (precision - 1) - 1) / max_val
    quantized = np.round(weights * scale) / scale
    return quantized


def apply_activation_function(data: np.ndarray, activation: str) -> np.ndarray:
    """
    Apply activation function to data.
    
    Args:
        data: Input data
        activation: Activation function name
        
    Returns:
        Activated data
    """
    if activation == "relu":
        return np.maximum(0, np.real(data))
    elif activation == "sigmoid":
        return 1 / (1 + np.exp(-np.real(data)))
    elif activation == "tanh":
        return np.tanh(np.real(data))
    elif activation == "linear" or activation is None:
        return data
    else:
        logger.warning(f"Unknown activation function: {activation}, using linear")
        return data


def calculate_he_initialization_scale(input_dim: int) -> float:
    """
    Calculate He initialization scale factor.
    
    Args:
        input_dim: Input dimension size
        
    Returns:
        Scale factor for weight initialization
    """
    return np.sqrt(2.0 / input_dim)


def create_complex_weights(input_dim: int, output_dim: int, 
                          num_channels: int, precision: int = 4) -> np.ndarray:
    """
    Create complex weights for photonic systems.
    
    Args:
        input_dim: Input dimension
        output_dim: Output dimension  
        num_channels: Number of wavelength channels
        precision: Weight precision in bits
        
    Returns:
        Complex weight matrix
    """
    amplitude_scale = calculate_he_initialization_scale(input_dim)
    
    # Initialize amplitudes
    amplitudes = np.random.normal(
        0, amplitude_scale, (input_dim, output_dim, num_channels)
    )
    
    # Initialize phases uniformly
    phases = np.random.uniform(0, 2*np.pi, (input_dim, output_dim, num_channels))
    
    # Apply quantization
    amplitudes = quantize_weights(amplitudes, precision)
    phases = quantize_weights(phases, precision)
    
    return amplitudes * np.exp(1j * phases)


def apply_thermal_compensation(data: np.ndarray, temperature: float, 
                             reference_temp: float = 300.0) -> np.ndarray:
    """
    Apply thermal drift compensation to optical data.
    
    Args:
        data: Input optical data
        temperature: Current temperature (K)
        reference_temp: Reference temperature (K)
        
    Returns:
        Compensated optical data
    """
    temp_diff = temperature - reference_temp
    thermal_drift = 10e-12 * temp_diff  # pm/K drift
    phase_correction = 2 * np.pi * thermal_drift / 1550e-9  # wavelength correction
    
    correction_factor = np.exp(-1j * phase_correction)
    return data * correction_factor


def validate_layer_config(config: Dict[str, Any]) -> bool:
    """
    Validate layer configuration parameters.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if configuration is valid
    """
    required_fields = ['input_dim', 'output_dim']
    
    for field in required_fields:
        if field not in config:
            logger.error(f"Missing required field: {field}")
            return False
    
    if config['input_dim'] <= 0 or config['output_dim'] <= 0:
        logger.error("Dimensions must be positive")
        return False
    
    if 'dropout_rate' in config:
        if not 0 <= config['dropout_rate'] <= 1:
            logger.error("Dropout rate must be between 0 and 1")
            return False
    
    if 'weight_precision' in config:
        if not 1 <= config['weight_precision'] <= 32:
            logger.error("Weight precision must be between 1 and 32 bits")
            return False
    
    return True


def calculate_layer_complexity(input_dim: int, output_dim: int, 
                             num_channels: int) -> Dict[str, Union[int, float]]:
    """
    Calculate computational complexity metrics for a layer.
    
    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        num_channels: Number of wavelength channels
        
    Returns:
        Dictionary with complexity metrics
    """
    total_weights = input_dim * output_dim * num_channels
    multiply_operations = total_weights
    addition_operations = (input_dim - 1) * output_dim * num_channels
    
    return {
        'total_parameters': total_weights,
        'multiply_operations': multiply_operations,
        'addition_operations': addition_operations,
        'memory_mb': total_weights * 16 / (1024 * 1024),  # Complex128 = 16 bytes
        'flops_estimate': multiply_operations + addition_operations
    }


def optimize_wavelength_allocation(num_channels: int, bandwidth: float, 
                                 spacing: float) -> Dict[str, Any]:
    """
    Optimize wavelength channel allocation.
    
    Args:
        num_channels: Number of channels
        bandwidth: Channel bandwidth (GHz)
        spacing: Channel spacing (nm)
        
    Returns:
        Optimized allocation parameters
    """
    # Calculate optimal spacing to minimize crosstalk
    min_spacing = bandwidth / 50.0  # 50 GHz guard band
    optimal_spacing = max(spacing, min_spacing)
    
    # Calculate channel utilization efficiency
    total_spectrum = num_channels * optimal_spacing
    utilized_spectrum = num_channels * bandwidth / 50.0  # Convert GHz to nm approx
    efficiency = utilized_spectrum / total_spectrum
    
    return {
        'optimal_spacing': optimal_spacing,
        'spectral_efficiency': efficiency,
        'total_bandwidth_required': total_spectrum,
        'crosstalk_risk': 'low' if optimal_spacing >= min_spacing else 'high'
    }


def create_benchmark_data(input_dim: int, batch_size: int = 32, 
                         data_type: str = 'mnist') -> Tuple[np.ndarray, np.ndarray]:
    """
    Create benchmark data for testing.
    
    Args:
        input_dim: Input dimension size
        batch_size: Batch size
        data_type: Type of benchmark data
        
    Returns:
        Tuple of (input_data, target_labels)
    """
    if data_type == 'mnist':
        # MNIST-like data (28x28 = 784 features)
        if input_dim == 784:
            X = np.random.randn(batch_size, input_dim) * 0.1 + 0.5
            # Add some structure to mimic image data
            X = X.reshape(batch_size, 28, 28)
            X += np.random.randn(batch_size, 28, 28) * 0.05  # Add noise
            X = X.reshape(batch_size, 784)
            X = np.clip(X, 0, 1)  # Normalize to [0,1]
            
            # 10 classes for MNIST
            y = np.eye(10)[np.random.randint(0, 10, batch_size)]
        else:
            X = np.random.randn(batch_size, input_dim) * 0.1 + 0.5
            y = np.eye(10)[np.random.randint(0, 10, batch_size)]
    
    elif data_type == 'vowel':
        # Vowel recognition task (smaller, audio-like)
        X = np.random.randn(batch_size, input_dim) * 0.2
        y = np.eye(5)[np.random.randint(0, 5, batch_size)]  # 5 vowels
    
    else:
        # Generic random data
        X = np.random.randn(batch_size, input_dim)
        output_dim = min(10, max(2, input_dim // 10))
        y = np.eye(output_dim)[np.random.randint(0, output_dim, batch_size)]
    
    return X, y