"""
Photonic neural network model architectures.

Implements state-of-the-art photonic neural network layers and models
based on recent breakthroughs in integrated photonic processors.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

try:
    from .core import PhotonicProcessor, WavelengthConfig, ThermalConfig, FabricationConfig, OpticalSignal
except ImportError:
    from core import PhotonicProcessor, WavelengthConfig, ThermalConfig, FabricationConfig, OpticalSignal


logger = logging.getLogger(__name__)


@dataclass
class LayerConfig:
    """Configuration for photonic neural network layers."""
    input_dim: int
    output_dim: int
    activation: str = "relu"
    use_thermal_compensation: bool = True
    dropout_rate: float = 0.0
    weight_precision: int = 4  # bits (based on demonstrated 4-bit precision improvements)


class MZILayer:
    """
    Mach-Zehnder Interferometer layer for photonic neural networks.
    
    Based on MIT's demonstration of coherent optical neural networks
    with integrated linear and nonlinear functions.
    """
    
    def __init__(self, 
                 config: LayerConfig,
                 wavelength_config: WavelengthConfig,
                 thermal_config: ThermalConfig,
                 fabrication_config: FabricationConfig):
        """
        Initialize MZI layer.
        
        Args:
            config: Layer configuration
            wavelength_config: Wavelength multiplexing configuration  
            thermal_config: Thermal management configuration
            fabrication_config: Fabrication process parameters
        """
        self.config = config
        self.processor = PhotonicProcessor(
            wavelength_config, thermal_config, fabrication_config
        )
        
        # Initialize weights using He initialization adapted for photonic systems
        self.weights = self._initialize_weights()
        self.biases = np.zeros((config.output_dim, wavelength_config.num_channels), dtype=complex)
        
        # Training state
        self.is_training = True
        self.weight_updates = []
        
    def _initialize_weights(self) -> np.ndarray:
        """Initialize weights optimized for photonic implementation."""
        # Use amplitude and phase encoding
        amplitude_scale = np.sqrt(2.0 / self.config.input_dim)  # He initialization
        
        # Initialize amplitudes
        amplitudes = np.random.normal(
            0, amplitude_scale, 
            (self.config.input_dim, self.config.output_dim, 
             self.processor.wavelength_config.num_channels)
        )
        
        # Initialize phases uniformly
        phases = np.random.uniform(
            0, 2*np.pi,
            (self.config.input_dim, self.config.output_dim, 
             self.processor.wavelength_config.num_channels)
        )
        
        # Apply weight precision quantization
        amplitudes = self._quantize_weights(amplitudes, self.config.weight_precision)
        phases = self._quantize_weights(phases, self.config.weight_precision)
        
        return amplitudes * np.exp(1j * phases)
    
    def _quantize_weights(self, weights: np.ndarray, precision: int) -> np.ndarray:
        """Quantize weights to specified bit precision."""
        max_val = np.max(np.abs(weights))
        if max_val == 0:
            return weights
            
        # Quantize to specified precision
        scale = (2 ** (precision - 1) - 1) / max_val
        quantized = np.round(weights * scale) / scale
        
        return quantized
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward pass through MZI layer.
        
        Args:
            inputs: Input data shaped (batch_size, input_dim, num_wavelengths)
            
        Returns:
            Layer outputs shaped (batch_size, output_dim, num_wavelengths)
        """
        # Ensure inputs are in correct format
        if inputs.ndim == 2:
            # Add wavelength dimension
            inputs = np.tile(inputs[:, :, np.newaxis], 
                           (1, 1, self.processor.wavelength_config.num_channels))
        
        # Convert to complex optical signals if needed
        if not np.iscomplexobj(inputs):
            inputs = inputs.astype(complex)
        
        # Apply dropout during training
        if self.is_training and self.config.dropout_rate > 0:
            dropout_mask = np.random.binomial(
                1, 1 - self.config.dropout_rate, inputs.shape
            )
            inputs = inputs * dropout_mask / (1 - self.config.dropout_rate)
        
        # Wavelength-multiplexed matrix multiplication
        linear_outputs = self.processor.wavelength_multiplexed_operation(inputs, self.weights)
        
        # Add biases
        linear_outputs = linear_outputs + self.biases[np.newaxis, :, :]
        
        # Apply thermal drift compensation
        compensated_outputs = self.processor.thermal_drift_compensation(linear_outputs)
        
        # Apply nonlinear activation
        activated_outputs = self.processor.nonlinear_optical_function_unit(
            compensated_outputs, self.config.activation
        )
        
        return activated_outputs
    
    def backward(self, grad_output: np.ndarray, layer_input: np.ndarray) -> np.ndarray:
        """
        Backward pass for gradient computation (for conventional training).
        
        Args:
            grad_output: Gradient from next layer
            layer_input: Input to this layer during forward pass
            
        Returns:
            Gradient with respect to layer input
        """
        # This is used for comparison with forward-only training
        batch_size = grad_output.shape[0]
        
        # Compute gradients with respect to weights
        weight_grad = np.zeros_like(self.weights)
        for w in range(self.processor.wavelength_config.num_channels):
            weight_grad[:, :, w] = (layer_input[:, :, w].T.conj() @ grad_output[:, :, w]) / batch_size
        
        # Store for parameter updates
        self.weight_updates.append(weight_grad)
        
        # Compute gradient with respect to input
        input_grad = np.zeros_like(layer_input)
        for w in range(self.processor.wavelength_config.num_channels):
            input_grad[:, :, w] = grad_output[:, :, w] @ self.weights[:, :, w].T.conj()
        
        return input_grad
    
    def update_weights(self, learning_rate: float = 0.001):
        """Update weights using accumulated gradients."""
        if self.weight_updates:
            avg_grad = np.mean(self.weight_updates, axis=0)
            self.weights -= learning_rate * avg_grad
            
            # Re-quantize weights after update
            amplitude = np.abs(self.weights)
            phase = np.angle(self.weights)
            amplitude = self._quantize_weights(amplitude, self.config.weight_precision)
            phase = self._quantize_weights(phase, self.config.weight_precision)
            self.weights = amplitude * np.exp(1j * phase)
            
            self.weight_updates.clear()
    
    def get_parameter_count(self) -> int:
        """Get total number of trainable parameters."""
        return self.weights.size + self.biases.size


class PhotonicNeuralNetwork:
    """
    Complete photonic neural network implementation.
    
    Based on MIT's demonstration of fully integrated photonic processors
    achieving 92.5% accuracy on classification tasks.
    """
    
    def __init__(self, 
                 layer_configs: List[LayerConfig],
                 wavelength_config: WavelengthConfig,
                 thermal_config: ThermalConfig,
                 fabrication_config: FabricationConfig):
        """
        Initialize photonic neural network.
        
        Args:
            layer_configs: Configuration for each layer
            wavelength_config: Wavelength multiplexing settings
            thermal_config: Thermal management settings  
            fabrication_config: Fabrication process parameters
        """
        self.layer_configs = layer_configs
        self.wavelength_config = wavelength_config
        self.thermal_config = thermal_config
        self.fabrication_config = fabrication_config
        
        # Build network layers
        self.layers = []
        for i, config in enumerate(layer_configs):
            layer = MZILayer(config, wavelength_config, thermal_config, fabrication_config)
            self.layers.append(layer)
            logger.info(f"Created layer {i}: {config.input_dim} -> {config.output_dim}")
        
        # Training state
        self.is_training = True
        self.training_history = {
            "loss": [],
            "accuracy": [],
            "latency": [],
            "power_consumption": []
        }
    
    def forward(self, inputs: np.ndarray, measure_latency: bool = True) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Forward propagation through the network.
        
        Args:
            inputs: Input data shaped (batch_size, input_dim)
            measure_latency: Whether to measure inference latency
            
        Returns:
            Tuple of (outputs, metrics) where metrics contains performance data
        """
        current_input = inputs
        total_latency = 0.0
        layer_metrics = []
        
        # Forward pass through all layers
        for i, layer in enumerate(self.layers):
            if measure_latency:
                output, latency = layer.processor.measure_inference_latency(
                    layer.forward, current_input
                )
                total_latency += latency
            else:
                output = layer.forward(current_input)
                latency = 0.0
            
            # Collect layer metrics
            layer_metrics.append({
                "layer_id": i,
                "latency_ns": latency,
                "power_mw": layer.processor.power_consumption,
                "temperature_k": layer.processor.current_temperature
            })
            
            current_input = output
        
        # Convert from wavelength-multiplexed to final output
        final_output = self._aggregate_wavelength_outputs(current_input)
        
        # Compile overall metrics
        metrics = {
            "total_latency_ns": total_latency,
            "total_power_mw": sum(m["power_mw"] for m in layer_metrics),
            "avg_temperature_k": np.mean([m["temperature_k"] for m in layer_metrics]),
            "layer_metrics": layer_metrics
        }
        
        return final_output, metrics
    
    def _aggregate_wavelength_outputs(self, wavelength_outputs: np.ndarray) -> np.ndarray:
        """Aggregate outputs across wavelength channels."""
        # Take magnitude and sum across wavelength channels
        power_outputs = np.abs(wavelength_outputs) ** 2
        return np.sum(power_outputs, axis=2)  # Sum over wavelength dimension
    
    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Make predictions with the network."""
        self.set_training(False)
        outputs, _ = self.forward(inputs)
        return outputs
    
    def set_training(self, training: bool):
        """Set training mode for all layers."""
        self.is_training = training
        for layer in self.layers:
            layer.is_training = training
    
    def get_total_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(layer.get_parameter_count() for layer in self.layers)
    
    def get_network_summary(self) -> Dict[str, Any]:
        """Get comprehensive network summary."""
        total_params = self.get_total_parameters()
        
        layer_info = []
        for i, (layer, config) in enumerate(zip(self.layers, self.layer_configs)):
            layer_info.append({
                "layer_id": i,
                "input_dim": config.input_dim,
                "output_dim": config.output_dim,
                "activation": config.activation,
                "parameters": layer.get_parameter_count(),
                "weight_precision_bits": config.weight_precision
            })
        
        return {
            "total_layers": len(self.layers),
            "total_parameters": total_params,
            "wavelength_channels": self.wavelength_config.num_channels,
            "operating_wavelength_nm": self.wavelength_config.center_wavelength,
            "layer_details": layer_info,
            "thermal_compensation": any(config.use_thermal_compensation 
                                      for config in self.layer_configs)
        }
    
    def save_model(self, filepath: str):
        """Save model weights and configuration."""
        model_data = {
            "layer_configs": self.layer_configs,
            "wavelength_config": self.wavelength_config,
            "thermal_config": self.thermal_config,
            "fabrication_config": self.fabrication_config,
            "weights": [layer.weights for layer in self.layers],
            "biases": [layer.biases for layer in self.layers],
            "training_history": self.training_history
        }
        
        np.save(filepath, model_data)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model weights and configuration."""
        model_data = np.load(filepath, allow_pickle=True).item()
        
        # Restore weights and biases
        for i, layer in enumerate(self.layers):
            layer.weights = model_data["weights"][i]
            layer.biases = model_data["biases"][i]
        
        self.training_history = model_data["training_history"]
        logger.info(f"Model loaded from {filepath}")


def create_benchmark_network(task: str = "mnist") -> PhotonicNeuralNetwork:
    """
    Create standard benchmark network configurations.
    
    Args:
        task: Benchmark task ("mnist", "cifar10", "vowel_classification")
        
    Returns:
        Configured photonic neural network
    """
    # Standard configurations
    wavelength_config = WavelengthConfig(num_channels=8)
    thermal_config = ThermalConfig()
    fabrication_config = FabricationConfig()
    
    if task == "mnist":
        # MNIST: 784 -> 128 -> 64 -> 10
        layer_configs = [
            LayerConfig(input_dim=784, output_dim=128, activation="relu"),
            LayerConfig(input_dim=128, output_dim=64, activation="relu"),
            LayerConfig(input_dim=64, output_dim=10, activation="sigmoid")
        ]
    elif task == "cifar10":
        # CIFAR-10: 3072 -> 512 -> 256 -> 10
        layer_configs = [
            LayerConfig(input_dim=3072, output_dim=512, activation="relu"),
            LayerConfig(input_dim=512, output_dim=256, activation="relu"),
            LayerConfig(input_dim=256, output_dim=10, activation="sigmoid")
        ]
    elif task == "vowel_classification":
        # 6-class vowel classification as in MIT demo: 10 -> 6 -> 6
        layer_configs = [
            LayerConfig(input_dim=10, output_dim=6, activation="relu"),
            LayerConfig(input_dim=6, output_dim=6, activation="sigmoid")
        ]
    else:
        raise ValueError(f"Unknown task: {task}")
    
    return PhotonicNeuralNetwork(
        layer_configs, wavelength_config, thermal_config, fabrication_config
    )