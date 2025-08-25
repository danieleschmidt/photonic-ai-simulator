"""
Tests for photonic neural network models.
"""

import pytest
import numpy as np
from src.models import (
    PhotonicNeuralNetwork, MZILayer, LayerConfig, 
    create_benchmark_network
)
from src.core import WavelengthConfig, ThermalConfig, FabricationConfig


class TestLayerConfig:
    """Test layer configuration."""
    
    def test_default_config(self):
        """Test default layer configuration."""
        config = LayerConfig(input_dim=10, output_dim=5)
        assert config.input_dim == 10
        assert config.output_dim == 5
        assert config.activation == "relu"
        assert config.weight_precision == 4


class TestMZILayer:
    """Test MZI layer functionality."""
    
    @pytest.fixture
    def layer_config(self):
        """Create test layer configuration."""
        return LayerConfig(input_dim=4, output_dim=3, activation="relu")
    
    @pytest.fixture
    def configs(self):
        """Create test configurations."""
        wavelength_config = WavelengthConfig(num_channels=2)
        thermal_config = ThermalConfig()
        fabrication_config = FabricationConfig()
        return wavelength_config, thermal_config, fabrication_config
    
    @pytest.fixture
    def mzi_layer(self, layer_config, configs):
        """Create test MZI layer."""
        wavelength_config, thermal_config, fabrication_config = configs
        return MZILayer(layer_config, wavelength_config, thermal_config, fabrication_config)
    
    def test_layer_initialization(self, mzi_layer, layer_config):
        """Test MZI layer initialization."""
        assert mzi_layer.config == layer_config
        assert mzi_layer.weights.shape == (4, 3, 2)  # input_dim, output_dim, num_channels
        assert mzi_layer.biases.shape == (3, 2)  # output_dim, num_channels
        assert mzi_layer.is_training == True
    
    def test_weight_quantization(self, mzi_layer):
        """Test weight quantization."""
        weights = np.random.randn(5, 5)
        quantized = mzi_layer._quantize_weights(weights, 4)
        
        # Should be different from original (unless very lucky)
        # Should be finite and bounded
        assert np.all(np.isfinite(quantized))
        assert np.max(np.abs(quantized)) <= np.max(np.abs(weights))
    
    def test_forward_pass_basic(self, mzi_layer):
        """Test basic forward pass."""
        batch_size = 3
        input_dim = 4
        inputs = np.random.randn(batch_size, input_dim)
        
        outputs = mzi_layer.forward(inputs)
        
        assert outputs.shape == (batch_size, 3, 2)  # batch, output_dim, wavelengths
        assert np.all(np.isfinite(outputs))
    
    def test_forward_pass_validation(self, mzi_layer):
        """Test forward pass input validation."""
        # Test non-array input
        with pytest.raises(ValueError, match="Inputs must be numpy array"):
            mzi_layer.forward("invalid")
        
        # Test empty array
        with pytest.raises(ValueError, match="Input array cannot be empty"):
            mzi_layer.forward(np.array([]))
        
        # Test non-finite values
        invalid_input = np.array([[1.0, np.inf, 2.0, 3.0]])
        with pytest.raises(ValueError, match="Input contains non-finite values"):
            mzi_layer.forward(invalid_input)
    
    def test_backward_pass(self, mzi_layer):
        """Test backward pass."""
        batch_size = 2
        inputs = np.random.randn(batch_size, 4)
        outputs = mzi_layer.forward(inputs)
        
        grad_output = np.random.randn(*outputs.shape).astype(complex)
        grad_input = mzi_layer.backward(grad_output, inputs)
        
        # Grad input should match input shape
        assert grad_input.shape == inputs.shape
        assert not np.isnan(grad_input).any()
        assert len(mzi_layer.weight_updates) == 1
    
    def test_weight_updates(self, mzi_layer):
        """Test weight update mechanism."""
        # Perform forward and backward pass
        inputs = np.random.randn(2, 4)
        outputs = mzi_layer.forward(inputs)
        grad_output = np.random.randn(*outputs.shape).astype(complex)
        mzi_layer.backward(grad_output, inputs)
        
        # Store original weights
        original_weights = mzi_layer.weights.copy()
        
        # Update weights
        mzi_layer.update_weights(learning_rate=0.1)
        
        # Weights should have changed
        assert not np.array_equal(original_weights, mzi_layer.weights)
        assert len(mzi_layer.weight_updates) == 0  # Should be cleared
    
    def test_parameter_count(self, mzi_layer):
        """Test parameter count calculation."""
        param_count = mzi_layer.get_parameter_count()
        expected_count = 4 * 3 * 2 + 3 * 2  # weights + biases
        assert param_count == expected_count


class TestPhotonicNeuralNetwork:
    """Test complete photonic neural network."""
    
    @pytest.fixture
    def layer_configs(self):
        """Create test layer configurations."""
        return [
            LayerConfig(input_dim=10, output_dim=8, activation="relu"),
            LayerConfig(input_dim=8, output_dim=4, activation="sigmoid")
        ]
    
    @pytest.fixture
    def configs(self):
        """Create test configurations."""
        wavelength_config = WavelengthConfig(num_channels=2)
        thermal_config = ThermalConfig()
        fabrication_config = FabricationConfig()
        return wavelength_config, thermal_config, fabrication_config
    
    @pytest.fixture
    def network(self, layer_configs, configs):
        """Create test network."""
        wavelength_config, thermal_config, fabrication_config = configs
        return PhotonicNeuralNetwork(
            layer_configs, wavelength_config, thermal_config, fabrication_config
        )
    
    def test_network_initialization(self, network, layer_configs):
        """Test network initialization."""
        assert len(network.layers) == 2
        assert network.layer_configs == layer_configs
        assert network.is_training == True
    
    def test_forward_propagation(self, network):
        """Test forward propagation through network."""
        batch_size = 5
        input_dim = 10
        inputs = np.random.randn(batch_size, input_dim)
        
        outputs, metrics = network.forward(inputs)
        
        assert outputs.shape == (batch_size, 4)  # Final output dimension
        assert np.all(np.isfinite(outputs))
        assert "total_latency_ns" in metrics
        assert "total_power_mw" in metrics
        assert metrics["total_latency_ns"] > 0
    
    def test_prediction(self, network):
        """Test prediction functionality."""
        inputs = np.random.randn(3, 10)
        
        network.set_training(True)
        assert network.is_training == True
        
        predictions = network.predict(inputs)
        assert network.is_training == False
        assert predictions.shape == (3, 4)
    
    def test_parameter_counting(self, network):
        """Test total parameter counting."""
        total_params = network.get_total_parameters()
        
        # Calculate expected parameters
        layer1_params = 10 * 8 * 2 + 8 * 2  # weights + biases
        layer2_params = 8 * 4 * 2 + 4 * 2
        expected_total = layer1_params + layer2_params
        
        assert total_params == expected_total
    
    def test_network_summary(self, network):
        """Test network summary generation."""
        summary = network.get_network_summary()
        
        assert summary["total_layers"] == 2
        assert summary["wavelength_channels"] == 2
        assert summary["operating_wavelength_nm"] == 1550.0
        assert len(summary["layer_details"]) == 2
        assert summary["thermal_compensation"] == True
    
    def test_model_save_load(self, network, tmp_path):
        """Test model save and load functionality."""
        # Perform forward pass to initialize some state
        inputs = np.random.randn(2, 10)
        network.forward(inputs)
        
        # Save model
        save_path = tmp_path / "test_model.npy"
        network.save_model(str(save_path))
        assert save_path.exists()
        
        # Create new network and load
        new_network = PhotonicNeuralNetwork(
            network.layer_configs,
            network.wavelength_config,
            network.thermal_config,
            network.fabrication_config
        )
        new_network.load_model(str(save_path))
        
        # Compare weights
        for i, (old_layer, new_layer) in enumerate(zip(network.layers, new_network.layers)):
            assert np.array_equal(old_layer.weights, new_layer.weights)
            assert np.array_equal(old_layer.biases, new_layer.biases)


class TestBenchmarkNetworks:
    """Test benchmark network creation."""
    
    def test_mnist_network(self):
        """Test MNIST benchmark network creation."""
        network = create_benchmark_network("mnist")
        
        assert len(network.layers) == 3
        assert network.layers[0].config.input_dim == 784
        assert network.layers[-1].config.output_dim == 10
        
        # Test forward pass
        inputs = np.random.randn(5, 784)
        outputs, metrics = network.forward(inputs)
        assert outputs.shape == (5, 10)
    
    def test_cifar10_network(self):
        """Test CIFAR-10 benchmark network creation."""
        network = create_benchmark_network("cifar10")
        
        assert len(network.layers) == 3
        assert network.layers[0].config.input_dim == 3072
        assert network.layers[-1].config.output_dim == 10
        
        # Test forward pass
        inputs = np.random.randn(3, 3072)
        outputs, metrics = network.forward(inputs)
        assert outputs.shape == (3, 10)
    
    def test_vowel_classification_network(self):
        """Test vowel classification benchmark network creation."""
        network = create_benchmark_network("vowel_classification")
        
        assert len(network.layers) == 2
        assert network.layers[0].config.input_dim == 10
        assert network.layers[-1].config.output_dim == 6
        
        # Test forward pass
        inputs = np.random.randn(4, 10)
        outputs, metrics = network.forward(inputs)
        assert outputs.shape == (4, 6)
    
    def test_invalid_task(self):
        """Test invalid task handling."""
        with pytest.raises(ValueError, match="Unknown task"):
            create_benchmark_network("invalid_task")