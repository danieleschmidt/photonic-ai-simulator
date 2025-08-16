"""
Tests for core photonic computing components.
"""

import pytest
import numpy as np
from src.core import (
    PhotonicProcessor, WavelengthConfig, ThermalConfig, 
    FabricationConfig, OpticalSignal, NoiseType
)


class TestWavelengthConfig:
    """Test wavelength configuration functionality."""
    
    def test_default_config(self):
        """Test default wavelength configuration."""
        config = WavelengthConfig()
        assert config.center_wavelength == 1550.0
        assert config.num_channels == 8
        assert len(config.wavelengths) == 8
    
    def test_custom_config(self):
        """Test custom wavelength configuration."""
        config = WavelengthConfig(
            center_wavelength=1310.0,
            num_channels=4,
            wavelength_spacing=1.6
        )
        assert config.center_wavelength == 1310.0
        assert config.num_channels == 4
        wavelengths = config.wavelengths
        assert len(wavelengths) == 4
        assert np.allclose(wavelengths[1] - wavelengths[0], 1.6)


class TestThermalConfig:
    """Test thermal management configuration."""
    
    def test_default_thermal_config(self):
        """Test default thermal configuration."""
        config = ThermalConfig()
        assert config.operating_temperature == 300.0
        assert config.power_per_heater == 15.0


class TestFabricationConfig:
    """Test fabrication configuration and noise application."""
    
    def test_fabrication_noise(self):
        """Test fabrication noise application."""
        config = FabricationConfig()
        values = np.ones(100)
        noisy_values = config.apply_fabrication_noise(values)
        
        # Should be close to original but with variation
        assert not np.array_equal(values, noisy_values)
        assert np.allclose(values, noisy_values, rtol=0.1)


class TestPhotonicProcessor:
    """Test photonic processor core functionality."""
    
    @pytest.fixture
    def processor(self):
        """Create test processor."""
        wavelength_config = WavelengthConfig(num_channels=4)
        thermal_config = ThermalConfig()
        fabrication_config = FabricationConfig()
        return PhotonicProcessor(
            wavelength_config, thermal_config, fabrication_config
        )
    
    def test_processor_initialization(self, processor):
        """Test processor initialization."""
        assert processor.wavelength_config.num_channels == 4
        assert processor.current_temperature == 300.0
        assert processor.power_consumption == 0.0
    
    def test_mzi_operation_basic(self, processor):
        """Test basic MZI operation."""
        inputs = np.array([1.0, 0.5])
        phase_shift = np.pi / 4
        
        output_1, output_2 = processor.mach_zehnder_operation(
            inputs, phase_shift
        )
        
        assert isinstance(output_1, (complex, np.complexfloating))
        assert isinstance(output_2, (complex, np.complexfloating))
        assert np.isfinite(output_1)
        assert np.isfinite(output_2)
    
    def test_mzi_operation_validation(self, processor):
        """Test MZI operation input validation."""
        # Test empty input
        with pytest.raises(ValueError, match="Invalid input array"):
            processor.mach_zehnder_operation(np.array([]), 0.5)
        
        # Test invalid phase shift
        with pytest.raises(ValueError, match="Phase shift must be a finite number"):
            processor.mach_zehnder_operation(np.array([1.0, 0.5]), np.inf)
        
        # Test invalid wavelength
        with pytest.raises(ValueError, match="Wavelength must be positive"):
            processor.mach_zehnder_operation(np.array([1.0, 0.5]), 0.5, -100)
    
    def test_wavelength_multiplexed_operation(self, processor):
        """Test wavelength multiplexed operations."""
        batch_size = 2
        input_dim = 3
        output_dim = 2
        num_wavelengths = 4
        
        inputs = np.random.randn(batch_size, input_dim, num_wavelengths)
        weights = np.random.randn(input_dim, output_dim, num_wavelengths).astype(complex)
        
        outputs = processor.wavelength_multiplexed_operation(inputs, weights)
        
        assert outputs.shape == (batch_size, output_dim, num_wavelengths)
        assert np.all(np.isfinite(outputs))
    
    def test_nonlinear_optical_function_unit(self, processor):
        """Test nonlinear optical function unit."""
        inputs = np.random.randn(10, 5).astype(complex)
        
        # Test different activation functions
        for activation in ["relu", "sigmoid", "tanh"]:
            outputs = processor.nonlinear_optical_function_unit(inputs, activation)
            assert outputs.shape == inputs.shape
            assert np.all(np.isfinite(outputs))
        
        # Test invalid activation
        with pytest.raises(ValueError, match="Unsupported activation type"):
            processor.nonlinear_optical_function_unit(inputs, "invalid")
    
    def test_thermal_drift_compensation(self, processor):
        """Test thermal drift compensation."""
        inputs = np.random.randn(5, 3).astype(complex)
        
        # Test with noise enabled
        processor.enable_noise = True
        compensated = processor.thermal_drift_compensation(inputs)
        assert compensated.shape == inputs.shape
        assert np.all(np.isfinite(compensated))
        
        # Test with noise disabled
        processor.enable_noise = False
        compensated = processor.thermal_drift_compensation(inputs)
        assert np.array_equal(compensated, inputs)
    
    def test_performance_tracking(self, processor):
        """Test performance metrics tracking."""
        # Initially empty
        metrics = processor.get_performance_metrics()
        assert metrics == {}
        
        # Test latency measurement
        inputs = np.array([1.0, 0.5])
        result, latency = processor.measure_inference_latency(
            processor.mach_zehnder_operation, inputs, np.pi/4
        )
        
        assert latency > 0
        assert len(processor.inference_times) == 1
        
        # Test metrics after measurement
        metrics = processor.get_performance_metrics()
        assert "avg_latency_ns" in metrics
        assert metrics["avg_latency_ns"] > 0


class TestOpticalSignal:
    """Test optical signal representation."""
    
    def test_optical_signal_creation(self):
        """Test optical signal creation."""
        amplitude = np.array([1.0, 0.5, 0.8])
        phase = np.array([0, np.pi/2, np.pi])
        
        signal = OpticalSignal(amplitude, phase)
        assert np.array_equal(signal.amplitude, amplitude)
        assert np.array_equal(signal.phase, phase)
    
    def test_complex_representation(self):
        """Test complex representation conversion."""
        amplitude = np.array([1.0, 1.0])
        phase = np.array([0, np.pi/2])
        
        signal = OpticalSignal(amplitude, phase)
        complex_repr = signal.complex_representation
        
        expected = np.array([1.0 + 0j, 0 + 1j])
        assert np.allclose(complex_repr, expected)
    
    def test_from_complex(self):
        """Test creation from complex representation."""
        complex_signal = np.array([1.0 + 0j, 0 + 1j, -1 + 0j])
        
        signal = OpticalSignal.from_complex(complex_signal)
        assert np.allclose(signal.amplitude, [1.0, 1.0, 1.0])
        assert np.allclose(signal.phase, [0, np.pi/2, np.pi])
    
    def test_power_calculation(self):
        """Test optical power calculation."""
        amplitude = np.array([2.0, 3.0, 1.0])
        phase = np.array([0, 0, 0])
        
        signal = OpticalSignal(amplitude, phase)
        power = signal.power()
        
        expected_power = np.array([4.0, 9.0, 1.0])
        assert np.array_equal(power, expected_power)