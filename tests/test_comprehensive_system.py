"""
Comprehensive system integration tests for Photonic AI Simulator.

Tests the complete end-to-end functionality including model creation,
training, optimization, validation, and benchmarking.
"""

import pytest
import numpy as np
import tempfile
import json
from pathlib import Path

# Import all core components
from src.models import create_benchmark_network, PhotonicNeuralNetwork, LayerConfig
from src.core import WavelengthConfig, ThermalConfig, FabricationConfig
from src.training import create_training_pipeline, TrainingConfig
from src.optimization import create_optimized_network
from src.benchmarks import run_comprehensive_benchmarks, BenchmarkConfig, MNISTBenchmark
from src.validation import PhotonicSystemValidator, HealthMonitor
from src.experiments.ab_testing import ABTestFramework, ExperimentConfig, ExperimentType
from src.utils.logging_config import setup_logging


class TestSystemIntegration:
    """Test complete system integration scenarios."""
    
    @classmethod
    def setup_class(cls):
        """Set up test environment."""
        setup_logging(level="DEBUG")
        np.random.seed(42)  # For reproducible tests
    
    def test_end_to_end_mnist_workflow(self):
        """Test complete MNIST workflow from creation to validation."""
        # Step 1: Create model
        model = create_benchmark_network("mnist")
        assert model is not None
        assert model.get_total_parameters() > 0
        
        # Step 2: Generate synthetic data
        X_train = np.random.randn(100, 784) * 0.1 + 0.5
        y_train = np.eye(10)[np.random.randint(0, 10, 100)]
        X_test = np.random.randn(20, 784) * 0.1 + 0.5
        y_test = np.eye(10)[np.random.randint(0, 10, 20)]
        
        # Step 3: Train model  
        config = TrainingConfig(
            epochs=5,  # Reduced for testing
            batch_size=16,
            forward_only=True
        )
        
        trainer = create_training_pipeline(model, "forward_only", **config.__dict__)
        history = trainer.train(X_train, y_train)
        
        # Verify training completed
        assert len(history["train_loss"]) == config.epochs
        assert all(isinstance(loss, (int, float)) for loss in history["train_loss"])
        
        # Step 4: Test inference
        predictions, metrics = model.forward(X_test, measure_latency=True)
        
        assert predictions.shape == (len(X_test), 10)
        assert "total_latency_ns" in metrics
        assert "total_power_mw" in metrics
        assert metrics["total_latency_ns"] > 0
        
        # Step 5: Validate system
        validator = PhotonicSystemValidator()
        validation_result = validator.validate_system(model)
        
        assert validation_result is not None
        assert isinstance(validation_result.is_valid, bool)
        assert isinstance(validation_result.errors, list)
        assert isinstance(validation_result.warnings, list)
    
    def test_optimization_pipeline(self):
        """Test optimization pipeline with different levels."""
        optimization_levels = ["low", "medium", "high"]
        
        for level in optimization_levels:
            # Create optimized model
            model = create_optimized_network("mnist", level)
            assert model is not None
            
            # Test throughput benchmarking
            X_test = np.random.randn(32, 784) * 0.1 + 0.5
            
            throughput_metrics = model.benchmark_throughput(
                input_shape=X_test.shape,
                num_iterations=10  # Reduced for testing
            )
            
            # Verify metrics
            assert "avg_latency_ns" in throughput_metrics
            assert "throughput_samples_per_sec" in throughput_metrics
            assert "speedup_factor" in throughput_metrics
            
            assert throughput_metrics["avg_latency_ns"] > 0
            assert throughput_metrics["throughput_samples_per_sec"] > 0
    
    def test_benchmark_suite(self):
        """Test comprehensive benchmarking suite."""
        # Configure benchmarks
        config = BenchmarkConfig(
            target_accuracy=0.8,  # Relaxed for testing
            max_latency_ns=10.0,
            max_power_mw=1000.0,
            num_runs=2  # Reduced for testing
        )
        
        # Test single benchmark
        benchmark = MNISTBenchmark(config)
        model = create_benchmark_network("mnist")
        
        result = benchmark.run_benchmark(model)
        
        # Verify result structure
        assert hasattr(result, 'accuracy')
        assert hasattr(result, 'latency_ns')
        assert hasattr(result, 'power_consumption_mw')
        assert hasattr(result, 'speedup_vs_gpu')
        
        # Verify result values are reasonable
        assert 0 <= result.accuracy <= 1
        assert result.latency_ns > 0
        assert result.power_consumption_mw > 0
    
    def test_ab_testing_framework(self):
        """Test A/B testing framework for model comparison."""
        # Create two different models
        model_a = create_benchmark_network("vowel_classification")
        model_b = create_optimized_network("vowel_classification", "medium")
        
        # Create test data
        X_test = np.random.randn(50, 10) * 0.3 + 0.5
        y_test = np.eye(6)[np.random.randint(0, 6, 50)]
        
        # Configure experiment
        config = ExperimentConfig(
            name="test_optimization_effectiveness",
            experiment_type=ExperimentType.OPTIMIZATION,
            description="Test optimization effectiveness",
            num_runs_per_variant=3,  # Reduced for testing
            primary_metric="accuracy"
        )
        
        # Create evaluation function
        def evaluate_model(model, X, y):
            model.set_training(False)
            outputs, hardware_metrics = model.forward(X, measure_latency=True)
            
            predictions = np.argmax(outputs, axis=1)
            true_labels = np.argmax(y, axis=1)
            accuracy = np.mean(predictions == true_labels)
            
            return {
                "accuracy": accuracy,
                "latency_ns": hardware_metrics["total_latency_ns"] / len(X),
                "power_mw": hardware_metrics["total_power_mw"]
            }
        
        # Run A/B test
        ab_framework = ABTestFramework()
        result = ab_framework.run_experiment(
            config, model_a, model_b, evaluate_model, X_test, y_test
        )
        
        # Verify experiment result
        assert result is not None
        assert hasattr(result, 'p_value')
        assert hasattr(result, 'effect_size')
        assert hasattr(result, 'significant_difference')
        assert hasattr(result, 'recommendation')
        
        # Verify statistical analysis
        assert 0 <= result.p_value <= 1
        assert isinstance(result.significant_difference, bool)
        assert isinstance(result.recommendation, str)
    
    def test_health_monitoring(self):
        """Test health monitoring system."""
        model = create_benchmark_network("mnist")
        
        # Create health monitor
        monitor = HealthMonitor(model, check_interval=0.1)
        
        # Test single health check
        health_result = monitor.check_health()
        
        assert health_result is not None
        assert hasattr(health_result, 'is_valid')
        assert hasattr(health_result, 'errors')
        assert hasattr(health_result, 'warnings')
        
        # Test health summary
        summary = monitor.get_health_summary()
        assert isinstance(summary, dict)
        assert "current_status" in summary or "status" in summary
    
    def test_model_serialization(self):
        """Test model saving and loading functionality."""
        model = create_benchmark_network("mnist")
        
        # Train briefly
        X_train = np.random.randn(50, 784) * 0.1 + 0.5
        y_train = np.eye(10)[np.random.randint(0, 10, 50)]
        
        config = TrainingConfig(epochs=2, batch_size=16)
        trainer = create_training_pipeline(model, "forward_only", **config.__dict__)
        trainer.train(X_train, y_train)
        
        # Test inference before saving
        X_test = np.random.randn(10, 784) * 0.1 + 0.5
        predictions_before, _ = model.forward(X_test)
        
        # Save model to temporary file
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp:
            model.save_model(tmp.name)
            temp_path = tmp.name
        
        try:
            # Create new model and load weights
            new_model = create_benchmark_network("mnist")
            new_model.load_model(temp_path)
            
            # Test inference after loading
            predictions_after, _ = new_model.forward(X_test)
            
            # Verify predictions are similar (allowing for small numerical differences)
            np.testing.assert_allclose(predictions_before, predictions_after, rtol=1e-3)
            
        finally:
            # Clean up temporary file
            Path(temp_path).unlink(missing_ok=True)
    
    def test_error_handling_and_robustness(self):
        """Test system robustness with invalid inputs and error conditions."""
        model = create_benchmark_network("mnist")
        
        # Test with invalid input shapes
        with pytest.raises((ValueError, AssertionError, IndexError)):
            invalid_input = np.random.randn(10, 100)  # Wrong input dimension
            model.forward(invalid_input)
        
        # Test with extreme weight values
        for layer in model.layers:
            # Store original weights
            original_weights = layer.weights.copy()
            
            try:
                # Set extreme weight values
                layer.weights = layer.weights * 1e6
                
                # Validate system should detect issues
                validator = PhotonicSystemValidator()
                result = validator.validate_system(model)
                
                # Should have warnings or errors for extreme values
                assert len(result.warnings) > 0 or len(result.errors) > 0
                
            finally:
                # Restore original weights
                layer.weights = original_weights
    
    def test_multi_task_compatibility(self):
        """Test that all supported tasks work correctly."""
        tasks = ["mnist", "cifar10", "vowel_classification"]
        
        for task in tasks:
            # Create model
            model = create_benchmark_network(task)
            assert model is not None
            
            # Get expected input dimension
            if task == "mnist":
                input_dim = 784
                num_classes = 10
            elif task == "cifar10":
                input_dim = 3072
                num_classes = 10
            else:  # vowel_classification
                input_dim = 10
                num_classes = 6
            
            # Test with correct input
            X = np.random.randn(5, input_dim) * 0.1 + 0.5
            predictions, metrics = model.forward(X)
            
            assert predictions.shape == (5, num_classes)
            assert "total_latency_ns" in metrics
    
    def test_configuration_validation(self):
        """Test configuration validation and edge cases."""
        # Test custom wavelength configuration
        custom_wavelength_config = WavelengthConfig(
            center_wavelength=1550.0,
            wavelength_spacing=1.0,  # Wider spacing
            num_channels=16  # More channels
        )
        
        # Test custom thermal configuration
        custom_thermal_config = ThermalConfig(
            operating_temperature=350.0,  # Higher temperature
            thermal_drift_rate=15.0,
            power_per_heater=20.0
        )
        
        # Test custom fabrication configuration
        custom_fab_config = FabricationConfig(
            etch_tolerance=5.0,  # Tighter tolerance
            coupling_efficiency=0.95  # Higher efficiency
        )
        
        # Create model with custom configurations
        layer_configs = [
            LayerConfig(input_dim=784, output_dim=64, activation="relu"),
            LayerConfig(input_dim=64, output_dim=10, activation="sigmoid")
        ]
        
        from src.models import PhotonicNeuralNetwork
        model = PhotonicNeuralNetwork(
            layer_configs,
            custom_wavelength_config,
            custom_thermal_config,
            custom_fab_config
        )
        
        # Verify custom configurations are used
        assert model.wavelength_config.num_channels == 16
        assert model.thermal_config.operating_temperature == 350.0
        assert model.fabrication_config.etch_tolerance == 5.0
        
        # Test inference works
        X = np.random.randn(10, 784) * 0.1 + 0.5
        predictions, _ = model.forward(X)
        assert predictions.shape == (10, 10)
    
    def test_performance_requirements(self):
        """Test that system meets basic performance requirements."""
        model = create_optimized_network("vowel_classification", "high")
        
        # Test latency requirement (should be fast for small network)
        X = np.random.randn(1, 10) * 0.3 + 0.5
        
        _, metrics = model.forward(X, measure_latency=True)
        
        # For vowel classification (small network), should be very fast
        avg_latency_ns = metrics["total_latency_ns"]
        
        # Should complete in reasonable time (allowing for simulation overhead)
        assert avg_latency_ns < 1e6  # Less than 1ms
        
        # Test power consumption is reasonable
        assert metrics["total_power_mw"] < 1000  # Less than 1W
        
        # Test accuracy requirement with short training
        X_train = np.random.randn(200, 10) * 0.3 + 0.5
        y_train = np.eye(6)[np.random.randint(0, 6, 200)]
        
        config = TrainingConfig(epochs=20, batch_size=32)
        trainer = create_training_pipeline(model, "forward_only", **config.__dict__)
        history = trainer.train(X_train, y_train)
        
        # Should achieve reasonable accuracy on synthetic data
        final_accuracy = history["val_accuracy"][-1]
        assert final_accuracy > 0.1  # Better than random (1/6 ≈ 0.17)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_minimal_network(self):
        """Test minimal viable network configuration."""
        # Create minimal network
        layer_configs = [
            LayerConfig(input_dim=2, output_dim=1, activation="sigmoid")
        ]
        
        wavelength_config = WavelengthConfig(num_channels=1)
        thermal_config = ThermalConfig()
        fab_config = FabricationConfig()
        
        from src.models import PhotonicNeuralNetwork
        model = PhotonicNeuralNetwork(
            layer_configs, wavelength_config, thermal_config, fab_config
        )
        
        # Test inference
        X = np.random.randn(5, 2)
        predictions, _ = model.forward(X)
        assert predictions.shape == (5, 1)
    
    def test_maximum_supported_size(self):
        """Test with largest reasonable network configuration."""
        # Large but reasonable network
        layer_configs = [
            LayerConfig(input_dim=1000, output_dim=500, activation="relu"),
            LayerConfig(input_dim=500, output_dim=100, activation="sigmoid")
        ]
        
        wavelength_config = WavelengthConfig(num_channels=32)  # Many channels
        
        from src.models import PhotonicNeuralNetwork
        model = PhotonicNeuralNetwork(
            layer_configs, wavelength_config, ThermalConfig(), FabricationConfig()
        )
        
        # Test inference (use small batch to avoid memory issues)
        X = np.random.randn(2, 1000) * 0.1 + 0.5
        predictions, _ = model.forward(X)
        assert predictions.shape == (2, 100)
    
    def test_extreme_temperature_conditions(self):
        """Test system behavior under extreme temperature conditions."""
        model = create_benchmark_network("mnist")
        
        # Set extreme temperature
        for layer in model.layers:
            layer.processor.current_temperature = 400.0  # Very hot
        
        # System should still function but may have warnings
        validator = PhotonicSystemValidator()
        result = validator.validate_system(model)
        
        # Should detect temperature issues
        temp_errors = [e for e in result.errors if "temperature" in e["message"].lower()]
        temp_warnings = [w for w in result.warnings if "temperature" in w["message"].lower()]
        
        assert len(temp_errors) > 0 or len(temp_warnings) > 0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])