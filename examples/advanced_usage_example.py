#!/usr/bin/env python3
"""
Advanced Usage Example for Photonic AI Simulator

This example demonstrates:
- Multi-wavelength neural network training
- Performance optimization techniques  
- Hardware simulation with noise
- Security and access control
- A/B testing for algorithm validation
"""

import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def advanced_photonic_example():
    """Complete example showing advanced photonic AI capabilities."""
    
    print("🔬 Advanced Photonic AI Simulator Example")
    print("=" * 50)
    
    # 1. Create advanced configuration
    print("\n1. Creating Advanced Configuration...")
    from core import PhotonicProcessor, WavelengthConfig, ThermalConfig, FabricationConfig
    from models import PhotonicNeuralNetwork, MZILayer
    from training import ForwardOnlyTrainer, HardwareAwareOptimizer, TrainingConfig
    from security import create_secure_photonic_system, SecurityLevel
    from experiments.ab_testing import ABTestFramework, ExperimentConfig
    
    # High-performance wavelength configuration
    wl_config = WavelengthConfig(
        center_wavelength=1550.0,  # C-band telecom
        wavelength_spacing=0.4,    # Dense WDM spacing
        num_channels=16,           # More channels for parallelism
        bandwidth=100.0            # High bandwidth per channel
    )
    
    # Realistic thermal configuration
    thermal_config = ThermalConfig(
        operating_temperature=300.0,
        thermal_drift_rate=8.0,
        power_per_heater=12.0,
        thermal_time_constant=0.8e-3,
        max_temperature_variation=3.0
    )
    
    # Production fabrication tolerances
    fab_config = FabricationConfig(
        etch_tolerance=5.0,                    # Tighter tolerance
        refractive_index_variation=0.0005,     # Low variation
        coupling_efficiency=0.92,              # High efficiency
        propagation_loss=0.05                  # Low loss
    )
    
    print("✓ Configuration created with 16 wavelength channels")
    
    # 2. Create photonic neural network
    print("\n2. Building Photonic Neural Network...")
    
    # Create processor with noise enabled for realism
    processor = PhotonicProcessor(wl_config, thermal_config, fab_config, enable_noise=True)
    
    # Build multi-layer network architecture
    layers = [
        MZILayer(input_dim=784, output_dim=256, wavelength_config=wl_config),
        MZILayer(input_dim=256, output_dim=128, wavelength_config=wl_config),
        MZILayer(input_dim=128, output_dim=10, wavelength_config=wl_config)
    ]
    
    model = PhotonicNeuralNetwork(layers, processor)
    print(f"✓ Created {len(layers)}-layer photonic neural network")
    
    # 3. Generate synthetic training data (MNIST-like)
    print("\n3. Preparing Training Data...")
    batch_size = 100
    X_train = np.random.randn(batch_size, 784) * 0.1 + 0.5  # Normalized input
    y_train = np.eye(10)[np.random.randint(0, 10, batch_size)]  # One-hot labels
    
    X_test = np.random.randn(20, 784) * 0.1 + 0.5
    y_test = np.eye(10)[np.random.randint(0, 10, 20)]
    
    print(f"✓ Generated {batch_size} training samples and {len(X_test)} test samples")
    
    # 4. Configure hardware-aware training
    print("\n4. Configuring Hardware-Aware Training...")
    
    config = TrainingConfig(
        forward_only=True,          # Photonic-compatible training
        learning_rate=0.001,
        epochs=10,
        batch_size=32,
        weight_precision=4,         # 4-bit quantization
        enable_thermal_compensation=True,
        target_accuracy=0.85
    )
    
    optimizer = HardwareAwareOptimizer(
        learning_rate=config.learning_rate,
        thermal_config=thermal_config,
        fabrication_config=fab_config
    )
    
    trainer = ForwardOnlyTrainer(model, config, optimizer)
    print("✓ Hardware-aware training configured")
    
    # 5. Train the model
    print("\n5. Training Photonic Neural Network...")
    training_history = trainer.train(X_train, y_train, X_test, y_test)
    
    final_accuracy = training_history['test_accuracy'][-1]
    training_time = training_history['total_time']
    print(f"✓ Training completed: {final_accuracy:.3f} accuracy in {training_time:.2f}s")
    
    # 6. Security-enhanced inference
    print("\n6. Demonstrating Secure Inference...")
    
    secure_system = create_secure_photonic_system(model, SecurityLevel.INTERNAL)
    
    # Authenticate user
    session_token = secure_system.access_controller.authenticate_user(
        "researcher", "research_password123", "192.168.1.100"
    )
    
    if session_token:
        # Perform secure inference
        test_sample = X_test[:5]  # First 5 test samples
        predictions = secure_system.secure_inference(session_token, test_sample)
        print(f"✓ Secure inference completed on {len(test_sample)} samples")
        
        # Get security status
        security_status = secure_system.get_security_status()
        print(f"✓ Security status: {security_status['threat_level']} threat level")
    
    # 7. Performance benchmarking
    print("\n7. Performance Benchmarking...")
    
    # Benchmark inference latency
    import time
    latencies = []
    for _ in range(50):
        start = time.perf_counter()
        _ = model.forward(X_test[:1])
        latency = (time.perf_counter() - start) * 1e9  # nanoseconds
        latencies.append(latency)
    
    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    print(f"✓ Average inference latency: {avg_latency:.1f} ± {std_latency:.1f} ns")
    
    # Calculate throughput
    throughput = 1e9 / avg_latency  # samples per second
    print(f"✓ Peak throughput: {throughput:.0f} samples/second")
    
    # 8. A/B Testing Framework
    print("\n8. Algorithm A/B Testing...")
    
    # Configure experiment
    experiment_config = ExperimentConfig(
        name="forward_vs_backprop_training",
        num_runs_per_variant=5,
        confidence_level=0.95,
        primary_metric="accuracy"
    )
    
    def evaluation_function(model, X, y):
        """Evaluation function for A/B testing."""
        predictions = model.forward(X)
        # Convert to class predictions
        pred_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y, axis=1)
        accuracy = np.mean(pred_classes == true_classes)
        return {"accuracy": accuracy}
    
    # Create baseline model for comparison
    baseline_model = PhotonicNeuralNetwork(layers, processor)
    
    ab_framework = ABTestFramework()
    result = ab_framework.run_experiment(
        experiment_config, baseline_model, model,
        evaluation_function, X_test, y_test
    )
    
    print(f"✓ A/B test completed: p-value = {result.p_value:.4f}")
    print(f"✓ Effect size: {result.effect_size:.3f}")
    
    # 9. Performance and power analysis
    print("\n9. Performance Analysis...")
    
    performance_metrics = processor.get_performance_metrics()
    print(f"✓ Power consumption: {performance_metrics['avg_power_mw']:.1f} mW")
    print(f"✓ Energy per inference: {performance_metrics['energy_per_inference_pj']:.1f} pJ")
    
    # Calculate speedup vs traditional computing
    gpu_latency = 1.2e6  # 1.2ms typical GPU latency
    speedup = gpu_latency / avg_latency
    print(f"✓ Speedup vs GPU: {speedup:.0f}x")
    
    # 10. Export model for deployment
    print("\n10. Model Export...")
    
    model_export = {
        "architecture": "photonic_neural_network",
        "layers": len(layers),
        "wavelength_channels": wl_config.num_channels,
        "performance": {
            "accuracy": float(final_accuracy),
            "latency_ns": float(avg_latency),
            "power_mw": float(performance_metrics['avg_power_mw']),
            "throughput_samples_per_sec": float(throughput)
        },
        "hardware_config": {
            "wavelength_spacing": wl_config.wavelength_spacing,
            "thermal_tolerance": thermal_config.max_temperature_variation,
            "fabrication_tolerance": fab_config.etch_tolerance
        }
    }
    
    print("✓ Model export data prepared")
    
    print("\n🎉 Advanced example completed successfully!")
    print(f"📊 Final Results:")
    print(f"   Accuracy: {final_accuracy:.3f}")
    print(f"   Latency: {avg_latency:.1f} ns")
    print(f"   Power: {performance_metrics['avg_power_mw']:.1f} mW")
    print(f"   Speedup: {speedup:.0f}x vs GPU")
    
    return model_export


if __name__ == "__main__":
    try:
        results = advanced_photonic_example()
        print(f"\n✅ Example completed with {results['performance']['accuracy']:.1%} accuracy")
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        import traceback
        traceback.print_exc()