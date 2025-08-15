#!/usr/bin/env python3
"""
Performance Optimization Example for Photonic AI Simulator

This example demonstrates:
- Multi-backend performance comparison (CPU, GPU, JAX)
- Wavelength channel scaling analysis
- Batch processing optimization
- Memory-efficient operations
- JIT compilation acceleration
"""

import numpy as np
import sys
import os
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def performance_optimization_example():
    """Comprehensive performance optimization demonstration."""
    
    print("⚡ Performance Optimization Example")
    print("=" * 40)
    
    # Import required modules
    from core import PhotonicProcessor, WavelengthConfig, ThermalConfig, FabricationConfig
    from models import PhotonicNeuralNetwork, MZILayer
    from optimization import (
        create_optimized_network, GPUAccelerationBackend, 
        JAXAccelerationBackend, CPUOptimizationBackend,
        ParallelProcessingManager, MemoryOptimizer
    )
    from scaling import AutoScalingManager, WorkloadPredictor, ResourceType
    
    print("\n1. Backend Performance Comparison...")
    
    # Common configuration
    base_config = WavelengthConfig(
        center_wavelength=1550.0,
        wavelength_spacing=0.4,
        num_channels=8,
        bandwidth=100.0
    )
    
    thermal_config = ThermalConfig()
    fab_config = FabricationConfig()
    
    # Test data
    batch_size = 64
    input_dim = 784
    test_data = np.random.randn(batch_size, input_dim).astype(np.float32)
    
    # Initialize backends
    backends = {
        'CPU': CPUOptimizationBackend(),
        'GPU': GPUAccelerationBackend() if GPUAccelerationBackend.is_available() else None,
        'JAX': JAXAccelerationBackend() if JAXAccelerationBackend.is_available() else None
    }
    
    # Performance results
    performance_results = {}
    
    for backend_name, backend in backends.items():
        if backend is None:
            print(f"   ⚠️  {backend_name} backend not available")
            continue
            
        print(f"   🔬 Testing {backend_name} backend...")
        
        # Create optimized model for this backend
        model = create_optimized_network(
            "mnist", 
            optimization_level="high",
            backend=backend,
            wavelength_config=base_config
        )
        
        # Warmup runs
        for _ in range(3):
            _ = model.forward(test_data[:4])
        
        # Benchmark inference
        num_iterations = 50
        latencies = []
        
        for i in range(num_iterations):
            start_time = time.perf_counter()
            outputs = model.forward(test_data)
            end_time = time.perf_counter()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
        
        # Calculate statistics
        avg_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        throughput = batch_size / (avg_latency / 1000)  # samples/sec
        
        performance_results[backend_name] = {
            'latency_ms': avg_latency,
            'latency_std': std_latency,
            'throughput_samples_per_sec': throughput,
            'memory_usage_mb': backend.get_memory_usage() if hasattr(backend, 'get_memory_usage') else 0
        }
        
        print(f"      Latency: {avg_latency:.2f} ± {std_latency:.2f} ms")
        print(f"      Throughput: {throughput:.0f} samples/sec")
    
    # 2. Wavelength Channel Scaling Analysis
    print("\n2. Wavelength Channel Scaling Analysis...")
    
    channel_counts = [4, 8, 16, 32, 64]
    scaling_results = []
    
    # Use best performing backend
    best_backend = backends.get('JAX') or backends.get('GPU') or backends.get('CPU')
    
    for num_channels in channel_counts:
        print(f"   Testing {num_channels} wavelength channels...")
        
        # Create scaled configuration
        scaled_config = WavelengthConfig(
            center_wavelength=1550.0,
            wavelength_spacing=0.4,
            num_channels=num_channels,
            bandwidth=100.0
        )
        
        # Create scaled model
        layers = [
            MZILayer(input_dim=784, output_dim=min(512, num_channels*16), wavelength_config=scaled_config),
            MZILayer(input_dim=min(512, num_channels*16), output_dim=10, wavelength_config=scaled_config)
        ]
        
        processor = PhotonicProcessor(scaled_config, thermal_config, fab_config, enable_noise=False)
        model = PhotonicNeuralNetwork(layers, processor)
        
        # Optimize for this configuration
        if best_backend:
            model = best_backend.optimize_model(model)
        
        # Benchmark scaled performance
        scaled_test_data = np.random.randn(32, 784).astype(np.float32)
        
        # Warmup
        _ = model.forward(scaled_test_data[:4])
        
        # Benchmark
        start_time = time.perf_counter()
        for _ in range(10):
            _ = model.forward(scaled_test_data)
        end_time = time.perf_counter()
        
        avg_time_per_batch = (end_time - start_time) / 10
        throughput = len(scaled_test_data) / avg_time_per_batch
        
        # Calculate theoretical vs actual speedup
        theoretical_speedup = num_channels / channel_counts[0]
        actual_speedup = throughput / (scaling_results[0]['throughput'] if scaling_results else throughput)
        parallel_efficiency = actual_speedup / theoretical_speedup if theoretical_speedup > 0 else 1.0
        
        scaling_results.append({
            'channels': num_channels,
            'throughput': throughput,
            'latency_ms': avg_time_per_batch * 1000,
            'theoretical_speedup': theoretical_speedup,
            'actual_speedup': actual_speedup,
            'parallel_efficiency': parallel_efficiency
        })
        
        print(f"      Throughput: {throughput:.0f} samples/sec")
        print(f"      Efficiency: {parallel_efficiency:.2f}")
    
    # 3. Batch Processing Optimization
    print("\n3. Batch Processing Optimization...")
    
    batch_sizes = [1, 4, 16, 64, 256, 1024]
    batch_results = []
    
    # Use optimal configuration from scaling analysis
    optimal_channels = 16  # Good balance of performance vs complexity
    optimal_config = WavelengthConfig(
        center_wavelength=1550.0,
        wavelength_spacing=0.4,
        num_channels=optimal_channels,
        bandwidth=100.0
    )
    
    # Create optimized model
    model = create_optimized_network(
        "mnist",
        optimization_level="high",
        backend=best_backend,
        wavelength_config=optimal_config
    )
    
    for batch_size in batch_sizes:
        print(f"   Testing batch size {batch_size}...")
        
        # Create test data
        batch_data = np.random.randn(batch_size, 784).astype(np.float32)
        
        # Warmup
        if batch_size <= 64:  # Avoid memory issues on warmup
            _ = model.forward(batch_data[:min(4, batch_size)])
        
        # Benchmark
        num_runs = max(10, 100 // batch_size)  # Fewer runs for larger batches
        
        start_time = time.perf_counter()
        for _ in range(num_runs):
            _ = model.forward(batch_data)
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        time_per_sample = total_time / (num_runs * batch_size)
        throughput = 1.0 / time_per_sample
        
        batch_results.append({
            'batch_size': batch_size,
            'time_per_sample_ms': time_per_sample * 1000,
            'throughput_samples_per_sec': throughput,
            'total_time_s': total_time
        })
        
        print(f"      Time per sample: {time_per_sample * 1000:.3f} ms")
        print(f"      Throughput: {throughput:.0f} samples/sec")
    
    # 4. Memory Optimization
    print("\n4. Memory Optimization Analysis...")
    
    memory_optimizer = MemoryOptimizer()
    
    # Test different memory optimization strategies
    strategies = ['baseline', 'gradient_checkpointing', 'mixed_precision', 'memory_pooling']
    memory_results = {}
    
    for strategy in strategies:
        print(f"   Testing {strategy} strategy...")
        
        # Configure memory optimization
        if strategy == 'gradient_checkpointing':
            memory_optimizer.enable_gradient_checkpointing(True)
        elif strategy == 'mixed_precision':
            memory_optimizer.enable_mixed_precision(True)
        elif strategy == 'memory_pooling':
            memory_optimizer.enable_memory_pooling(True)
        
        # Create test model
        test_config = WavelengthConfig(num_channels=32)  # Larger model for memory test
        
        layers = [
            MZILayer(input_dim=784, output_dim=1024, wavelength_config=test_config),
            MZILayer(input_dim=1024, output_dim=512, wavelength_config=test_config),
            MZILayer(input_dim=512, output_dim=10, wavelength_config=test_config)
        ]
        
        processor = PhotonicProcessor(test_config, thermal_config, fab_config)
        model = PhotonicNeuralNetwork(layers, processor)
        
        # Apply memory optimization
        if strategy != 'baseline':
            model = memory_optimizer.optimize_memory_usage(model)
        
        # Measure memory usage during inference
        test_batch = np.random.randn(128, 784).astype(np.float32)
        
        # Simulate memory measurement
        initial_memory = memory_optimizer.get_memory_usage()
        _ = model.forward(test_batch)
        peak_memory = memory_optimizer.get_memory_usage()
        
        memory_usage_mb = peak_memory - initial_memory
        
        memory_results[strategy] = {
            'memory_usage_mb': memory_usage_mb,
            'memory_efficiency': 1.0 / memory_usage_mb if memory_usage_mb > 0 else float('inf')
        }
        
        print(f"      Memory usage: {memory_usage_mb:.1f} MB")
        
        # Reset optimizer
        memory_optimizer.reset()
    
    # 5. Autoscaling Demonstration
    print("\n5. Autoscaling and Dynamic Resource Management...")
    
    # Initialize autoscaling manager
    scaling_manager = AutoScalingManager()
    workload_predictor = WorkloadPredictor()
    
    # Simulate varying workload
    print("   Simulating variable workload...")
    
    workload_pattern = [10, 50, 200, 500, 800, 1000, 800, 400, 100, 50]  # Requests per second
    scaling_decisions = []
    
    for t, load in enumerate(workload_pattern):
        # Predict next workload
        predicted_load = workload_predictor.predict_next_load(workload_pattern[:t+1])
        
        # Make scaling decision
        current_capacity = 100 * (t + 1)  # Simulate growing capacity
        decision = scaling_manager.make_scaling_decision(
            current_load=load,
            predicted_load=predicted_load,
            current_capacity=current_capacity,
            resource_type=ResourceType.photonic_cores
        )
        
        scaling_decisions.append({
            'time': t,
            'current_load': load,
            'predicted_load': predicted_load,
            'scaling_action': decision.action.value,
            'target_capacity': decision.target_capacity
        })
        
        print(f"      t={t}: Load={load}, Predicted={predicted_load:.0f}, Action={decision.action.value}")
    
    # 6. Performance Summary and Recommendations
    print("\n6. Performance Analysis Summary...")
    
    # Find best performing backend
    if performance_results:
        best_perf = min(performance_results.items(), key=lambda x: x[1]['latency_ms'])
        print(f"   🏆 Best Backend: {best_perf[0]} ({best_perf[1]['latency_ms']:.2f}ms latency)")
    
    # Find optimal batch size
    if batch_results:
        optimal_batch = max(batch_results, key=lambda x: x['throughput_samples_per_sec'])
        print(f"   📊 Optimal Batch Size: {optimal_batch['batch_size']} ({optimal_batch['throughput_samples_per_sec']:.0f} samples/sec)")
    
    # Find optimal channel count
    if scaling_results:
        optimal_scale = max(scaling_results, key=lambda x: x['parallel_efficiency'])
        print(f"   🔀 Optimal Wavelength Channels: {optimal_scale['channels']} (efficiency: {optimal_scale['parallel_efficiency']:.2f})")
    
    # Memory optimization recommendation
    if memory_results:
        best_memory = min(memory_results.items(), key=lambda x: x[1]['memory_usage_mb'])
        print(f"   💾 Best Memory Strategy: {best_memory[0]} ({best_memory[1]['memory_usage_mb']:.1f}MB)")
    
    # Generate optimization recommendations
    recommendations = []
    
    if performance_results:
        if 'JAX' in performance_results:
            recommendations.append("Use JAX backend for maximum performance")
        elif 'GPU' in performance_results:
            recommendations.append("Use GPU acceleration when available")
    
    if batch_results:
        optimal_batch_size = max(batch_results, key=lambda x: x['throughput_samples_per_sec'])['batch_size']
        recommendations.append(f"Use batch size {optimal_batch_size} for optimal throughput")
    
    if scaling_results:
        efficient_channels = [r for r in scaling_results if r['parallel_efficiency'] > 0.8]
        if efficient_channels:
            best_channels = max(efficient_channels, key=lambda x: x['channels'])['channels']
            recommendations.append(f"Use {best_channels} wavelength channels for balanced performance")
    
    recommendations.append("Enable memory optimization for large models")
    recommendations.append("Implement autoscaling for production deployments")
    
    print(f"\n💡 Optimization Recommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec}")
    
    # Export performance profile
    performance_profile = {
        'backend_comparison': performance_results,
        'scaling_analysis': scaling_results,
        'batch_optimization': batch_results,
        'memory_optimization': memory_results,
        'autoscaling_simulation': scaling_decisions,
        'recommendations': recommendations,
        'optimal_configuration': {
            'backend': best_perf[0] if performance_results else 'CPU',
            'batch_size': optimal_batch['batch_size'] if batch_results else 64,
            'wavelength_channels': optimal_scale['channels'] if scaling_results else 16,
            'memory_strategy': best_memory[0] if memory_results else 'baseline'
        }
    }
    
    print(f"\n🎯 Performance optimization analysis complete!")
    return performance_profile


if __name__ == "__main__":
    try:
        profile = performance_optimization_example()
        optimal = profile['optimal_configuration']
        print(f"\n✅ Optimal configuration identified:")
        print(f"   Backend: {optimal['backend']}")
        print(f"   Batch size: {optimal['batch_size']}")
        print(f"   Wavelength channels: {optimal['wavelength_channels']}")
        print(f"   Memory strategy: {optimal['memory_strategy']}")
    except Exception as e:
        print(f"\n❌ Performance optimization failed: {e}")
        import traceback
        traceback.print_exc()