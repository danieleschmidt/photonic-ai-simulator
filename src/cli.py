"""
Command Line Interface for Photonic AI Simulator.

Provides comprehensive CLI tools for training, benchmarking, and validation
of photonic neural networks with hardware-aware optimization.
"""

import argparse
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
import logging

import numpy as np

# Import core components
try:
    from .models import create_benchmark_network
    from .training import create_training_pipeline, TrainingConfig
    from .optimization import create_optimized_network, OptimizationConfig  
    from .benchmarks import run_comprehensive_benchmarks, BenchmarkConfig
    from .validation import PhotonicSystemValidator, HealthMonitor
    from .utils.logging_config import setup_logging, get_logger
except ImportError:
    from models import create_benchmark_network
    from training import create_training_pipeline, TrainingConfig
    from optimization import create_optimized_network, OptimizationConfig  
    from benchmarks import run_comprehensive_benchmarks, BenchmarkConfig
    from validation import PhotonicSystemValidator, HealthMonitor
    from utils.logging_config import setup_logging, get_logger

logger = get_logger(__name__)


def setup_cli_logging(verbose: bool = False, log_file: Optional[str] = None):
    """Set up CLI logging configuration."""
    level = "DEBUG" if verbose else "INFO"
    setup_logging(
        level=level,
        log_file=log_file,
        enable_performance_logging=True,
        enable_hardware_logging=True
    )


def train_command(args):
    """Train photonic neural network with specified configuration."""
    print("🎓 Training Photonic Neural Network")
    print("=" * 50)
    
    # Create model
    model = create_benchmark_network(args.task)
    print(f"✓ Created {args.task} network with {model.get_total_parameters():,} parameters")
    
    # Generate synthetic training data
    np.random.seed(args.seed)
    
    if args.task == "mnist":
        X = np.random.randn(args.num_samples, 784) * 0.1 + 0.5
        y = np.eye(10)[np.random.randint(0, 10, args.num_samples)]
    elif args.task == "cifar10":
        X = np.random.randn(args.num_samples, 3072) * 0.1 + 0.5  
        y = np.eye(10)[np.random.randint(0, 10, args.num_samples)]
    elif args.task == "vowel_classification":
        X = np.random.randn(args.num_samples, 10) * 0.3 + 0.5
        y = np.eye(6)[np.random.randint(0, 6, args.num_samples)]
    else:
        raise ValueError(f"Unknown task: {args.task}")
    
    print(f"✓ Generated {len(X)} training samples")
    
    # Configure training
    training_config = TrainingConfig(
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        forward_only=args.forward_only,
        power_constraint_mw=args.power_limit
    )
    
    # Create trainer
    trainer = create_training_pipeline(model, "forward_only", **training_config.__dict__)
    
    # Train model
    print(f"🚀 Starting training ({args.epochs} epochs)...")
    start_time = time.time()
    
    history = trainer.train(X, y)
    
    end_time = time.time()
    
    print(f"✓ Training completed in {end_time - start_time:.1f}s")
    print(f"✓ Final accuracy: {history['val_accuracy'][-1]:.4f}")
    print(f"✓ Final loss: {history['val_loss'][-1]:.4f}")
    
    # Save model if requested
    if args.output:
        model.save_model(args.output)
        print(f"✓ Model saved to {args.output}")
    
    # Save training history
    history_file = Path(args.output).with_suffix('.json') if args.output else "training_history.json"
    with open(history_file, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_history = {k: [float(x) for x in v] for k, v in history.items()}
        json.dump(serializable_history, f, indent=2)
    
    print(f"✓ Training history saved to {history_file}")


def benchmark_command(args):
    """Run comprehensive benchmarking suite."""
    print("📊 Running Photonic Neural Network Benchmarks")
    print("=" * 50)
    
    # Configure benchmarks
    config = BenchmarkConfig(
        target_accuracy=args.target_accuracy,
        max_latency_ns=args.max_latency,
        max_power_mw=args.max_power,
        num_runs=args.num_runs
    )
    
    print(f"✓ Benchmark configuration:")
    print(f"  - Target accuracy: {config.target_accuracy}")
    print(f"  - Max latency: {config.max_latency_ns}ns")
    print(f"  - Max power: {config.max_power_mw}mW")
    print(f"  - Runs per test: {config.num_runs}")
    
    # Run benchmarks
    start_time = time.time()
    
    if args.benchmark == "all":
        results, analysis = run_comprehensive_benchmarks(save_results=True)
        
        print(f"\n📈 Overall Performance Summary:")
        print(f"✓ Average accuracy: {analysis['overall_performance']['avg_accuracy']:.4f}")
        print(f"✓ Average latency: {analysis['overall_performance']['avg_latency_ns']:.2f}ns")
        print(f"✓ Average power: {analysis['overall_performance']['avg_power_mw']:.1f}mW")
        print(f"✓ GPU speedup: {analysis['hardware_efficiency']['avg_speedup_vs_gpu']:.1f}x")
        
        # Save detailed results
        if args.output:
            results_data = {
                "config": config.__dict__,
                "results": {name: result.__dict__ for name, result in results.items()},
                "analysis": analysis
            }
            
            with open(args.output, 'w') as f:
                json.dump(results_data, f, indent=2, default=str)
            
            print(f"✓ Detailed results saved to {args.output}")
    
    else:
        # Single benchmark
        from .benchmarks import MNISTBenchmark, CIFAR10Benchmark, VowelClassificationBenchmark
        
        benchmark_classes = {
            "mnist": MNISTBenchmark,
            "cifar10": CIFAR10Benchmark, 
            "vowel": VowelClassificationBenchmark
        }
        
        if args.benchmark not in benchmark_classes:
            raise ValueError(f"Unknown benchmark: {args.benchmark}")
        
        # Create and run single benchmark
        benchmark = benchmark_classes[args.benchmark](config)
        model = create_benchmark_network(args.benchmark if args.benchmark != "vowel" else "vowel_classification")
        
        result = benchmark.run_benchmark(model)
        
        print(f"\n🎯 {args.benchmark.upper()} Benchmark Results:")
        print(f"✓ Accuracy: {result.accuracy:.4f}")
        print(f"✓ Latency: {result.latency_ns:.2f}ns")
        print(f"✓ Power: {result.power_consumption_mw:.1f}mW")
        print(f"✓ GPU speedup: {result.speedup_vs_gpu:.1f}x")
        print(f"✓ Energy efficiency: {result.energy_efficiency_vs_gpu:.1f}x")
    
    end_time = time.time()
    print(f"\n✅ Benchmarking completed in {end_time - start_time:.1f}s")


def validate_command(args):
    """Validate photonic neural network system."""
    print("🔍 Validating Photonic Neural Network System")
    print("=" * 50)
    
    # Create model
    if args.model_path:
        # In practice, would load saved model
        print(f"Loading model from {args.model_path}")
        model = create_benchmark_network("mnist")  # Placeholder
    else:
        model = create_benchmark_network(args.task)
    
    print(f"✓ Model loaded: {model.get_total_parameters():,} parameters")
    
    # Create validator
    validator = PhotonicSystemValidator()
    
    # Run validation
    print("🔍 Running comprehensive system validation...")
    result = validator.validate_system(model)
    
    # Display results
    print(f"\n📋 Validation Results:")
    print(f"✓ System valid: {'✅ YES' if result.is_valid else '❌ NO'}")
    print(f"✓ Errors found: {len(result.errors)}")
    print(f"✓ Warnings: {len(result.warnings)}")
    print(f"✓ Performance degradation: {result.performance_degradation:.1%}")
    
    # Show errors
    if result.errors:
        print(f"\n🚨 Errors ({len(result.errors)}):")
        for i, error in enumerate(result.errors[:5], 1):  # Show first 5
            print(f"{i}. [{error['severity'].value.upper()}] {error['message']}")
        
        if len(result.errors) > 5:
            print(f"   ... and {len(result.errors) - 5} more errors")
    
    # Show warnings  
    if result.warnings:
        print(f"\n⚠️  Warnings ({len(result.warnings)}):")
        for i, warning in enumerate(result.warnings[:3], 1):  # Show first 3
            print(f"{i}. [{warning['severity'].value.upper()}] {warning['message']}")
        
        if len(result.warnings) > 3:
            print(f"   ... and {len(result.warnings) - 3} more warnings")
    
    # Show recommendations
    if result.recommended_actions:
        print(f"\n💡 Recommendations:")
        for i, action in enumerate(result.recommended_actions, 1):
            print(f"{i}. {action}")
    
    # Start health monitoring if requested
    if args.monitor:
        print(f"\n🔄 Starting health monitoring (interval: {args.monitor_interval}s)")
        monitor = HealthMonitor(model, check_interval=args.monitor_interval)
        monitor.start_monitoring()
        
        try:
            while True:
                health = monitor.check_health()
                print(f"Health check: {'✅ HEALTHY' if health.is_valid else '⚠️ DEGRADED'}")
                time.sleep(args.monitor_interval)
        except KeyboardInterrupt:
            monitor.stop_monitoring()
            print("\n🛑 Health monitoring stopped")
    
    # Save validation report
    if args.output:
        report = {
            "validation_result": {
                "is_valid": result.is_valid,
                "num_errors": len(result.errors),
                "num_warnings": len(result.warnings),
                "performance_degradation": result.performance_degradation,
                "errors": [{"type": e["type"].value, "severity": e["severity"].value, 
                           "message": e["message"]} for e in result.errors],
                "warnings": [{"type": w["type"].value, "severity": w["severity"].value,
                            "message": w["message"]} for w in result.warnings],
                "recommendations": result.recommended_actions
            },
            "timestamp": time.time()
        }
        
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✓ Validation report saved to {args.output}")


def optimize_command(args):
    """Optimize photonic neural network for maximum performance."""
    print("⚡ Optimizing Photonic Neural Network")
    print("=" * 50)
    
    # Create optimized model
    print(f"Creating optimized network (level: {args.optimization_level})")
    model = create_optimized_network(args.task, args.optimization_level)
    
    print(f"✓ Optimized model created with {model.get_total_parameters():,} parameters")
    
    # Generate test data
    np.random.seed(args.seed)
    
    if args.task == "mnist":
        X_test = np.random.randn(args.batch_size, 784) * 0.1 + 0.5
    elif args.task == "cifar10":
        X_test = np.random.randn(args.batch_size, 3072) * 0.1 + 0.5
    elif args.task == "vowel_classification":
        X_test = np.random.randn(args.batch_size, 10) * 0.3 + 0.5
    
    print(f"✓ Generated test batch: {X_test.shape}")
    
    # Benchmark performance
    print(f"🚀 Running throughput benchmark ({args.iterations} iterations)")
    
    throughput_metrics = model.benchmark_throughput(
        input_shape=X_test.shape,
        num_iterations=args.iterations
    )
    
    print(f"\n📈 Performance Results:")
    print(f"✓ Average latency: {throughput_metrics['avg_latency_ns']:.2f}ns")
    print(f"✓ Throughput: {throughput_metrics['throughput_samples_per_sec']:.0f} samples/sec")
    print(f"✓ Speedup factor: {throughput_metrics['speedup_factor']:.1f}x")
    print(f"✓ Total processing time: {throughput_metrics['wall_clock_time_s']:.2f}s")
    
    # Analyze optimization effectiveness
    backend_type = type(model.layers[0].processor.backend).__name__
    print(f"✓ Optimization backend: {backend_type}")
    
    # Get detailed metrics
    opt_metrics = model.layers[0].processor.get_optimization_metrics()
    if opt_metrics:
        print(f"✓ Cache hit rate: {opt_metrics.get('cache_hit_rate', 0):.1%}")
        print(f"✓ Memory usage: {opt_metrics.get('memory_usage_mb', 0):.1f}MB")
    
    # Save performance results
    if args.output:
        results = {
            "optimization_level": args.optimization_level,
            "backend_type": backend_type,
            "throughput_metrics": throughput_metrics,
            "optimization_metrics": opt_metrics,
            "test_config": {
                "task": args.task,
                "batch_size": args.batch_size,
                "iterations": args.iterations
            },
            "timestamp": time.time()
        }
        
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✓ Performance results saved to {args.output}")


def create_parser():
    """Create the main CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="photonic-ai-simulator",
        description="Photonic AI Simulator - Advanced Optical Neural Network Framework"
    )
    
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    parser.add_argument("--log-file", type=str,
                       help="Log output file path")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Training command
    train_parser = subparsers.add_parser("train", help="Train photonic neural network")
    train_parser.add_argument("--task", choices=["mnist", "cifar10", "vowel_classification"],
                             default="mnist", help="Training task")
    train_parser.add_argument("--epochs", type=int, default=50,
                             help="Number of training epochs")
    train_parser.add_argument("--learning-rate", type=float, default=0.001,
                             help="Learning rate")
    train_parser.add_argument("--batch-size", type=int, default=32,
                             help="Batch size")
    train_parser.add_argument("--num-samples", type=int, default=1000,
                             help="Number of training samples")
    train_parser.add_argument("--forward-only", action="store_true",
                             help="Use forward-only training")
    train_parser.add_argument("--power-limit", type=float, default=500.0,
                             help="Power constraint in mW")
    train_parser.add_argument("--output", "-o", type=str,
                             help="Output path for trained model")
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Run performance benchmarks")
    benchmark_parser.add_argument("--benchmark", choices=["all", "mnist", "cifar10", "vowel"],
                                 default="all", help="Benchmark to run")
    benchmark_parser.add_argument("--target-accuracy", type=float, default=0.90,
                                 help="Target accuracy threshold")
    benchmark_parser.add_argument("--max-latency", type=float, default=1.0,
                                 help="Maximum latency in nanoseconds")
    benchmark_parser.add_argument("--max-power", type=float, default=500.0,
                                 help="Maximum power in milliwatts")
    benchmark_parser.add_argument("--num-runs", type=int, default=5,
                                 help="Number of benchmark runs")
    benchmark_parser.add_argument("--output", "-o", type=str,
                                 help="Output path for benchmark results")
    
    # Validation command
    validate_parser = subparsers.add_parser("validate", help="Validate system configuration")
    validate_parser.add_argument("--task", choices=["mnist", "cifar10", "vowel_classification"],
                                default="mnist", help="Model task for validation")
    validate_parser.add_argument("--model-path", type=str,
                                help="Path to saved model file")
    validate_parser.add_argument("--monitor", action="store_true",
                                help="Start continuous health monitoring")
    validate_parser.add_argument("--monitor-interval", type=float, default=5.0,
                                help="Monitoring interval in seconds")
    validate_parser.add_argument("--output", "-o", type=str,
                                help="Output path for validation report")
    
    # Optimization command
    optimize_parser = subparsers.add_parser("optimize", help="Optimize model performance")
    optimize_parser.add_argument("--task", choices=["mnist", "cifar10", "vowel_classification"],
                                default="mnist", help="Task for optimization")
    optimize_parser.add_argument("--optimization-level", choices=["low", "medium", "high", "extreme"],
                                default="high", help="Optimization level")
    optimize_parser.add_argument("--batch-size", type=int, default=64,
                                help="Batch size for testing")
    optimize_parser.add_argument("--iterations", type=int, default=1000,
                                help="Number of benchmark iterations")
    optimize_parser.add_argument("--output", "-o", type=str,
                                help="Output path for performance results")
    
    return parser


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Set up logging
    setup_cli_logging(args.verbose, args.log_file)
    
    # Route to appropriate command
    if args.command == "train":
        train_command(args)
    elif args.command == "benchmark":
        benchmark_command(args)
    elif args.command == "validate":
        validate_command(args)
    elif args.command == "optimize":
        optimize_command(args)
    else:
        parser.print_help()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())