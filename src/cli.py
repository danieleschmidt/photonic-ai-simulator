"""
Command-line interface for photonic AI simulator.

Provides benchmarking, training, and validation commands for 
photonic neural networks with performance optimization.
"""

import argparse
import sys
import json
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Optional

try:
    from .models import create_benchmark_network, PhotonicNeuralNetwork
    from .training import create_training_pipeline, TrainingConfig
    from .benchmarks import run_comprehensive_benchmarks
    from .validation import validate_performance_targets
except ImportError:
    from models import create_benchmark_network, PhotonicNeuralNetwork
    from training import create_training_pipeline, TrainingConfig
    from benchmarks import run_comprehensive_benchmarks
    from validation import validate_performance_targets


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def benchmark_command():
    """CLI command for running benchmarks."""
    parser = argparse.ArgumentParser(
        description="Run photonic AI simulator benchmarks"
    )
    parser.add_argument(
        "--task", 
        choices=["mnist", "cifar10", "vowel_classification", "all"],
        default="all",
        help="Benchmark task to run"
    )
    parser.add_argument(
        "--samples", 
        type=int, 
        default=1000,
        help="Number of samples for benchmarking"
    )
    parser.add_argument(
        "--output", 
        type=str,
        default="benchmark_results.json",
        help="Output file for results"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting benchmarks for task: {args.task}")
    
    # Run benchmarks
    if args.task == "all":
        tasks = ["mnist", "cifar10", "vowel_classification"]
    else:
        tasks = [args.task]
    
    all_results = {}
    
    for task in tasks:
        logger.info(f"Running benchmark for {task}")
        
        # Create benchmark network
        model = create_benchmark_network(task)
        
        # Generate sample data
        if task == "mnist":
            X = np.random.randn(args.samples, 784)
            y = np.eye(10)[np.random.randint(0, 10, args.samples)]
        elif task == "cifar10":
            X = np.random.randn(args.samples, 3072)
            y = np.eye(10)[np.random.randint(0, 10, args.samples)]
        elif task == "vowel_classification":
            X = np.random.randn(args.samples, 10)
            y = np.eye(6)[np.random.randint(0, 6, args.samples)]
        
        # Run comprehensive benchmarks
        results = run_comprehensive_benchmarks(model, X, y)
        all_results[task] = results
        
        logger.info(f"Completed {task} benchmark:")
        logger.info(f"  Average latency: {results['latency_stats']['mean']:.2f}ns")
        logger.info(f"  Average power: {results['power_stats']['mean']:.2f}mW")
        logger.info(f"  Accuracy: {results['accuracy']:.3f}")
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    logger.info(f"Results saved to {args.output}")
    return all_results


def train_command():
    """CLI command for training models."""
    parser = argparse.ArgumentParser(
        description="Train photonic neural networks"
    )
    parser.add_argument(
        "--task",
        choices=["mnist", "cifar10", "vowel_classification"],
        required=True,
        help="Training task"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=32,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning-rate", 
        type=float, 
        default=0.001,
        help="Learning rate"
    )
    parser.add_argument(
        "--samples", 
        type=int, 
        default=1000,
        help="Number of training samples"
    )
    parser.add_argument(
        "--output-model", 
        type=str,
        default="trained_model.npy",
        help="Output model file"
    )
    parser.add_argument(
        "--output-history", 
        type=str,
        default="training_history.json",
        help="Output training history file"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting training for task: {args.task}")
    
    # Create model
    model = create_benchmark_network(args.task)
    
    # Generate sample data
    if args.task == "mnist":
        X = np.random.randn(args.samples, 784)
        y = np.eye(10)[np.random.randint(0, 10, args.samples)]
    elif args.task == "cifar10":
        X = np.random.randn(args.samples, 3072)
        y = np.eye(10)[np.random.randint(0, 10, args.samples)]
    elif args.task == "vowel_classification":
        X = np.random.randn(args.samples, 10)
        y = np.eye(6)[np.random.randint(0, 6, args.samples)]
    
    # Create trainer
    trainer = create_training_pipeline(
        model,
        training_type="forward_only",
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs
    )
    
    # Train model
    logger.info("Starting training...")
    history = trainer.train(X, y)
    
    # Save model and history
    model.save_model(args.output_model)
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_history = {}
    for key, values in history.items():
        if isinstance(values, list) and len(values) > 0:
            if isinstance(values[0], np.ndarray):
                serializable_history[key] = [v.tolist() for v in values]
            else:
                serializable_history[key] = values
        else:
            serializable_history[key] = values
    
    with open(args.output_history, 'w') as f:
        json.dump(serializable_history, f, indent=2, default=str)
    
    logger.info(f"Training completed!")
    logger.info(f"Final training accuracy: {history['train_accuracy'][-1]:.3f}")
    logger.info(f"Final validation accuracy: {history['val_accuracy'][-1]:.3f}")
    logger.info(f"Model saved to {args.output_model}")
    logger.info(f"History saved to {args.output_history}")
    
    return history


def validate_command():
    """CLI command for validating models."""
    parser = argparse.ArgumentParser(
        description="Validate photonic neural network performance"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Path to model file (.npy)"
    )
    parser.add_argument(
        "--task",
        choices=["mnist", "cifar10", "vowel_classification"],
        help="Validation task (required if no model specified)"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Number of validation samples"
    )
    parser.add_argument(
        "--target-latency",
        type=float,
        default=1.0,
        help="Target latency in nanoseconds"
    )
    parser.add_argument(
        "--target-accuracy",
        type=float,
        default=0.9,
        help="Target accuracy"
    )
    parser.add_argument(
        "--target-power",
        type=float,
        default=500.0,
        help="Target power consumption in mW"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="validation_results.json",
        help="Output file for validation results"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    logger = logging.getLogger(__name__)
    
    # Load or create model
    if args.model:
        if not args.task:
            logger.error("Task must be specified when loading a model")
            sys.exit(1)
        
        model = create_benchmark_network(args.task)
        model.load_model(args.model)
        logger.info(f"Loaded model from {args.model}")
    else:
        if not args.task:
            logger.error("Either model or task must be specified")
            sys.exit(1)
        
        model = create_benchmark_network(args.task)
        logger.info(f"Created new {args.task} model for validation")
    
    # Generate sample data
    if args.task == "mnist":
        X = np.random.randn(args.samples, 784)
        y = np.eye(10)[np.random.randint(0, 10, args.samples)]
    elif args.task == "cifar10":
        X = np.random.randn(args.samples, 3072)
        y = np.eye(10)[np.random.randint(0, 10, args.samples)]
    elif args.task == "vowel_classification":
        X = np.random.randn(args.samples, 10)
        y = np.eye(6)[np.random.randint(0, 6, args.samples)]
    
    # Define performance targets
    targets = {
        "latency_ns": args.target_latency,
        "accuracy": args.target_accuracy,
        "power_mw": args.target_power
    }
    
    logger.info("Starting validation...")
    logger.info(f"Targets: latency={args.target_latency}ns, "
               f"accuracy={args.target_accuracy}, power={args.target_power}mW")
    
    # Run validation
    results = validate_performance_targets(model, X, y, targets)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Print summary
    logger.info("Validation Results:")
    logger.info(f"  Latency: {results['measured']['latency_ns']:.2f}ns "
               f"(target: {args.target_latency}ns) - "
               f"{'PASS' if results['pass']['latency'] else 'FAIL'}")
    logger.info(f"  Accuracy: {results['measured']['accuracy']:.3f} "
               f"(target: {args.target_accuracy}) - "
               f"{'PASS' if results['pass']['accuracy'] else 'FAIL'}")
    logger.info(f"  Power: {results['measured']['power_mw']:.2f}mW "
               f"(target: {args.target_power}mW) - "
               f"{'PASS' if results['pass']['power'] else 'FAIL'}")
    logger.info(f"Overall: {'PASS' if results['overall_pass'] else 'FAIL'}")
    logger.info(f"Results saved to {args.output}")
    
    return results


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Photonic AI Simulator CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all benchmarks
  photonic-benchmark --task all --samples 1000
  
  # Train vowel classification model
  photonic-train --task vowel_classification --epochs 50
  
  # Validate trained model
  photonic-validate --model trained_model.npy --task mnist
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Benchmark subcommand
    bench_parser = subparsers.add_parser('benchmark', help='Run benchmarks')
    bench_parser.set_defaults(func=benchmark_command)
    
    # Train subcommand
    train_parser = subparsers.add_parser('train', help='Train models')
    train_parser.set_defaults(func=train_command)
    
    # Validate subcommand
    validate_parser = subparsers.add_parser('validate', help='Validate models')
    validate_parser.set_defaults(func=validate_command)
    
    args = parser.parse_args()
    
    if hasattr(args, 'func'):
        return args.func()
    else:
        parser.print_help()
        return None


if __name__ == "__main__":
    main()