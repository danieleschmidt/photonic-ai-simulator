"""
Photonic AI Simulator: Optical computing simulation framework for 100x ML acceleration.

This package provides state-of-the-art photonic neural network simulation capabilities
based on recent breakthroughs in integrated photonic processors.

Key Features:
- Sub-nanosecond inference latency simulation
- Thermal drift modeling and compensation
- Hardware-aware training algorithms
- CMOS-compatible fabrication parameter modeling
"""

from .core import PhotonicProcessor, WavelengthConfig
from .models import PhotonicNeuralNetwork, MZILayer
from .training import ForwardOnlyTrainer, HardwareAwareOptimizer
from .benchmarks import MNISTBenchmark, CIFAR10Benchmark, VowelClassificationBenchmark

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"

__all__ = [
    "PhotonicProcessor",
    "WavelengthConfig", 
    "PhotonicNeuralNetwork",
    "MZILayer",
    "ForwardOnlyTrainer",
    "HardwareAwareOptimizer",
    "MNISTBenchmark",
    "CIFAR10Benchmark", 
    "VowelClassificationBenchmark",
]