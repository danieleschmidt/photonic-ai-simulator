"""
Test suite for photonic AI simulator.

Comprehensive testing framework ensuring code quality,
performance validation, and hardware simulation accuracy.
"""

from .test_core import TestPhotonicProcessor, TestWavelengthConfig
from .test_models import TestPhotonicNeuralNetwork, TestMZILayer
from .test_training import TestForwardOnlyTrainer
from .test_benchmarks import TestBenchmarkSuite
from .test_optimization import TestOptimization
from .integration_tests import IntegrationTestSuite

__all__ = [
    "TestPhotonicProcessor",
    "TestWavelengthConfig", 
    "TestPhotonicNeuralNetwork",
    "TestMZILayer",
    "TestForwardOnlyTrainer",
    "TestBenchmarkSuite",
    "TestOptimization",
    "IntegrationTestSuite"
]