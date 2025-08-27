"""
Foundational algorithms for photonic computing and optical neural networks.

This module provides core algorithms specifically designed for photonic AI systems,
including MZI optimization, wavelength routing, and backpropagation-free training.
"""

from .mzi_optimization import (
    MZIOptimizer,
    MZIConfiguration,
    create_identity_target,
    create_hadamard_target,
    create_fourier_target
)

from .wavelength_routing import (
    WavelengthRouter,
    WavelengthChannel,
    OpticalPath,
    RoutingStrategy
)

from .optical_backprop_alternatives import (
    ForwardOnlyLearning,
    EquilibriumPropagation,
    DirectFeedbackAlignment,
    OpticalTrainingAlgorithm,
    create_optical_trainer,
    TrainingMetrics
)

__all__ = [
    # MZI Optimization
    'MZIOptimizer',
    'MZIConfiguration',
    'create_identity_target',
    'create_hadamard_target', 
    'create_fourier_target',
    
    # Wavelength Routing
    'WavelengthRouter',
    'WavelengthChannel',
    'OpticalPath',
    'RoutingStrategy',
    
    # Optical Training Algorithms
    'ForwardOnlyLearning',
    'EquilibriumPropagation',
    'DirectFeedbackAlignment',
    'OpticalTrainingAlgorithm',
    'create_optical_trainer',
    'TrainingMetrics'
]