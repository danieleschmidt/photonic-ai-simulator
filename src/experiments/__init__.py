"""
Experimental framework for photonic AI research.

Provides A/B testing capabilities, hypothesis-driven development tools,
and reproducible experimental methodology for photonic neural networks.
"""

from .ab_testing import ABTestFramework, ExperimentConfig, ExperimentResult
from .hypothesis_testing import HypothesisTest, StatisticalValidator
from .reproducibility import ReproducibilityFramework, ExperimentTracker

__all__ = [
    "ABTestFramework", 
    "ExperimentConfig",
    "ExperimentResult",
    "HypothesisTest",
    "StatisticalValidator", 
    "ReproducibilityFramework",
    "ExperimentTracker"
]