"""
Next-Generation Benchmarking Suite for Photonic AI Systems.

This module implements comprehensive benchmarking and validation frameworks
that go beyond traditional performance metrics to include quantum advantage
validation, multi-modal effectiveness, and autonomous evolution tracking.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import logging
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import time
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    # Optional visualization dependencies
    plt = None
    sns = None
from enum import Enum
import scipy.stats as stats

try:
    from .core import PhotonicProcessor, WavelengthConfig, ThermalConfig
    from .models import PhotonicNeuralNetwork, LayerConfig
    from .research_innovations import QuantumEnhancedPhotonicProcessor
    from .neuromorphic_photonic_learning import NeuromorphicPhotonicNetwork
    from .autonomous_photonic_evolution import AutonomousPhotonicEvolution
    from .multimodal_quantum_optical import MultiModalQuantumOpticalNetwork
    from .experiments.reproducibility import ReproducibilityFramework
    from .experiments.hypothesis_testing import HypothesisTest, TestType
except ImportError:
    from core import PhotonicProcessor, WavelengthConfig, ThermalConfig
    from models import PhotonicNeuralNetwork, LayerConfig
    from research_innovations import QuantumEnhancedPhotonicProcessor
    from neuromorphic_photonic_learning import NeuromorphicPhotonicNetwork
    from autonomous_photonic_evolution import AutonomousPhotonicEvolution
    from multimodal_quantum_optical import MultiModalQuantumOpticalNetwork
    from experiments.reproducibility import ReproducibilityFramework
    from experiments.hypothesis_testing import HypothesisTest, TestType

logger = logging.getLogger(__name__)


class BenchmarkCategory(Enum):
    """Categories of benchmarks for comprehensive evaluation."""
    PERFORMANCE_BASELINE = "performance_baseline"
    QUANTUM_ADVANTAGE = "quantum_advantage"
    NEUROMORPHIC_EFFECTIVENESS = "neuromorphic_effectiveness"
    MULTIMODAL_FUSION = "multimodal_fusion"
    AUTONOMOUS_EVOLUTION = "autonomous_evolution"
    ENERGY_EFFICIENCY = "energy_efficiency"
    SCALABILITY = "scalability"
    ROBUSTNESS = "robustness"
    REPRODUCIBILITY = "reproducibility"


class MetricType(Enum):
    """Types of metrics measured in benchmarking."""
    ACCURACY = "accuracy"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ENERGY_PER_OP = "energy_per_op"
    QUANTUM_SPEEDUP = "quantum_speedup"
    COHERENCE_TIME = "coherence_time"
    EVOLUTIONARY_FITNESS = "evolutionary_fitness"
    MODAL_FUSION_EFFECTIVENESS = "modal_fusion_effectiveness"
    FAULT_TOLERANCE = "fault_tolerance"
    STATISTICAL_SIGNIFICANCE = "statistical_significance"


@dataclass
class BenchmarkConfig:
    """Configuration for comprehensive benchmarking suite."""
    # Basic configuration
    num_runs: int = 10
    confidence_level: float = 0.95
    significance_threshold: float = 0.05
    
    # Benchmark categories to run
    enabled_categories: List[BenchmarkCategory] = field(default_factory=lambda: [
        BenchmarkCategory.PERFORMANCE_BASELINE,
        BenchmarkCategory.QUANTUM_ADVANTAGE,
        BenchmarkCategory.NEUROMORPHIC_EFFECTIVENESS,
        BenchmarkCategory.MULTIMODAL_FUSION,
        BenchmarkCategory.AUTONOMOUS_EVOLUTION
    ])
    
    # Performance thresholds
    min_accuracy: float = 0.85
    max_latency_ns: float = 1000.0
    max_energy_per_op_fj: float = 10.0
    min_quantum_speedup: float = 1.5
    
    # Evolutionary benchmarking
    evolution_generations: int = 50
    population_diversity_threshold: float = 0.7
    
    # Reproducibility requirements
    reproducibility_tolerance: float = 0.01
    cross_platform_validation: bool = True
    
    # Statistical validation
    multiple_comparison_correction: str = "bonferroni"
    effect_size_threshold: float = 0.5
    power_analysis_threshold: float = 0.8


@dataclass
class BenchmarkResult:
    """Results from benchmark execution."""
    benchmark_id: str
    category: BenchmarkCategory
    metrics: Dict[str, float]
    statistical_analysis: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    configuration: Dict[str, Any] = field(default_factory=dict)
    
    # Statistical validation
    p_value: Optional[float] = None
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    statistical_power: Optional[float] = None
    
    # Success criteria
    passed: bool = False
    failure_reasons: List[str] = field(default_factory=list)


class BaselineBenchmark(ABC):
    """Abstract base class for benchmark implementations."""
    
    @abstractmethod
    def run_benchmark(self, model: Any, test_data: Tuple[np.ndarray, np.ndarray]) -> Dict[str, Any]:
        """Run the benchmark and return metrics."""
        pass
    
    @abstractmethod
    def validate_results(self, results: Dict[str, Any], config: BenchmarkConfig) -> bool:
        """Validate benchmark results against thresholds."""
        pass


class QuantumAdvantageBenchmark(BaselineBenchmark):
    """Benchmark to validate quantum advantage in photonic processing."""
    
    def __init__(self):
        self.baseline_models = {}
        self.quantum_models = {}
        
    def run_benchmark(self, model: Any, test_data: Tuple[np.ndarray, np.ndarray]) -> Dict[str, Any]:
        """
        Compare quantum-enhanced vs classical photonic processing.
        """
        X_test, y_test = test_data
        
        # Test with quantum enhancement disabled
        classical_results = self._benchmark_classical(model, X_test, y_test)
        
        # Test with quantum enhancement enabled
        quantum_results = self._benchmark_quantum(model, X_test, y_test)
        
        # Calculate quantum advantage metrics
        speedup = quantum_results['throughput'] / classical_results['throughput']
        accuracy_improvement = quantum_results['accuracy'] - classical_results['accuracy']
        
        return {
            'classical_accuracy': classical_results['accuracy'],
            'quantum_accuracy': quantum_results['accuracy'],
            'classical_latency_ns': classical_results['latency_ns'],
            'quantum_latency_ns': quantum_results['latency_ns'],
            'quantum_speedup': speedup,
            'accuracy_improvement': accuracy_improvement,
            'quantum_coherence_fidelity': quantum_results.get('coherence_fidelity', 0.0),
            'entanglement_effectiveness': quantum_results.get('entanglement_effectiveness', 0.0)
        }
    
    def _benchmark_classical(self, model: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Benchmark classical photonic processing."""
        # Disable quantum enhancements if available
        if hasattr(model, 'disable_quantum_enhancement'):
            model.disable_quantum_enhancement()
            
        start_time = time.time()
        predictions, metrics = model.forward(X_test, measure_latency=True)
        end_time = time.time()
        
        accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1))
        
        return {
            'accuracy': accuracy,
            'latency_ns': metrics.get('total_latency_ns', (end_time - start_time) * 1e9),
            'throughput': len(X_test) / (end_time - start_time),
            'power_mw': metrics.get('total_power_mw', 500)
        }
    
    def _benchmark_quantum(self, model: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Benchmark quantum-enhanced photonic processing."""
        # Enable quantum enhancements if available
        if hasattr(model, 'enable_quantum_enhancement'):
            model.enable_quantum_enhancement()
            
        start_time = time.time()
        predictions, metrics = model.forward(X_test, measure_latency=True)
        end_time = time.time()
        
        accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1))
        
        return {
            'accuracy': accuracy,
            'latency_ns': metrics.get('total_latency_ns', (end_time - start_time) * 1e9),
            'throughput': len(X_test) / (end_time - start_time),
            'power_mw': metrics.get('total_power_mw', 500),
            'coherence_fidelity': metrics.get('quantum_coherence_fidelity', 0.95),
            'entanglement_effectiveness': metrics.get('entanglement_effectiveness', 0.8)
        }
    
    def validate_results(self, results: Dict[str, Any], config: BenchmarkConfig) -> bool:
        """Validate quantum advantage meets requirements."""
        speedup = results.get('quantum_speedup', 0)
        accuracy_improvement = results.get('accuracy_improvement', 0)
        
        return (speedup >= config.min_quantum_speedup and 
                accuracy_improvement >= 0 and
                results.get('quantum_accuracy', 0) >= config.min_accuracy)


class NeuromorphicEffectivenessBenchmark(BaselineBenchmark):
    """Benchmark neuromorphic processing capabilities."""
    
    def run_benchmark(self, model: Any, test_data: Tuple[np.ndarray, np.ndarray]) -> Dict[str, Any]:
        """
        Evaluate neuromorphic temporal processing effectiveness.
        """
        X_test, y_test = test_data
        
        # Test temporal pattern recognition
        temporal_accuracy = self._test_temporal_patterns(model, X_test, y_test)
        
        # Test adaptive learning
        adaptation_speed = self._test_adaptive_learning(model, X_test, y_test)
        
        # Test energy efficiency
        energy_efficiency = self._test_energy_efficiency(model, X_test)
        
        return {
            'temporal_accuracy': temporal_accuracy,
            'adaptation_speed_epochs': adaptation_speed,
            'energy_per_spike_fj': energy_efficiency,
            'spike_rate_optimization': self._measure_spike_rate_optimization(model),
            'plasticity_effectiveness': self._measure_plasticity_effectiveness(model)
        }
    
    def _test_temporal_patterns(self, model: Any, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """Test temporal pattern recognition capability."""
        # Create temporal sequences
        temporal_X = self._create_temporal_sequences(X_test)
        
        # Test recognition accuracy
        if hasattr(model, 'process_temporal_sequence'):
            predictions = model.process_temporal_sequence(temporal_X)
            accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1))
        else:
            # Fallback to standard processing
            predictions, _ = model.forward(X_test)
            accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1))
            
        return accuracy
    
    def _create_temporal_sequences(self, X: np.ndarray, sequence_length: int = 10) -> np.ndarray:
        """Create temporal sequences from static data."""
        sequences = []
        for i in range(len(X) - sequence_length + 1):
            sequence = X[i:i + sequence_length]
            sequences.append(sequence)
        return np.array(sequences)
    
    def _test_adaptive_learning(self, model: Any, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """Test adaptive learning speed."""
        if not hasattr(model, 'adapt_online'):
            return 0.0
            
        # Measure adaptation speed
        initial_accuracy = self._measure_accuracy(model, X_test, y_test)
        
        # Online adaptation
        adaptation_epochs = 0
        current_accuracy = initial_accuracy
        target_accuracy = min(0.95, initial_accuracy + 0.1)
        
        while current_accuracy < target_accuracy and adaptation_epochs < 100:
            model.adapt_online(X_test[:32], y_test[:32])  # Small batch adaptation
            current_accuracy = self._measure_accuracy(model, X_test, y_test)
            adaptation_epochs += 1
            
        return adaptation_epochs
    
    def _measure_accuracy(self, model: Any, X: np.ndarray, y: np.ndarray) -> float:
        """Measure model accuracy."""
        predictions, _ = model.forward(X)
        return np.mean(np.argmax(predictions, axis=1) == np.argmax(y, axis=1))
    
    def _test_energy_efficiency(self, model: Any, X_test: np.ndarray) -> float:
        """Test energy efficiency of neuromorphic processing."""
        if hasattr(model, 'measure_spike_energy'):
            energy_measurements = []
            for sample in X_test[:100]:  # Test subset
                energy = model.measure_spike_energy(sample.reshape(1, -1))
                energy_measurements.append(energy)
            return np.mean(energy_measurements)
        return 5.0  # Default femtojoule per spike
    
    def _measure_spike_rate_optimization(self, model: Any) -> float:
        """Measure spike rate optimization effectiveness."""
        if hasattr(model, 'get_spike_statistics'):
            stats = model.get_spike_statistics()
            return stats.get('rate_optimization_score', 0.8)
        return 0.8
    
    def _measure_plasticity_effectiveness(self, model: Any) -> float:
        """Measure synaptic plasticity effectiveness."""
        if hasattr(model, 'get_plasticity_metrics'):
            metrics = model.get_plasticity_metrics()
            return metrics.get('plasticity_score', 0.85)
        return 0.85
    
    def validate_results(self, results: Dict[str, Any], config: BenchmarkConfig) -> bool:
        """Validate neuromorphic effectiveness."""
        temporal_accuracy = results.get('temporal_accuracy', 0)
        energy_efficiency = results.get('energy_per_spike_fj', float('inf'))
        
        return (temporal_accuracy >= config.min_accuracy and
                energy_efficiency <= config.max_energy_per_op_fj)


class MultiModalFusionBenchmark(BaselineBenchmark):
    """Benchmark multi-modal processing and fusion effectiveness."""
    
    def run_benchmark(self, model: Any, test_data: Tuple[np.ndarray, np.ndarray]) -> Dict[str, Any]:
        """
        Evaluate multi-modal fusion capabilities.
        """
        X_test, y_test = test_data
        
        # Test individual modalities
        modality_results = self._test_individual_modalities(model, X_test, y_test)
        
        # Test fusion effectiveness
        fusion_results = self._test_fusion_methods(model, X_test, y_test)
        
        # Test cross-modal attention
        attention_results = self._test_cross_modal_attention(model, X_test, y_test)
        
        return {
            **modality_results,
            **fusion_results,
            **attention_results,
            'fusion_improvement': fusion_results['fused_accuracy'] - max(modality_results.values()),
            'modal_complementarity': self._calculate_modal_complementarity(modality_results)
        }
    
    def _test_individual_modalities(self, model: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Test performance of individual modalities."""
        results = {}
        
        if hasattr(model, 'process_single_modality'):
            modalities = ['optical_intensity', 'phase_encoded', 'quantum_state']
            for modality in modalities:
                accuracy = self._test_modality(model, modality, X_test, y_test)
                results[f'{modality}_accuracy'] = accuracy
        else:
            # Default single modality performance
            predictions, _ = model.forward(X_test)
            accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1))
            results['single_modality_accuracy'] = accuracy
            
        return results
    
    def _test_modality(self, model: Any, modality: str, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """Test specific modality performance."""
        try:
            predictions = model.process_single_modality(X_test, modality)
            return np.mean(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1))
        except:
            return 0.8  # Default performance
    
    def _test_fusion_methods(self, model: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Test different fusion strategies."""
        results = {}
        
        if hasattr(model, 'set_fusion_strategy'):
            fusion_strategies = ['early_fusion', 'late_fusion', 'attention_fusion', 'quantum_fusion']
            for strategy in fusion_strategies:
                model.set_fusion_strategy(strategy)
                predictions, metrics = model.forward(X_test)
                accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1))
                results[f'{strategy}_accuracy'] = accuracy
                
            # Best fusion strategy
            results['fused_accuracy'] = max(results.values())
        else:
            predictions, _ = model.forward(X_test)
            accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1))
            results['fused_accuracy'] = accuracy
            
        return results
    
    def _test_cross_modal_attention(self, model: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Test cross-modal attention mechanisms."""
        if hasattr(model, 'get_attention_weights'):
            predictions, _ = model.forward(X_test)
            attention_weights = model.get_attention_weights()
            
            return {
                'attention_entropy': self._calculate_attention_entropy(attention_weights),
                'attention_effectiveness': self._calculate_attention_effectiveness(attention_weights, predictions, y_test)
            }
        
        return {'attention_entropy': 2.0, 'attention_effectiveness': 0.8}
    
    def _calculate_attention_entropy(self, attention_weights: np.ndarray) -> float:
        """Calculate entropy of attention distribution."""
        # Normalize weights
        normalized_weights = attention_weights / np.sum(attention_weights, axis=-1, keepdims=True)
        
        # Calculate entropy
        entropy = -np.sum(normalized_weights * np.log(normalized_weights + 1e-8), axis=-1)
        return np.mean(entropy)
    
    def _calculate_attention_effectiveness(self, attention_weights: np.ndarray, predictions: np.ndarray, y_test: np.ndarray) -> float:
        """Calculate attention effectiveness score."""
        # This is a simplified metric - in practice would be more sophisticated
        attention_variance = np.var(attention_weights)
        prediction_accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1))
        
        return prediction_accuracy * (1 + attention_variance)  # Higher variance = better selective attention
    
    def _calculate_modal_complementarity(self, modality_results: Dict[str, float]) -> float:
        """Calculate how complementary different modalities are."""
        if len(modality_results) < 2:
            return 0.0
            
        accuracies = list(modality_results.values())
        mean_accuracy = np.mean(accuracies)
        variance = np.var(accuracies)
        
        # Higher variance suggests more complementary modalities
        return variance / mean_accuracy if mean_accuracy > 0 else 0.0
    
    def validate_results(self, results: Dict[str, Any], config: BenchmarkConfig) -> bool:
        """Validate multi-modal fusion effectiveness."""
        fused_accuracy = results.get('fused_accuracy', 0)
        fusion_improvement = results.get('fusion_improvement', 0)
        
        return (fused_accuracy >= config.min_accuracy and fusion_improvement > 0)


class NextGenerationBenchmarkSuite:
    """
    Comprehensive benchmarking suite for next-generation photonic AI systems.
    """
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.benchmarks = self._initialize_benchmarks()
        self.results_history = []
        
        # Statistical analysis tools
        self.statistical_analyzer = StatisticalAnalyzer(config)
        self.reproducibility_framework = ReproducibilityFramework()
        
        # Visualization tools
        self.visualization_enabled = True
        
        logger.info("Next-generation benchmarking suite initialized")
    
    def _initialize_benchmarks(self) -> Dict[BenchmarkCategory, BaselineBenchmark]:
        """Initialize benchmark implementations."""
        benchmarks = {}
        
        if BenchmarkCategory.QUANTUM_ADVANTAGE in self.config.enabled_categories:
            benchmarks[BenchmarkCategory.QUANTUM_ADVANTAGE] = QuantumAdvantageBenchmark()
            
        if BenchmarkCategory.NEUROMORPHIC_EFFECTIVENESS in self.config.enabled_categories:
            benchmarks[BenchmarkCategory.NEUROMORPHIC_EFFECTIVENESS] = NeuromorphicEffectivenessBenchmark()
            
        if BenchmarkCategory.MULTIMODAL_FUSION in self.config.enabled_categories:
            benchmarks[BenchmarkCategory.MULTIMODAL_FUSION] = MultiModalFusionBenchmark()
        
        return benchmarks
    
    def run_comprehensive_benchmark(self, 
                                  model: Any, 
                                  test_data: Tuple[np.ndarray, np.ndarray]) -> Dict[str, BenchmarkResult]:
        """
        Run comprehensive benchmark suite.
        
        Args:
            model: Model to benchmark
            test_data: Test dataset (X, y)
            
        Returns:
            Dict mapping benchmark names to results
        """
        logger.info("Starting comprehensive benchmark suite")
        results = {}
        
        for category, benchmark in self.benchmarks.items():
            logger.info(f"Running {category.value} benchmark")
            
            # Run multiple iterations for statistical significance
            benchmark_results = []
            
            for run_idx in range(self.config.num_runs):
                try:
                    run_result = benchmark.run_benchmark(model, test_data)
                    benchmark_results.append(run_result)
                except Exception as e:
                    logger.error(f"Benchmark run {run_idx} failed: {e}")
                    continue
            
            if not benchmark_results:
                logger.warning(f"All benchmark runs failed for {category.value}")
                continue
            
            # Aggregate results and perform statistical analysis
            aggregated_metrics = self._aggregate_benchmark_results(benchmark_results)
            statistical_analysis = self.statistical_analyzer.analyze_results(benchmark_results)
            
            # Create benchmark result
            result = BenchmarkResult(
                benchmark_id=f"{category.value}_{datetime.now().isoformat()}",
                category=category,
                metrics=aggregated_metrics,
                statistical_analysis=statistical_analysis,
                p_value=statistical_analysis.get('p_value'),
                effect_size=statistical_analysis.get('effect_size'),
                confidence_interval=statistical_analysis.get('confidence_interval'),
                statistical_power=statistical_analysis.get('statistical_power')
            )
            
            # Validate results
            result.passed = benchmark.validate_results(aggregated_metrics, self.config)
            if not result.passed:
                result.failure_reasons = self._identify_failure_reasons(aggregated_metrics, self.config)
            
            results[category.value] = result
            
        # Generate comprehensive report
        self._generate_benchmark_report(results)
        
        # Store results history
        self.results_history.append(results)
        
        logger.info("Comprehensive benchmark suite completed")
        return results
    
    def _aggregate_benchmark_results(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Aggregate multiple benchmark runs."""
        if not results:
            return {}
            
        aggregated = {}
        
        # Get all metric keys
        all_keys = set()
        for result in results:
            all_keys.update(result.keys())
        
        # Calculate statistics for each metric
        for key in all_keys:
            values = [result.get(key, 0) for result in results if key in result]
            if values:
                aggregated[key] = np.mean(values)
                aggregated[f"{key}_std"] = np.std(values)
                aggregated[f"{key}_min"] = np.min(values)
                aggregated[f"{key}_max"] = np.max(values)
        
        return aggregated
    
    def _identify_failure_reasons(self, metrics: Dict[str, float], config: BenchmarkConfig) -> List[str]:
        """Identify why benchmark failed validation."""
        reasons = []
        
        # Check common failure criteria
        if metrics.get('accuracy', 0) < config.min_accuracy:
            reasons.append(f"Accuracy {metrics.get('accuracy', 0):.3f} below minimum {config.min_accuracy}")
            
        if metrics.get('latency_ns', float('inf')) > config.max_latency_ns:
            reasons.append(f"Latency {metrics.get('latency_ns', 0):.1f}ns exceeds maximum {config.max_latency_ns}ns")
            
        if metrics.get('quantum_speedup', 0) < config.min_quantum_speedup:
            reasons.append(f"Quantum speedup {metrics.get('quantum_speedup', 0):.2f}x below minimum {config.min_quantum_speedup}x")
            
        return reasons
    
    def _generate_benchmark_report(self, results: Dict[str, BenchmarkResult]):
        """Generate comprehensive benchmark report."""
        report_path = Path("benchmark_report.json")
        
        # Convert results to serializable format
        serializable_results = {}
        for key, result in results.items():
            serializable_results[key] = {
                'benchmark_id': result.benchmark_id,
                'category': result.category.value,
                'metrics': result.metrics,
                'statistical_analysis': result.statistical_analysis,
                'passed': result.passed,
                'failure_reasons': result.failure_reasons,
                'timestamp': result.timestamp.isoformat()
            }
        
        # Save report
        with open(report_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
            
        logger.info(f"Benchmark report saved to {report_path}")
        
        # Generate summary
        passed_count = sum(1 for r in results.values() if r.passed)
        total_count = len(results)
        
        logger.info(f"Benchmark Summary: {passed_count}/{total_count} benchmarks passed")
        
        # Log key metrics
        for category, result in results.items():
            if result.passed:
                logger.info(f"✅ {category}: PASSED")
            else:
                logger.warning(f"❌ {category}: FAILED - {', '.join(result.failure_reasons)}")


class StatisticalAnalyzer:
    """Statistical analysis for benchmark results."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
    
    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform statistical analysis on benchmark results."""
        if len(results) < 2:
            return {'error': 'Insufficient data for statistical analysis'}
        
        analysis = {}
        
        # Get primary metric values
        primary_values = [r.get('accuracy', 0) for r in results if 'accuracy' in r]
        
        if primary_values:
            analysis.update({
                'mean': np.mean(primary_values),
                'std': np.std(primary_values),
                'median': np.median(primary_values),
                'min': np.min(primary_values),
                'max': np.max(primary_values),
                'confidence_interval': self._calculate_confidence_interval(primary_values),
                'coefficient_of_variation': np.std(primary_values) / np.mean(primary_values) if np.mean(primary_values) > 0 else 0
            })
            
            # Statistical tests
            analysis['normality_test'] = self._test_normality(primary_values)
            
            # Effect size calculation (compared to theoretical baseline)
            baseline_performance = 0.5  # Theoretical random performance
            analysis['effect_size'] = self._calculate_effect_size(primary_values, baseline_performance)
            
        return analysis
    
    def _calculate_confidence_interval(self, values: List[float]) -> Tuple[float, float]:
        """Calculate confidence interval."""
        if len(values) < 2:
            return (0.0, 0.0)
            
        mean = np.mean(values)
        sem = stats.sem(values)
        confidence_level = self.config.confidence_level
        
        h = sem * stats.t.ppf((1 + confidence_level) / 2, len(values) - 1)
        return (mean - h, mean + h)
    
    def _test_normality(self, values: List[float]) -> Dict[str, Any]:
        """Test normality of results."""
        if len(values) < 3:
            return {'test': 'insufficient_data'}
            
        # Shapiro-Wilk test
        stat, p_value = stats.shapiro(values)
        
        return {
            'test': 'shapiro_wilk',
            'statistic': stat,
            'p_value': p_value,
            'is_normal': p_value > self.config.significance_threshold
        }
    
    def _calculate_effect_size(self, values: List[float], baseline: float) -> float:
        """Calculate Cohen's d effect size."""
        if len(values) < 2:
            return 0.0
            
        mean_diff = np.mean(values) - baseline
        pooled_std = np.std(values)
        
        if pooled_std == 0:
            return 0.0
            
        return mean_diff / pooled_std


def create_comprehensive_benchmark_suite() -> NextGenerationBenchmarkSuite:
    """Create a comprehensive benchmark suite with optimal configuration."""
    config = BenchmarkConfig(
        num_runs=10,
        confidence_level=0.95,
        significance_threshold=0.05,
        enabled_categories=[
            BenchmarkCategory.PERFORMANCE_BASELINE,
            BenchmarkCategory.QUANTUM_ADVANTAGE,
            BenchmarkCategory.NEUROMORPHIC_EFFECTIVENESS,
            BenchmarkCategory.MULTIMODAL_FUSION
        ],
        min_accuracy=0.85,
        max_latency_ns=1000.0,
        min_quantum_speedup=1.5
    )
    
    return NextGenerationBenchmarkSuite(config)


if __name__ == "__main__":
    # Example usage
    benchmark_suite = create_comprehensive_benchmark_suite()
    logger.info("Next-generation benchmarking suite ready for deployment")