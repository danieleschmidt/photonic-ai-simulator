"""
A/B testing framework for photonic neural network experiments.

Implements controlled experiments with statistical significance testing
for evaluating novel photonic computing approaches.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
import logging
from enum import Enum
import time
from concurrent.futures import ThreadPoolExecutor
import json

from ..models import PhotonicNeuralNetwork
from ..benchmarks import BenchmarkResult


logger = logging.getLogger(__name__)


class ExperimentType(Enum):
    """Types of A/B experiments."""
    ARCHITECTURE_COMPARISON = "architecture"
    TRAINING_METHOD = "training"
    HARDWARE_CONFIG = "hardware"
    OPTIMIZATION = "optimization"


@dataclass
class ExperimentConfig:
    """Configuration for A/B testing experiments."""
    name: str
    experiment_type: ExperimentType
    description: str
    
    # Statistical parameters
    confidence_level: float = 0.95
    statistical_power: float = 0.8
    min_effect_size: float = 0.05
    
    # Experiment design
    num_runs_per_variant: int = 10
    randomize_order: bool = True
    parallel_execution: bool = True
    
    # Success metrics
    primary_metric: str = "accuracy"
    secondary_metrics: List[str] = field(default_factory=lambda: ["latency_ns", "power_mw"])
    
    # Validation
    cross_validation_folds: int = 5
    bootstrap_samples: int = 1000


@dataclass 
class ExperimentResult:
    """Results from an A/B experiment."""
    config: ExperimentConfig
    variant_a_results: List[Dict[str, float]]
    variant_b_results: List[Dict[str, float]]
    
    # Statistical analysis
    p_value: float
    confidence_interval: Tuple[float, float]
    effect_size: float
    statistical_power: float
    
    # Summary statistics
    variant_a_mean: Dict[str, float]
    variant_b_mean: Dict[str, float]
    variant_a_std: Dict[str, float]
    variant_b_std: Dict[str, float]
    
    # Conclusions
    significant_difference: bool
    winner: Optional[str]
    recommendation: str


class ABTestFramework:
    """
    A/B testing framework for photonic neural networks.
    
    Implements rigorous statistical testing methodology to evaluate
    novel approaches against established baselines.
    """
    
    def __init__(self, random_seed: int = 42):
        """Initialize A/B testing framework."""
        self.random_seed = random_seed
        np.random.seed(random_seed)
        self.experiment_history = []
        
    def run_experiment(self, 
                      config: ExperimentConfig,
                      variant_a_model: PhotonicNeuralNetwork,
                      variant_b_model: PhotonicNeuralNetwork,
                      evaluation_function: Callable,
                      X_test: np.ndarray,
                      y_test: np.ndarray) -> ExperimentResult:
        """
        Run complete A/B experiment with statistical analysis.
        
        Args:
            config: Experiment configuration
            variant_a_model: Baseline model (control)
            variant_b_model: Novel model (treatment) 
            evaluation_function: Function to evaluate model performance
            X_test: Test data
            y_test: Test labels
            
        Returns:
            Comprehensive experiment results with statistical analysis
        """
        logger.info(f"Starting A/B experiment: {config.name}")
        logger.info(f"Type: {config.experiment_type.value}, Runs per variant: {config.num_runs_per_variant}")
        
        # Run experiments for both variants
        variant_a_results = self._run_variant_experiments(
            variant_a_model, evaluation_function, X_test, y_test, 
            config.num_runs_per_variant, "Variant A (Control)"
        )
        
        variant_b_results = self._run_variant_experiments(
            variant_b_model, evaluation_function, X_test, y_test,
            config.num_runs_per_variant, "Variant B (Treatment)"
        )
        
        # Perform statistical analysis
        statistical_results = self._perform_statistical_analysis(
            variant_a_results, variant_b_results, config
        )
        
        # Create experiment result
        result = ExperimentResult(
            config=config,
            variant_a_results=variant_a_results,
            variant_b_results=variant_b_results,
            **statistical_results
        )
        
        # Store experiment
        self.experiment_history.append(result)
        
        logger.info(f"Experiment completed. p-value: {result.p_value:.4f}, "
                   f"Effect size: {result.effect_size:.4f}")
        
        return result
    
    def _run_variant_experiments(self,
                               model: PhotonicNeuralNetwork,
                               evaluation_function: Callable,
                               X_test: np.ndarray,
                               y_test: np.ndarray,
                               num_runs: int,
                               variant_name: str) -> List[Dict[str, float]]:
        """Run multiple experiments for a single variant."""
        logger.info(f"Running {num_runs} experiments for {variant_name}")
        
        results = []
        
        if self.config.parallel_execution:
            # Parallel execution for faster results
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                for run in range(num_runs):
                    future = executor.submit(
                        self._single_experiment_run, 
                        model, evaluation_function, X_test, y_test, run
                    )
                    futures.append(future)
                
                for future in futures:
                    result = future.result()
                    results.append(result)
        else:
            # Sequential execution
            for run in range(num_runs):
                result = self._single_experiment_run(
                    model, evaluation_function, X_test, y_test, run
                )
                results.append(result)
        
        return results
    
    def _single_experiment_run(self,
                             model: PhotonicNeuralNetwork,
                             evaluation_function: Callable,
                             X_test: np.ndarray,
                             y_test: np.ndarray,
                             run_id: int) -> Dict[str, float]:
        """Execute a single experimental run."""
        # Add run-specific randomization
        np.random.seed(self.random_seed + run_id)
        
        # Add slight data permutation to test robustness
        if len(X_test) > 100:
            indices = np.random.choice(len(X_test), size=min(100, len(X_test)), replace=False)
            X_run = X_test[indices]
            y_run = y_test[indices]
        else:
            X_run, y_run = X_test, y_test
        
        # Run evaluation
        start_time = time.perf_counter()
        result = evaluation_function(model, X_run, y_run)
        end_time = time.perf_counter()
        
        # Add timing information
        result["total_runtime_s"] = end_time - start_time
        result["run_id"] = run_id
        
        return result
    
    def _perform_statistical_analysis(self,
                                    variant_a_results: List[Dict[str, float]],
                                    variant_b_results: List[Dict[str, float]],
                                    config: ExperimentConfig) -> Dict[str, Any]:
        """Perform comprehensive statistical analysis."""
        # Extract primary metric values
        a_values = [r[config.primary_metric] for r in variant_a_results]
        b_values = [r[config.primary_metric] for r in variant_b_results]
        
        # Perform t-test
        from scipy.stats import ttest_ind, norm
        t_stat, p_value = ttest_ind(a_values, b_values)
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(((np.std(a_values) ** 2) + (np.std(b_values) ** 2)) / 2)
        effect_size = (np.mean(b_values) - np.mean(a_values)) / (pooled_std + 1e-8)
        
        # Calculate confidence interval
        mean_diff = np.mean(b_values) - np.mean(a_values)
        se_diff = np.sqrt(np.var(a_values)/len(a_values) + np.var(b_values)/len(b_values))
        alpha = 1 - config.confidence_level
        critical_value = norm.ppf(1 - alpha/2)
        
        ci_lower = mean_diff - critical_value * se_diff
        ci_upper = mean_diff + critical_value * se_diff
        confidence_interval = (ci_lower, ci_upper)
        
        # Calculate statistical power (post-hoc)
        from scipy.stats import norm
        z_alpha = norm.ppf(1 - (1 - config.confidence_level) / 2)
        z_beta = norm.ppf(config.statistical_power)
        n = len(a_values)
        power = 1 - norm.cdf(z_alpha - effect_size * np.sqrt(n/2))
        
        # Determine significance and winner
        significant = p_value < (1 - config.confidence_level)
        
        if significant:
            winner = "Variant B" if np.mean(b_values) > np.mean(a_values) else "Variant A"
        else:
            winner = None
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            significant, p_value, effect_size, power, config
        )
        
        # Calculate summary statistics for all metrics
        all_metrics = set()
        for r in variant_a_results + variant_b_results:
            all_metrics.update(r.keys())
        
        variant_a_mean = {metric: np.mean([r.get(metric, 0) for r in variant_a_results]) 
                         for metric in all_metrics}
        variant_b_mean = {metric: np.mean([r.get(metric, 0) for r in variant_b_results])
                         for metric in all_metrics}
        variant_a_std = {metric: np.std([r.get(metric, 0) for r in variant_a_results])
                        for metric in all_metrics}
        variant_b_std = {metric: np.std([r.get(metric, 0) for r in variant_b_results])
                        for metric in all_metrics}
        
        return {
            "p_value": p_value,
            "confidence_interval": confidence_interval,
            "effect_size": effect_size,
            "statistical_power": power,
            "variant_a_mean": variant_a_mean,
            "variant_b_mean": variant_b_mean,
            "variant_a_std": variant_a_std,
            "variant_b_std": variant_b_std,
            "significant_difference": significant,
            "winner": winner,
            "recommendation": recommendation
        }
    
    def _generate_recommendation(self,
                               significant: bool,
                               p_value: float,
                               effect_size: float,
                               power: float,
                               config: ExperimentConfig) -> str:
        """Generate actionable recommendation based on results."""
        if not significant:
            if power < 0.8:
                return (f"No significant difference found (p={p_value:.4f}). "
                       f"Consider increasing sample size for adequate power (current: {power:.2f}).")
            else:
                return (f"No significant difference found (p={p_value:.4f}) with adequate power. "
                       f"Variants perform similarly on {config.primary_metric}.")
        
        if effect_size > 0.8:
            effect_desc = "large"
        elif effect_size > 0.5:
            effect_desc = "medium"
        elif effect_size > 0.2:
            effect_desc = "small"
        else:
            effect_desc = "minimal"
        
        winner = "Variant B" if effect_size > 0 else "Variant A"
        
        return (f"Significant difference found (p={p_value:.4f}) with {effect_desc} effect size "
               f"({effect_size:.3f}). Recommend {winner} for {config.primary_metric} optimization.")
    
    def compare_multiple_variants(self,
                                config: ExperimentConfig,
                                models: Dict[str, PhotonicNeuralNetwork],
                                evaluation_function: Callable,
                                X_test: np.ndarray,
                                y_test: np.ndarray) -> Dict[str, ExperimentResult]:
        """
        Compare multiple variants using pairwise A/B tests.
        
        Args:
            config: Experiment configuration
            models: Dictionary of named models to compare
            evaluation_function: Model evaluation function
            X_test: Test data
            y_test: Test labels
            
        Returns:
            Dictionary of pairwise comparison results
        """
        logger.info(f"Running multi-variant comparison with {len(models)} models")
        
        results = {}
        model_names = list(models.keys())
        
        # Perform pairwise comparisons
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                name_a, name_b = model_names[i], model_names[j]
                
                # Create experiment configuration for this pair
                pair_config = ExperimentConfig(
                    name=f"{config.name}_{name_a}_vs_{name_b}",
                    experiment_type=config.experiment_type,
                    description=f"Comparison between {name_a} and {name_b}",
                    **{k: v for k, v in config.__dict__.items() 
                       if k not in ['name', 'description']}
                )
                
                # Run pairwise experiment
                result = self.run_experiment(
                    pair_config, models[name_a], models[name_b],
                    evaluation_function, X_test, y_test
                )
                
                results[f"{name_a}_vs_{name_b}"] = result
        
        return results
    
    def generate_experiment_report(self, result: ExperimentResult) -> Dict[str, Any]:
        """Generate comprehensive experiment report."""
        config = result.config
        
        report = {
            "experiment_info": {
                "name": config.name,
                "type": config.experiment_type.value,
                "description": config.description,
                "runs_per_variant": config.num_runs_per_variant
            },
            "statistical_summary": {
                "p_value": result.p_value,
                "effect_size": result.effect_size,
                "confidence_interval": result.confidence_interval,
                "statistical_power": result.statistical_power,
                "significant": result.significant_difference
            },
            "performance_comparison": {
                "primary_metric": config.primary_metric,
                "variant_a_mean": result.variant_a_mean[config.primary_metric],
                "variant_b_mean": result.variant_b_mean[config.primary_metric],
                "improvement": ((result.variant_b_mean[config.primary_metric] - 
                               result.variant_a_mean[config.primary_metric]) / 
                               result.variant_a_mean[config.primary_metric] * 100),
            },
            "secondary_metrics": {},
            "conclusion": {
                "winner": result.winner,
                "recommendation": result.recommendation
            }
        }
        
        # Add secondary metrics
        for metric in config.secondary_metrics:
            if metric in result.variant_a_mean:
                report["secondary_metrics"][metric] = {
                    "variant_a": result.variant_a_mean[metric],
                    "variant_b": result.variant_b_mean[metric],
                    "improvement": ((result.variant_b_mean[metric] - 
                                   result.variant_a_mean[metric]) / 
                                   result.variant_a_mean[metric] * 100)
                }
        
        return report
    
    def save_experiment_results(self, filepath: str):
        """Save all experiment results to file."""
        export_data = []
        
        for result in self.experiment_history:
            report = self.generate_experiment_report(result)
            export_data.append(report)
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Saved {len(export_data)} experiment results to {filepath}")


def create_standard_evaluation_function() -> Callable:
    """Create standard evaluation function for photonic neural networks."""
    
    def evaluate_model(model: PhotonicNeuralNetwork, 
                      X_test: np.ndarray, 
                      y_test: np.ndarray) -> Dict[str, float]:
        """Standard evaluation function."""
        model.set_training(False)
        
        # Measure performance
        start_time = time.perf_counter_ns()
        outputs, hardware_metrics = model.forward(X_test, measure_latency=True)
        end_time = time.perf_counter_ns()
        
        # Calculate metrics
        predictions = np.argmax(outputs, axis=1)
        true_labels = np.argmax(y_test, axis=1)
        accuracy = np.mean(predictions == true_labels)
        
        # Hardware metrics
        avg_latency_ns = hardware_metrics["total_latency_ns"] / len(X_test)
        power_mw = hardware_metrics["total_power_mw"]
        
        return {
            "accuracy": accuracy,
            "latency_ns": avg_latency_ns,
            "power_mw": power_mw,
            "throughput_ops": len(X_test) * 1e9 / (end_time - start_time),
            "energy_per_op_fj": power_mw * 1e-3 * avg_latency_ns * 1e-9 / len(X_test) * 1e15
        }
    
    return evaluate_model