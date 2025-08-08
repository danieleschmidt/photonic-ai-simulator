"""
Comprehensive benchmarking and statistical analysis framework.

Implements rigorous performance validation, statistical significance testing,
and comparative analysis against literature benchmarks.
"""

import pytest
import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Any
from scipy import stats
from dataclasses import dataclass
import json

from src.models import create_benchmark_network
from src.benchmarks import MNISTBenchmark, CIFAR10Benchmark, VowelClassificationBenchmark
from src.benchmarks import BenchmarkConfig, run_comprehensive_benchmarks
from src.optimization import create_optimized_network, OptimizationConfig
from src.experiments.ab_testing import ABTestFramework, ExperimentConfig, ExperimentType


logger = logging.getLogger(__name__)


@dataclass
class PerformanceBaseline:
    """Reference performance baselines from literature."""
    # MIT demonstrations (2024)
    mit_vowel_accuracy: float = 0.925  # 92.5% on vowel classification
    mit_latency_ps: float = 410        # 410 picoseconds
    
    # Hardware-aware training results
    mnist_accuracy_target: float = 0.95   # 95% at 50 GHz
    cifar10_accuracy_target: float = 0.806 # 80.6% with optimization
    
    # Power consumption targets
    max_power_per_shifter_mw: float = 15.0  # 15mW per phase shifter
    total_system_power_mw: float = 500.0    # 500mW total budget


class ComprehensiveStatisticalAnalysis:
    """
    Advanced statistical analysis framework for photonic neural network benchmarks.
    
    Implements rigorous statistical validation including:
    - Multiple comparison corrections (Bonferroni, Holm-Sidak)
    - Effect size calculations (Cohen's d, Glass's delta)
    - Power analysis and sample size determination
    - Bootstrap confidence intervals
    - Distribution testing and normality checks
    """
    
    def __init__(self, alpha: float = 0.05, power: float = 0.8):
        """
        Initialize statistical analysis framework.
        
        Args:
            alpha: Significance level (Type I error rate)
            power: Statistical power (1 - Type II error rate)
        """
        self.alpha = alpha
        self.power = power
        self.baseline = PerformanceBaseline()
        
    def validate_against_literature(self, 
                                  results: Dict[str, Any],
                                  task: str) -> Dict[str, Any]:
        """
        Validate results against published literature benchmarks.
        
        Args:
            results: Benchmark results to validate
            task: Task type ("mnist", "cifar10", "vowel_classification")
            
        Returns:
            Statistical validation report
        """
        logger.info(f"Validating {task} results against literature baselines")
        
        validation_report = {
            "task": task,
            "validation_passed": True,
            "comparisons": [],
            "overall_assessment": ""
        }
        
        # Get task-specific targets
        targets = self._get_literature_targets(task)
        
        # Validate each metric
        for metric_name, target_value in targets.items():
            if metric_name in results:
                observed_value = results[metric_name]
                comparison = self._compare_to_target(
                    observed_value, target_value, metric_name
                )
                validation_report["comparisons"].append(comparison)
                
                if not comparison["meets_target"]:
                    validation_report["validation_passed"] = False
        
        # Generate overall assessment
        validation_report["overall_assessment"] = self._generate_assessment(
            validation_report["comparisons"]
        )
        
        return validation_report
    
    def _get_literature_targets(self, task: str) -> Dict[str, float]:
        """Get literature performance targets for specific task."""
        if task == "mnist":
            return {
                "accuracy": self.baseline.mnist_accuracy_target,
                "max_latency_ns": 1.0,  # Sub-nanosecond target
                "max_power_mw": self.baseline.total_system_power_mw
            }
        elif task == "cifar10":
            return {
                "accuracy": self.baseline.cifar10_accuracy_target,
                "max_latency_ns": 1.0,
                "max_power_mw": self.baseline.total_system_power_mw
            }
        elif task == "vowel_classification":
            return {
                "accuracy": self.baseline.mit_vowel_accuracy,
                "max_latency_ns": self.baseline.mit_latency_ps / 1000,  # Convert ps to ns
                "max_power_mw": 100.0  # Small network power budget
            }
        else:
            return {}
    
    def _compare_to_target(self, observed: float, target: float, metric: str) -> Dict[str, Any]:
        """Compare observed value to literature target."""
        if "accuracy" in metric:
            # Higher is better for accuracy
            meets_target = observed >= target * 0.95  # 5% tolerance
            improvement = ((observed - target) / target) * 100
        elif "latency" in metric or "power" in metric:
            # Lower is better for latency and power
            meets_target = observed <= target * 1.05  # 5% tolerance
            improvement = ((target - observed) / target) * 100
        else:
            # Default comparison
            meets_target = abs(observed - target) / target < 0.05
            improvement = ((observed - target) / target) * 100
        
        return {
            "metric": metric,
            "observed": observed,
            "target": target,
            "meets_target": meets_target,
            "improvement_percent": improvement,
            "tolerance_met": meets_target
        }
    
    def _generate_assessment(self, comparisons: List[Dict[str, Any]]) -> str:
        """Generate overall assessment based on comparisons."""
        total_comparisons = len(comparisons)
        passed_comparisons = sum(1 for c in comparisons if c["meets_target"])
        
        if passed_comparisons == total_comparisons:
            return f"EXCELLENT: All {total_comparisons} metrics meet or exceed literature targets"
        elif passed_comparisons >= total_comparisons * 0.8:
            return f"GOOD: {passed_comparisons}/{total_comparisons} metrics meet targets"
        elif passed_comparisons >= total_comparisons * 0.6:
            return f"ACCEPTABLE: {passed_comparisons}/{total_comparisons} metrics meet targets"
        else:
            return f"NEEDS IMPROVEMENT: Only {passed_comparisons}/{total_comparisons} metrics meet targets"
    
    def compute_effect_sizes(self, 
                           group_a_results: List[float],
                           group_b_results: List[float]) -> Dict[str, float]:
        """
        Compute multiple effect size measures.
        
        Args:
            group_a_results: Control group results
            group_b_results: Treatment group results
            
        Returns:
            Dictionary of effect size measures
        """
        # Convert to numpy arrays
        a = np.array(group_a_results)
        b = np.array(group_b_results)
        
        # Cohen's d (standardized mean difference)
        pooled_std = np.sqrt(((len(a) - 1) * np.var(a) + (len(b) - 1) * np.var(b)) / 
                           (len(a) + len(b) - 2))
        cohens_d = (np.mean(b) - np.mean(a)) / pooled_std
        
        # Glass's delta (using control group std)
        glass_delta = (np.mean(b) - np.mean(a)) / np.std(a)
        
        # Hedge's g (bias-corrected Cohen's d)
        correction_factor = 1 - (3 / (4 * (len(a) + len(b)) - 9))
        hedges_g = cohens_d * correction_factor
        
        # Cliff's delta (non-parametric effect size)
        cliffs_delta = self._compute_cliffs_delta(a, b)
        
        return {
            "cohens_d": cohens_d,
            "glass_delta": glass_delta,
            "hedges_g": hedges_g,
            "cliffs_delta": cliffs_delta,
            "interpretation": self._interpret_effect_size(abs(cohens_d))
        }
    
    def _compute_cliffs_delta(self, group_a: np.ndarray, group_b: np.ndarray) -> float:
        """Compute Cliff's delta non-parametric effect size."""
        n_a, n_b = len(group_a), len(group_b)
        
        # Count pairs where b > a minus pairs where a > b
        dominance = 0
        for a_val in group_a:
            for b_val in group_b:
                if b_val > a_val:
                    dominance += 1
                elif a_val > b_val:
                    dominance -= 1
        
        return dominance / (n_a * n_b)
    
    def _interpret_effect_size(self, effect_size: float) -> str:
        """Interpret effect size magnitude."""
        if effect_size < 0.2:
            return "negligible"
        elif effect_size < 0.5:
            return "small"
        elif effect_size < 0.8:
            return "medium"
        else:
            return "large"
    
    def bootstrap_confidence_intervals(self, 
                                     data: np.ndarray,
                                     statistic_func: callable = np.mean,
                                     confidence_level: float = 0.95,
                                     n_bootstrap: int = 10000) -> Tuple[float, float]:
        """
        Compute bootstrap confidence intervals.
        
        Args:
            data: Input data
            statistic_func: Function to compute statistic
            confidence_level: Confidence level (0-1)
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        bootstrap_stats = []
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_stat = statistic_func(bootstrap_sample)
            bootstrap_stats.append(bootstrap_stat)
        
        # Compute confidence interval
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_stats, lower_percentile)
        ci_upper = np.percentile(bootstrap_stats, upper_percentile)
        
        return ci_lower, ci_upper
    
    def test_normality(self, data: np.ndarray) -> Dict[str, Any]:
        """Test normality assumptions using multiple tests."""
        results = {}
        
        # Shapiro-Wilk test (best for small samples)
        if len(data) <= 5000:
            shapiro_stat, shapiro_p = stats.shapiro(data)
            results["shapiro_wilk"] = {
                "statistic": shapiro_stat,
                "p_value": shapiro_p,
                "is_normal": shapiro_p > self.alpha
            }
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_p = stats.kstest(data, 'norm', args=(np.mean(data), np.std(data)))
        results["kolmogorov_smirnov"] = {
            "statistic": ks_stat,
            "p_value": ks_p,
            "is_normal": ks_p > self.alpha
        }
        
        # Anderson-Darling test
        ad_result = stats.anderson(data)
        results["anderson_darling"] = {
            "statistic": ad_result.statistic,
            "critical_values": ad_result.critical_values,
            "significance_levels": ad_result.significance_levels
        }
        
        return results
    
    def multiple_comparisons_correction(self, 
                                      p_values: List[float],
                                      method: str = "holm") -> List[float]:
        """
        Apply multiple comparisons correction.
        
        Args:
            p_values: List of uncorrected p-values
            method: Correction method ("bonferroni", "holm", "benjamini_hochberg")
            
        Returns:
            List of corrected p-values
        """
        p_array = np.array(p_values)
        n = len(p_array)
        
        if method == "bonferroni":
            return np.minimum(p_array * n, 1.0).tolist()
        
        elif method == "holm":
            # Holm-Bonferroni method
            sorted_indices = np.argsort(p_array)
            sorted_p = p_array[sorted_indices]
            
            corrected_p = np.zeros_like(sorted_p)
            for i, p in enumerate(sorted_p):
                corrected_p[i] = min(p * (n - i), 1.0)
            
            # Enforce monotonicity
            for i in range(1, len(corrected_p)):
                corrected_p[i] = max(corrected_p[i], corrected_p[i-1])
            
            # Restore original order
            final_corrected = np.zeros_like(corrected_p)
            final_corrected[sorted_indices] = corrected_p
            
            return final_corrected.tolist()
        
        elif method == "benjamini_hochberg":
            # Benjamini-Hochberg FDR control
            sorted_indices = np.argsort(p_array)
            sorted_p = p_array[sorted_indices]
            
            corrected_p = np.zeros_like(sorted_p)
            for i in range(len(sorted_p) - 1, -1, -1):
                if i == len(sorted_p) - 1:
                    corrected_p[i] = sorted_p[i]
                else:
                    corrected_p[i] = min(sorted_p[i] * n / (i + 1), corrected_p[i + 1])
            
            # Restore original order
            final_corrected = np.zeros_like(corrected_p)
            final_corrected[sorted_indices] = corrected_p
            
            return final_corrected.tolist()
        
        else:
            raise ValueError(f"Unknown correction method: {method}")


class PerformanceRegressionSuite:
    """Performance regression testing to ensure consistent improvements."""
    
    def __init__(self):
        self.historical_results = {}
        self.regression_thresholds = {
            "accuracy": 0.02,      # 2% accuracy regression threshold
            "latency_ns": 0.10,    # 10% latency regression threshold  
            "power_mw": 0.15       # 15% power regression threshold
        }
    
    def record_baseline(self, task: str, results: Dict[str, float]):
        """Record baseline performance for regression testing."""
        self.historical_results[task] = results
        logger.info(f"Recorded baseline for {task}: {results}")
    
    def check_regression(self, task: str, current_results: Dict[str, float]) -> Dict[str, Any]:
        """Check for performance regressions against historical baselines."""
        if task not in self.historical_results:
            return {"status": "no_baseline", "message": "No historical baseline available"}
        
        baseline = self.historical_results[task]
        regressions = []
        improvements = []
        
        for metric, current_value in current_results.items():
            if metric in baseline:
                baseline_value = baseline[metric]
                
                if metric == "accuracy":
                    # Higher is better for accuracy
                    change_ratio = (current_value - baseline_value) / baseline_value
                    if change_ratio < -self.regression_thresholds[metric]:
                        regressions.append({
                            "metric": metric,
                            "baseline": baseline_value,
                            "current": current_value,
                            "regression_percent": abs(change_ratio) * 100
                        })
                    elif change_ratio > 0:
                        improvements.append({
                            "metric": metric,
                            "improvement_percent": change_ratio * 100
                        })
                
                elif metric in ["latency_ns", "power_mw"]:
                    # Lower is better for latency and power
                    change_ratio = (current_value - baseline_value) / baseline_value
                    if change_ratio > self.regression_thresholds.get(metric, 0.10):
                        regressions.append({
                            "metric": metric,
                            "baseline": baseline_value,
                            "current": current_value,
                            "regression_percent": change_ratio * 100
                        })
                    elif change_ratio < 0:
                        improvements.append({
                            "metric": metric,
                            "improvement_percent": abs(change_ratio) * 100
                        })
        
        return {
            "status": "regression_detected" if regressions else "no_regression",
            "regressions": regressions,
            "improvements": improvements,
            "overall_assessment": self._assess_performance_change(regressions, improvements)
        }
    
    def _assess_performance_change(self, regressions: List, improvements: List) -> str:
        """Generate overall assessment of performance changes."""
        if not regressions and improvements:
            return f"IMPROVEMENT: {len(improvements)} metrics improved, no regressions"
        elif regressions and not improvements:
            return f"REGRESSION: {len(regressions)} metrics regressed"
        elif regressions and improvements:
            return f"MIXED: {len(improvements)} improvements, {len(regressions)} regressions"
        else:
            return "STABLE: No significant changes detected"


@pytest.mark.benchmark
class TestComprehensiveBenchmarkSuite:
    """Comprehensive benchmark test suite with statistical validation."""
    
    def setup_method(self):
        """Set up test environment."""
        self.statistical_analyzer = ComprehensiveStatisticalAnalysis()
        self.regression_suite = PerformanceRegressionSuite()
        self.ab_test_framework = ABTestFramework()
        
    def test_mnist_literature_validation(self):
        """Test MNIST benchmark against literature targets."""
        logger.info("Running MNIST literature validation test")
        
        # Create optimized network
        model = create_optimized_network("mnist", "high")
        
        # Run benchmark
        benchmark = MNISTBenchmark(BenchmarkConfig(num_runs=3))
        result = benchmark.run_benchmark(model)
        
        # Validate against literature
        validation = self.statistical_analyzer.validate_against_literature(
            {
                "accuracy": result.accuracy,
                "latency_ns": result.latency_ns,
                "power_mw": result.power_consumption_mw
            },
            "mnist"
        )
        
        # Assert performance targets
        assert validation["validation_passed"], f"MNIST validation failed: {validation['overall_assessment']}"
        assert result.accuracy >= 0.90, f"MNIST accuracy {result.accuracy} below minimum 90%"
        assert result.latency_ns <= 10.0, f"MNIST latency {result.latency_ns}ns exceeds 10ns limit"
        
    def test_vowel_classification_mit_baseline(self):
        """Test vowel classification against MIT demonstration baseline."""
        logger.info("Running vowel classification MIT baseline test")
        
        # Create network matching MIT demonstration
        model = create_optimized_network("vowel_classification", "extreme")
        
        # Run benchmark
        benchmark = VowelClassificationBenchmark(BenchmarkConfig(num_runs=5))
        result = benchmark.run_benchmark(model)
        
        # Validate against MIT baseline
        validation = self.statistical_analyzer.validate_against_literature(
            {
                "accuracy": result.accuracy,
                "latency_ns": result.latency_ns,
                "power_mw": result.power_consumption_mw
            },
            "vowel_classification"
        )
        
        # Assert MIT performance targets
        assert result.accuracy >= 0.90, f"Vowel accuracy {result.accuracy} below MIT baseline range"
        assert result.latency_ns <= 1.0, f"Vowel latency {result.latency_ns}ns exceeds sub-ns target"
        
    def test_optimization_effectiveness_ab_test(self):
        """Test optimization effectiveness using A/B testing framework."""
        logger.info("Running optimization effectiveness A/B test")
        
        # Create baseline and optimized models
        baseline_model = create_benchmark_network("mnist")
        optimized_model = create_optimized_network("mnist", "high")
        
        # Generate test data
        X_test = np.random.randn(100, 784)
        y_test = np.eye(10)[np.random.randint(0, 10, 100)]
        
        # Configure A/B experiment
        config = ExperimentConfig(
            name="optimization_effectiveness",
            experiment_type=ExperimentType.OPTIMIZATION,
            description="Testing optimization impact on performance",
            num_runs_per_variant=10,
            primary_metric="latency_ns",
            secondary_metrics=["accuracy", "power_mw"]
        )
        
        # Create evaluation function
        def evaluate_model(model, X, y):
            outputs, metrics = model.forward(X, measure_latency=True)
            predictions = np.argmax(outputs, axis=1)
            true_labels = np.argmax(y, axis=1)
            accuracy = np.mean(predictions == true_labels)
            
            return {
                "accuracy": accuracy,
                "latency_ns": metrics["total_latency_ns"] / len(X),
                "power_mw": metrics["total_power_mw"]
            }
        
        # Run A/B experiment
        experiment_result = self.ab_test_framework.run_experiment(
            config, baseline_model, optimized_model, evaluate_model, X_test, y_test
        )
        
        # Assert optimization effectiveness
        assert experiment_result.significant_difference, "No significant optimization improvement detected"
        assert experiment_result.winner == "Variant B", "Optimized model did not outperform baseline"
        
        # Check effect size
        effect_sizes = self.statistical_analyzer.compute_effect_sizes(
            [r["latency_ns"] for r in experiment_result.variant_a_results],
            [r["latency_ns"] for r in experiment_result.variant_b_results]
        )
        
        assert abs(effect_sizes["cohens_d"]) > 0.5, "Optimization effect size too small"
        
    def test_statistical_power_analysis(self):
        """Test statistical power of benchmark experiments."""
        logger.info("Running statistical power analysis")
        
        # Generate simulated results for power analysis
        baseline_accuracies = np.random.normal(0.85, 0.02, 50)  # 85% ± 2%
        improved_accuracies = np.random.normal(0.90, 0.02, 50)  # 90% ± 2%
        
        # Test normality assumptions
        normality_baseline = self.statistical_analyzer.test_normality(baseline_accuracies)
        normality_improved = self.statistical_analyzer.test_normality(improved_accuracies)
        
        # Compute effect size
        effect_sizes = self.statistical_analyzer.compute_effect_sizes(
            baseline_accuracies.tolist(), improved_accuracies.tolist()
        )
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(baseline_accuracies, improved_accuracies)
        
        # Bootstrap confidence intervals
        ci_baseline = self.statistical_analyzer.bootstrap_confidence_intervals(baseline_accuracies)
        ci_improved = self.statistical_analyzer.bootstrap_confidence_intervals(improved_accuracies)
        
        # Assert statistical validity
        assert p_value < 0.05, f"Statistical test not significant (p={p_value})"
        assert abs(effect_sizes["cohens_d"]) > 0.8, "Effect size not large enough for practical significance"
        assert ci_improved[0] > ci_baseline[1], "Confidence intervals overlap significantly"
        
    def test_performance_regression_detection(self):
        """Test performance regression detection system."""
        logger.info("Running performance regression detection test")
        
        # Record baseline performance
        baseline_results = {
            "accuracy": 0.92,
            "latency_ns": 0.8,
            "power_mw": 450.0
        }
        self.regression_suite.record_baseline("test_task", baseline_results)
        
        # Test case 1: No regression
        current_results_stable = {
            "accuracy": 0.921,
            "latency_ns": 0.82,
            "power_mw": 455.0
        }
        regression_check = self.regression_suite.check_regression("test_task", current_results_stable)
        assert regression_check["status"] == "no_regression"
        
        # Test case 2: Accuracy regression
        current_results_regressed = {
            "accuracy": 0.88,  # 4.3% drop - should trigger regression
            "latency_ns": 0.75,
            "power_mw": 430.0
        }
        regression_check = self.regression_suite.check_regression("test_task", current_results_regressed)
        assert regression_check["status"] == "regression_detected"
        assert len(regression_check["regressions"]) > 0
        
    def test_multiple_comparisons_correction(self):
        """Test multiple comparisons correction methods."""
        logger.info("Running multiple comparisons correction test")
        
        # Generate multiple p-values (some significant, some not)
        p_values = [0.001, 0.01, 0.03, 0.05, 0.08, 0.12, 0.20, 0.35, 0.50, 0.80]
        
        # Test different correction methods
        bonferroni_corrected = self.statistical_analyzer.multiple_comparisons_correction(
            p_values, "bonferroni"
        )
        holm_corrected = self.statistical_analyzer.multiple_comparisons_correction(
            p_values, "holm"
        )
        bh_corrected = self.statistical_analyzer.multiple_comparisons_correction(
            p_values, "benjamini_hochberg"
        )
        
        # Assert corrections are more conservative than uncorrected
        for i, original_p in enumerate(p_values):
            assert bonferroni_corrected[i] >= original_p, "Bonferroni correction should increase p-values"
            assert holm_corrected[i] >= original_p, "Holm correction should increase p-values"
            # BH correction can decrease p-values, so we don't assert this
        
        # Assert Bonferroni is most conservative
        for i in range(len(p_values)):
            assert bonferroni_corrected[i] >= holm_corrected[i], "Bonferroni should be more conservative than Holm"
    
    def test_comprehensive_benchmark_integration(self):
        """Integration test running full benchmark suite with statistical analysis."""
        logger.info("Running comprehensive benchmark integration test")
        
        # Run full benchmark suite
        results, comparative_analysis = run_comprehensive_benchmarks(save_results=False)
        
        # Validate all benchmark results
        for task_name, task_result in results.items():
            validation = self.statistical_analyzer.validate_against_literature(
                {
                    "accuracy": task_result.accuracy,
                    "latency_ns": task_result.latency_ns, 
                    "power_mw": task_result.power_consumption_mw
                },
                task_name
            )
            
            # Log validation results
            logger.info(f"{task_name} validation: {validation['overall_assessment']}")
            
            # Basic performance assertions
            assert task_result.accuracy > 0.7, f"{task_name} accuracy too low: {task_result.accuracy}"
            assert task_result.latency_ns < 100.0, f"{task_name} latency too high: {task_result.latency_ns}ns"
            assert task_result.power_consumption_mw < 1000.0, f"{task_name} power too high: {task_result.power_consumption_mw}mW"
        
        # Validate comparative analysis
        assert "overall_performance" in comparative_analysis
        assert "hardware_efficiency" in comparative_analysis
        assert comparative_analysis["overall_performance"]["avg_accuracy"] > 0.8
        assert comparative_analysis["hardware_efficiency"]["avg_speedup_vs_gpu"] > 100  # 100x speedup target


if __name__ == "__main__":
    # Run comprehensive benchmark suite
    pytest.main([__file__, "-v", "--tb=short"])