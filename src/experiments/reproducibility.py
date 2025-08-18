"""
Reproducibility framework for photonic AI research.

Provides comprehensive tools for ensuring experimental reproducibility,
including seed management, environment tracking, and result validation.
"""

import json
import hashlib
import platform
import time
import os
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

@dataclass
class ExperimentMetadata:
    """Comprehensive experiment metadata for reproducibility."""
    experiment_id: str
    timestamp: str
    platform_info: Dict[str, str]
    python_version: str
    numpy_version: str
    random_seed: int
    git_commit: Optional[str]
    environment_hash: str
    parameter_hash: str
    hardware_info: Dict[str, Any]

@dataclass
class ExperimentResult:
    """Standardized experiment result format."""
    experiment_id: str
    metrics: Dict[str, float]
    timing_info: Dict[str, float]
    resource_usage: Dict[str, float]
    validation_status: str
    reproducibility_score: float
    artifacts: List[str]

class ReproducibilityFramework:
    """
    Comprehensive reproducibility framework for photonic AI experiments.
    
    Ensures experimental reproducibility through standardized seed management,
    environment tracking, and result validation protocols.
    """
    
    def __init__(self, experiment_name: str, base_seed: int = 42):
        self.experiment_name = experiment_name
        self.base_seed = base_seed
        self.experiment_id = self._generate_experiment_id()
        self.metadata = self._collect_metadata()
        self.results_history: List[ExperimentResult] = []
        
    def _generate_experiment_id(self) -> str:
        """Generate unique experiment identifier."""
        timestamp = datetime.now().isoformat()
        id_string = f"{self.experiment_name}_{timestamp}_{self.base_seed}"
        return hashlib.md5(id_string.encode()).hexdigest()[:12]
    
    def _collect_metadata(self) -> ExperimentMetadata:
        """Collect comprehensive experiment metadata."""
        # Platform information
        platform_info = {
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'architecture': platform.architecture()[0]
        }
        
        # Hardware information
        hardware_info = {}
        if PSUTIL_AVAILABLE:
            hardware_info.update({
                'cpu_count': psutil.cpu_count(),
                'cpu_freq_max': psutil.cpu_freq().max if psutil.cpu_freq() else None,
                'memory_total': psutil.virtual_memory().total,
                'disk_usage': psutil.disk_usage('/').total
            })
        else:
            hardware_info.update({
                'cpu_count': os.cpu_count(),
                'memory_total': 'unknown',
                'disk_usage': 'unknown'
            })
        
        # Environment hash
        env_vars = sorted([f"{k}={v}" for k, v in os.environ.items() 
                          if not k.startswith('_') and 'TOKEN' not in k and 'KEY' not in k])
        env_string = '|'.join(env_vars)
        environment_hash = hashlib.sha256(env_string.encode()).hexdigest()[:16]
        
        # Parameter hash (will be updated when parameters are set)
        parameter_hash = hashlib.sha256(str(self.base_seed).encode()).hexdigest()[:16]
        
        # Git commit (if available)
        git_commit = self._get_git_commit()
        
        return ExperimentMetadata(
            experiment_id=self.experiment_id,
            timestamp=datetime.now().isoformat(),
            platform_info=platform_info,
            python_version=platform.python_version(),
            numpy_version=np.__version__,
            random_seed=self.base_seed,
            git_commit=git_commit,
            environment_hash=environment_hash,
            parameter_hash=parameter_hash,
            hardware_info=hardware_info
        )
    
    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash."""
        try:
            import subprocess
            result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            pass
        return None
    
    def set_deterministic_seeds(self, additional_seed: int = 0) -> Dict[str, int]:
        """
        Set deterministic seeds for all random number generators.
        
        Args:
            additional_seed: Additional seed offset for experiment variations
            
        Returns:
            Dictionary of all seeds used
        """
        final_seed = self.base_seed + additional_seed
        
        # Set NumPy seed
        np.random.seed(final_seed)
        
        # Set Python built-in random seed
        import random
        random.seed(final_seed)
        
        # Try to set PyTorch seed if available
        torch_seed = None
        try:
            import torch
            torch.manual_seed(final_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(final_seed)
            torch_seed = final_seed
        except ImportError:
            pass
        
        # Try to set JAX seed if available
        jax_seed = None
        try:
            import jax
            jax_seed = final_seed
        except ImportError:
            pass
        
        seeds = {
            'base_seed': self.base_seed,
            'additional_seed': additional_seed,
            'final_seed': final_seed,
            'numpy_seed': final_seed,
            'python_seed': final_seed
        }
        
        if torch_seed is not None:
            seeds['torch_seed'] = torch_seed
        if jax_seed is not None:
            seeds['jax_seed'] = jax_seed
            
        return seeds
    
    def create_experiment_context(self, parameters: Dict[str, Any]) -> 'ExperimentContext':
        """
        Create a managed experiment context.
        
        Args:
            parameters: Experiment parameters
            
        Returns:
            ExperimentContext manager
        """
        return ExperimentContext(self, parameters)
    
    def validate_reproducibility(self, 
                               results: List[ExperimentResult],
                               tolerance: float = 1e-6) -> Dict[str, Any]:
        """
        Validate reproducibility of experiment results.
        
        Args:
            results: List of experiment results to validate
            tolerance: Numerical tolerance for result comparison
            
        Returns:
            Validation report with reproducibility metrics
        """
        if len(results) < 2:
            return {
                'status': 'insufficient_data',
                'message': 'Need at least 2 results for reproducibility validation',
                'reproducibility_score': 0.0
            }
        
        # Compare metrics across results
        metric_comparisons = {}
        reference_result = results[0]
        
        for metric_name in reference_result.metrics:
            values = [r.metrics.get(metric_name, float('nan')) for r in results]
            values_array = np.array(values)
            
            if np.any(np.isnan(values_array)):
                metric_comparisons[metric_name] = {
                    'status': 'missing_values',
                    'reproducible': False
                }
                continue
            
            # Calculate coefficient of variation
            mean_val = np.mean(values_array)
            std_val = np.std(values_array)
            cv = std_val / abs(mean_val) if abs(mean_val) > 1e-10 else float('inf')
            
            # Check if all values are within tolerance
            max_diff = np.max(np.abs(values_array - mean_val))
            reproducible = max_diff <= tolerance
            
            metric_comparisons[metric_name] = {
                'mean': mean_val,
                'std': std_val,
                'coefficient_of_variation': cv,
                'max_difference': max_diff,
                'reproducible': reproducible,
                'tolerance_met': max_diff <= tolerance
            }
        
        # Calculate overall reproducibility score
        reproducible_metrics = sum(1 for comp in metric_comparisons.values() 
                                 if comp.get('reproducible', False))
        total_metrics = len(metric_comparisons)
        reproducibility_score = reproducible_metrics / total_metrics if total_metrics > 0 else 0.0
        
        # Overall status
        if reproducibility_score >= 0.9:
            status = 'highly_reproducible'
        elif reproducibility_score >= 0.7:
            status = 'moderately_reproducible'
        elif reproducibility_score >= 0.5:
            status = 'partially_reproducible'
        else:
            status = 'poorly_reproducible'
        
        return {
            'status': status,
            'reproducibility_score': reproducibility_score,
            'metric_comparisons': metric_comparisons,
            'num_experiments': len(results),
            'recommendations': self._generate_reproducibility_recommendations(metric_comparisons)
        }
    
    def _generate_reproducibility_recommendations(self, metric_comparisons: Dict) -> List[str]:
        """Generate recommendations for improving reproducibility."""
        recommendations = []
        
        high_variance_metrics = [name for name, comp in metric_comparisons.items()
                               if comp.get('coefficient_of_variation', 0) > 0.1]
        
        if high_variance_metrics:
            recommendations.append(
                f"High variance detected in metrics: {', '.join(high_variance_metrics)}. "
                "Consider increasing sample size or reducing experimental noise."
            )
        
        missing_metrics = [name for name, comp in metric_comparisons.items()
                          if comp.get('status') == 'missing_values']
        
        if missing_metrics:
            recommendations.append(
                f"Missing values in metrics: {', '.join(missing_metrics)}. "
                "Ensure all experiments collect the same metrics."
            )
        
        non_reproducible = [name for name, comp in metric_comparisons.items()
                           if not comp.get('reproducible', True)]
        
        if len(non_reproducible) > len(metric_comparisons) / 2:
            recommendations.append(
                "Many metrics show poor reproducibility. Check seed management, "
                "environment consistency, and measurement protocols."
            )
        
        return recommendations
    
    def save_metadata(self, filepath: str) -> None:
        """Save experiment metadata to file."""
        with open(filepath, 'w') as f:
            json.dump(asdict(self.metadata), f, indent=2, default=str)
    
    def load_metadata(self, filepath: str) -> ExperimentMetadata:
        """Load experiment metadata from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return ExperimentMetadata(**data)

class ExperimentContext:
    """
    Context manager for reproducible experiments.
    
    Ensures proper setup and teardown of experimental conditions,
    including seed management and resource monitoring.
    """
    
    def __init__(self, framework: ReproducibilityFramework, parameters: Dict[str, Any]):
        self.framework = framework
        self.parameters = parameters
        self.start_time = None
        self.end_time = None
        self.resource_monitor = ResourceMonitor() if PSUTIL_AVAILABLE else None
        
        # Update parameter hash
        param_string = json.dumps(parameters, sort_keys=True, default=str)
        self.framework.metadata.parameter_hash = hashlib.sha256(param_string.encode()).hexdigest()[:16]
    
    def __enter__(self) -> 'ExperimentContext':
        """Enter experiment context."""
        self.start_time = time.time()
        
        # Set deterministic seeds
        self.seeds = self.framework.set_deterministic_seeds()
        
        # Start resource monitoring
        if self.resource_monitor:
            self.resource_monitor.start_monitoring()
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit experiment context."""
        self.end_time = time.time()
        
        # Stop resource monitoring
        if self.resource_monitor:
            self.resource_monitor.stop_monitoring()
    
    def record_result(self, 
                     metrics: Dict[str, float],
                     artifacts: List[str] = None) -> ExperimentResult:
        """
        Record experiment result with timing and resource information.
        
        Args:
            metrics: Dictionary of experimental metrics
            artifacts: List of artifact file paths
            
        Returns:
            ExperimentResult object
        """
        if artifacts is None:
            artifacts = []
            
        # Timing information
        timing_info = {
            'duration_seconds': self.end_time - self.start_time if self.end_time else None,
            'start_time': self.start_time,
            'end_time': self.end_time
        }
        
        # Resource usage
        resource_usage = {}
        if self.resource_monitor:
            resource_usage = self.resource_monitor.get_peak_usage()
        
        # Create result
        result = ExperimentResult(
            experiment_id=self.framework.experiment_id,
            metrics=metrics,
            timing_info=timing_info,
            resource_usage=resource_usage,
            validation_status='pending',
            reproducibility_score=0.0,
            artifacts=artifacts
        )
        
        # Add to framework history
        self.framework.results_history.append(result)
        
        return result

class ResourceMonitor:
    """Monitor system resource usage during experiments."""
    
    def __init__(self):
        self.monitoring = False
        self.start_memory = None
        self.peak_memory = 0
        self.start_cpu_times = None
        self.cpu_samples = []
        
    def start_monitoring(self):
        """Start resource monitoring."""
        if not PSUTIL_AVAILABLE:
            return
            
        self.monitoring = True
        process = psutil.Process()
        self.start_memory = process.memory_info().rss
        self.peak_memory = self.start_memory
        self.start_cpu_times = process.cpu_times()
        
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring = False
        
    def get_peak_usage(self) -> Dict[str, float]:
        """Get peak resource usage statistics."""
        if not PSUTIL_AVAILABLE:
            return {}
            
        try:
            process = psutil.Process()
            current_memory = process.memory_info().rss
            end_cpu_times = process.cpu_times()
            
            memory_usage_mb = current_memory / (1024 * 1024)
            peak_memory_mb = self.peak_memory / (1024 * 1024)
            
            cpu_usage = 0.0
            if self.start_cpu_times:
                cpu_usage = ((end_cpu_times.user - self.start_cpu_times.user) + 
                           (end_cpu_times.system - self.start_cpu_times.system))
            
            return {
                'memory_usage_mb': memory_usage_mb,
                'peak_memory_mb': peak_memory_mb,
                'cpu_time_seconds': cpu_usage,
                'memory_delta_mb': (current_memory - self.start_memory) / (1024 * 1024)
            }
        except psutil.NoSuchProcess:
            return {}

class ExperimentTracker:
    """
    High-level experiment tracking and management.
    
    Provides utilities for managing multiple experiments, comparing results,
    and generating reproducibility reports.
    """
    
    def __init__(self, project_name: str):
        self.project_name = project_name
        self.experiments: Dict[str, ReproducibilityFramework] = {}
        self.results_database: List[ExperimentResult] = []
        
    def create_experiment(self, experiment_name: str, base_seed: int = 42) -> ReproducibilityFramework:
        """Create and register new experiment."""
        framework = ReproducibilityFramework(experiment_name, base_seed)
        self.experiments[framework.experiment_id] = framework
        return framework
    
    def get_experiment(self, experiment_id: str) -> Optional[ReproducibilityFramework]:
        """Get experiment by ID."""
        return self.experiments.get(experiment_id)
    
    def add_result(self, result: ExperimentResult):
        """Add result to database."""
        self.results_database.append(result)
    
    def compare_experiments(self, 
                          experiment_ids: List[str],
                          metric_names: List[str] = None) -> Dict[str, Any]:
        """
        Compare results across multiple experiments.
        
        Args:
            experiment_ids: List of experiment IDs to compare
            metric_names: List of metric names to compare (all if None)
            
        Returns:
            Comparison report
        """
        # Get results for specified experiments
        experiment_results = {}
        for exp_id in experiment_ids:
            results = [r for r in self.results_database if r.experiment_id == exp_id]
            if results:
                experiment_results[exp_id] = results
        
        if not experiment_results:
            return {'status': 'no_results', 'message': 'No results found for specified experiments'}
        
        # If no metric names specified, use all available metrics
        if metric_names is None:
            all_metrics = set()
            for results in experiment_results.values():
                for result in results:
                    all_metrics.update(result.metrics.keys())
            metric_names = list(all_metrics)
        
        # Compare metrics
        comparisons = {}
        for metric in metric_names:
            metric_data = {}
            for exp_id, results in experiment_results.items():
                values = [r.metrics.get(metric, float('nan')) for r in results]
                values_array = np.array([v for v in values if not np.isnan(v)])
                
                if len(values_array) > 0:
                    metric_data[exp_id] = {
                        'mean': np.mean(values_array),
                        'std': np.std(values_array),
                        'median': np.median(values_array),
                        'min': np.min(values_array),
                        'max': np.max(values_array),
                        'count': len(values_array)
                    }
            
            comparisons[metric] = metric_data
        
        return {
            'status': 'success',
            'comparisons': comparisons,
            'experiment_count': len(experiment_results),
            'metric_count': len(metric_names)
        }
    
    def generate_reproducibility_report(self) -> Dict[str, Any]:
        """Generate comprehensive reproducibility report for all experiments."""
        report = {
            'project_name': self.project_name,
            'timestamp': datetime.now().isoformat(),
            'total_experiments': len(self.experiments),
            'total_results': len(self.results_database),
            'experiment_summaries': {},
            'overall_reproducibility': 0.0
        }
        
        # Analyze each experiment
        experiment_scores = []
        for exp_id, framework in self.experiments.items():
            results = [r for r in self.results_database if r.experiment_id == exp_id]
            
            if len(results) >= 2:
                validation = framework.validate_reproducibility(results)
                report['experiment_summaries'][exp_id] = {
                    'name': framework.experiment_name,
                    'result_count': len(results),
                    'reproducibility_score': validation['reproducibility_score'],
                    'status': validation['status']
                }
                experiment_scores.append(validation['reproducibility_score'])
            else:
                report['experiment_summaries'][exp_id] = {
                    'name': framework.experiment_name,
                    'result_count': len(results),
                    'reproducibility_score': 0.0,
                    'status': 'insufficient_data'
                }
        
        # Calculate overall reproducibility
        if experiment_scores:
            report['overall_reproducibility'] = np.mean(experiment_scores)
        
        return report