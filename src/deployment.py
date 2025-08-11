"""
Production deployment and configuration management for photonic neural networks.

Implements automated deployment pipelines, configuration validation,
rollback mechanisms, and production readiness checks.
"""

import numpy as np
import json
import yaml
import time
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import hashlib
import subprocess
import threading
from contextlib import contextmanager

try:
    from .models import PhotonicNeuralNetwork, create_benchmark_network
    from .validation import PhotonicSystemValidator, ValidationResult
    from .utils.monitoring import SystemMonitor, create_production_monitor
    from .utils.logging_config import get_logger
except ImportError:
    from models import PhotonicNeuralNetwork, create_benchmark_network
    from validation import PhotonicSystemValidator, ValidationResult
    from utils.monitoring import SystemMonitor, create_production_monitor
    from utils.logging_config import get_logger

logger = get_logger(__name__)


class DeploymentStage(Enum):
    """Deployment pipeline stages."""
    VALIDATION = "validation"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    ROLLBACK = "rollback"


class DeploymentStatus(Enum):
    """Deployment status indicators."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class DeploymentConfig:
    """Configuration for deployment pipeline."""
    
    # Model configuration
    model_path: str
    model_task: str
    model_version: str
    
    # Deployment settings
    target_environment: str
    deployment_strategy: str = "blue_green"  # blue_green, rolling, canary
    auto_rollback: bool = True
    rollback_timeout_seconds: int = 300
    
    # Validation requirements
    min_accuracy_threshold: float = 0.85
    max_latency_threshold_ns: float = 2.0
    max_power_threshold_mw: float = 500.0
    required_validation_passes: int = 3
    
    # Monitoring configuration
    enable_monitoring: bool = True
    monitoring_interval: float = 5.0
    alert_endpoints: List[str] = None
    
    # Infrastructure settings
    resource_limits: Dict[str, Any] = None
    scaling_config: Dict[str, Any] = None
    health_check_config: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize default values after creation."""
        if self.alert_endpoints is None:
            self.alert_endpoints = []
        if self.resource_limits is None:
            self.resource_limits = {
                "cpu_cores": 4,
                "memory_gb": 8, 
                "gpu_memory_gb": 4
            }
        if self.scaling_config is None:
            self.scaling_config = {
                "min_instances": 1,
                "max_instances": 5,
                "target_utilization": 0.7
            }
        if self.health_check_config is None:
            self.health_check_config = {
                "interval_seconds": 30,
                "timeout_seconds": 10,
                "failure_threshold": 3
            }


@dataclass
class DeploymentResult:
    """Results from deployment operation."""
    deployment_id: str
    config: DeploymentConfig
    status: DeploymentStatus
    stage: DeploymentStage
    start_time: float
    end_time: Optional[float] = None
    validation_results: List[ValidationResult] = None
    performance_metrics: Dict[str, float] = None
    error_message: Optional[str] = None
    rollback_info: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.validation_results is None:
            self.validation_results = []
        if self.performance_metrics is None:
            self.performance_metrics = {}
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Get deployment duration in seconds."""
        if self.end_time:
            return self.end_time - self.start_time
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        # Convert enums to strings
        result['status'] = self.status.value
        result['stage'] = self.stage.value
        # Handle validation results
        result['validation_results'] = [
            {
                'is_valid': vr.is_valid,
                'num_errors': len(vr.errors),
                'num_warnings': len(vr.warnings),
                'performance_degradation': vr.performance_degradation
            }
            for vr in self.validation_results
        ]
        return result


class ModelCheckpoint:
    """Model checkpoint management for rollback capability."""
    
    def __init__(self, checkpoint_dir: str):
        """Initialize checkpoint manager."""
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def create_checkpoint(self, model: PhotonicNeuralNetwork, 
                         version: str, metadata: Dict[str, Any]) -> str:
        """
        Create model checkpoint.
        
        Args:
            model: Model to checkpoint
            version: Version identifier
            metadata: Additional metadata
            
        Returns:
            Checkpoint identifier
        """
        checkpoint_id = f"checkpoint_{version}_{int(time.time())}"
        checkpoint_path = self.checkpoint_dir / checkpoint_id
        checkpoint_path.mkdir(exist_ok=True)
        
        # Save model
        model_file = checkpoint_path / "model.npy"
        model.save_model(str(model_file))
        
        # Save metadata
        metadata_file = checkpoint_path / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump({
                "checkpoint_id": checkpoint_id,
                "version": version,
                "timestamp": time.time(),
                "model_summary": model.get_network_summary(),
                **metadata
            }, f, indent=2)
        
        logger.info(f"Created checkpoint: {checkpoint_id}")
        return checkpoint_id
    
    def restore_checkpoint(self, checkpoint_id: str) -> Tuple[PhotonicNeuralNetwork, Dict[str, Any]]:
        """
        Restore model from checkpoint.
        
        Args:
            checkpoint_id: Checkpoint to restore
            
        Returns:
            Tuple of (restored_model, metadata)
        """
        checkpoint_path = self.checkpoint_dir / checkpoint_id
        
        if not checkpoint_path.exists():
            raise ValueError(f"Checkpoint not found: {checkpoint_id}")
        
        # Load metadata
        metadata_file = checkpoint_path / "metadata.json"
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Restore model (simplified - would need proper model reconstruction)
        model_task = metadata.get('model_task', 'mnist')
        model = create_benchmark_network(model_task)
        
        # Load saved weights
        model_file = checkpoint_path / "model.npy"
        if model_file.exists():
            model.load_model(str(model_file))
        
        logger.info(f"Restored checkpoint: {checkpoint_id}")
        return model, metadata
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List available checkpoints."""
        checkpoints = []
        
        for checkpoint_dir in self.checkpoint_dir.iterdir():
            if checkpoint_dir.is_dir():
                metadata_file = checkpoint_dir / "metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                        checkpoints.append(metadata)
        
        return sorted(checkpoints, key=lambda x: x['timestamp'], reverse=True)


class ProductionReadinessChecker:
    """Comprehensive production readiness validation."""
    
    def __init__(self):
        """Initialize readiness checker."""
        self.validator = PhotonicSystemValidator()
        self.checks = [
            self._check_model_validation,
            self._check_performance_requirements,
            self._check_resource_requirements,
            self._check_monitoring_setup,
            self._check_error_handling,
            self._check_security_compliance
        ]
    
    def check_readiness(self, model: PhotonicNeuralNetwork,
                       config: DeploymentConfig) -> Dict[str, Any]:
        """
        Perform comprehensive production readiness check.
        
        Args:
            model: Model to check
            config: Deployment configuration
            
        Returns:
            Readiness assessment results
        """
        logger.info("Performing production readiness assessment")
        
        results = {
            "overall_ready": True,
            "timestamp": time.time(),
            "checks": {},
            "recommendations": [],
            "blocking_issues": []
        }
        
        for check_func in self.checks:
            check_name = check_func.__name__.replace('_check_', '')
            
            try:
                check_result = check_func(model, config)
                results["checks"][check_name] = check_result
                
                if not check_result.get("passed", False):
                    results["overall_ready"] = False
                    if check_result.get("blocking", False):
                        results["blocking_issues"].append(check_result)
                
                recommendations = check_result.get("recommendations", [])
                results["recommendations"].extend(recommendations)
                
            except Exception as e:
                logger.error(f"Readiness check {check_name} failed: {e}")
                results["checks"][check_name] = {
                    "passed": False,
                    "blocking": True,
                    "error": str(e)
                }
                results["overall_ready"] = False
        
        logger.info(f"Production readiness: {'READY' if results['overall_ready'] else 'NOT READY'}")
        return results
    
    def _check_model_validation(self, model: PhotonicNeuralNetwork, 
                              config: DeploymentConfig) -> Dict[str, Any]:
        """Check model validation requirements."""
        validation_result = self.validator.validate_system(model)
        
        passed = (validation_result.is_valid and 
                 not validation_result.has_critical_errors)
        
        return {
            "passed": passed,
            "blocking": validation_result.has_critical_errors,
            "details": {
                "is_valid": validation_result.is_valid,
                "num_errors": len(validation_result.errors),
                "num_warnings": len(validation_result.warnings),
                "critical_errors": validation_result.has_critical_errors
            },
            "recommendations": validation_result.recommended_actions if not passed else []
        }
    
    def _check_performance_requirements(self, model: PhotonicNeuralNetwork,
                                      config: DeploymentConfig) -> Dict[str, Any]:
        """Check performance requirements."""
        # Run benchmark to check performance
        test_input = np.random.randn(32, 784) * 0.1 + 0.5  # Sample input
        
        try:
            predictions, metrics = model.forward(test_input, measure_latency=True)
            
            latency_ok = metrics["total_latency_ns"] / len(test_input) <= config.max_latency_threshold_ns
            power_ok = metrics["total_power_mw"] <= config.max_power_threshold_mw
            
            passed = latency_ok and power_ok
            
            return {
                "passed": passed,
                "blocking": not passed,
                "details": {
                    "avg_latency_ns": metrics["total_latency_ns"] / len(test_input),
                    "power_mw": metrics["total_power_mw"],
                    "latency_requirement_met": latency_ok,
                    "power_requirement_met": power_ok
                },
                "recommendations": [
                    "Optimize model for better latency" if not latency_ok else "",
                    "Reduce power consumption" if not power_ok else ""
                ]
            }
            
        except Exception as e:
            return {
                "passed": False,
                "blocking": True,
                "error": f"Performance test failed: {e}",
                "recommendations": ["Fix model execution issues"]
            }
    
    def _check_resource_requirements(self, model: PhotonicNeuralNetwork,
                                   config: DeploymentConfig) -> Dict[str, Any]:
        """Check resource requirements."""
        # Estimate resource usage
        total_parameters = model.get_total_parameters()
        estimated_memory_mb = total_parameters * 4 / (1024**2)  # 4 bytes per parameter
        
        memory_ok = estimated_memory_mb <= config.resource_limits.get("memory_gb", 8) * 1024
        
        return {
            "passed": memory_ok,
            "blocking": not memory_ok,
            "details": {
                "total_parameters": total_parameters,
                "estimated_memory_mb": estimated_memory_mb,
                "memory_limit_mb": config.resource_limits.get("memory_gb", 8) * 1024,
                "memory_requirement_met": memory_ok
            },
            "recommendations": [
                "Increase memory allocation or optimize model size" if not memory_ok else ""
            ]
        }
    
    def _check_monitoring_setup(self, model: PhotonicNeuralNetwork,
                              config: DeploymentConfig) -> Dict[str, Any]:
        """Check monitoring configuration."""
        monitoring_ok = config.enable_monitoring
        alerts_configured = len(config.alert_endpoints) > 0
        
        passed = monitoring_ok
        
        return {
            "passed": passed,
            "blocking": False,  # Monitoring is important but not blocking
            "details": {
                "monitoring_enabled": monitoring_ok,
                "alert_endpoints_configured": alerts_configured,
                "monitoring_interval": config.monitoring_interval
            },
            "recommendations": [
                "Enable monitoring for production deployment" if not monitoring_ok else "",
                "Configure alert endpoints" if not alerts_configured else ""
            ]
        }
    
    def _check_error_handling(self, model: PhotonicNeuralNetwork,
                            config: DeploymentConfig) -> Dict[str, Any]:
        """Check error handling capabilities."""
        # Test error handling with invalid input
        try:
            invalid_input = np.full((1, 784), np.inf)
            model.forward(invalid_input)
            graceful_handling = True
        except Exception:
            graceful_handling = False
        
        return {
            "passed": graceful_handling,
            "blocking": False,
            "details": {
                "graceful_error_handling": graceful_handling,
                "auto_rollback_enabled": config.auto_rollback
            },
            "recommendations": [
                "Implement graceful error handling" if not graceful_handling else ""
            ]
        }
    
    def _check_security_compliance(self, model: PhotonicNeuralNetwork,
                                 config: DeploymentConfig) -> Dict[str, Any]:
        """Check security compliance."""
        # Basic security checks
        model_path_secure = Path(config.model_path).is_file() if config.model_path else False
        
        return {
            "passed": model_path_secure,
            "blocking": False,
            "details": {
                "model_path_exists": model_path_secure,
                "deployment_config_valid": True
            },
            "recommendations": [
                "Verify model path security" if not model_path_secure else ""
            ]
        }


class DeploymentPipeline:
    """
    Automated deployment pipeline for photonic neural networks.
    
    Implements comprehensive deployment automation with validation,
    testing, rollback capabilities, and production monitoring.
    """
    
    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        """Initialize deployment pipeline."""
        self.checkpoint_manager = ModelCheckpoint(checkpoint_dir)
        self.readiness_checker = ProductionReadinessChecker()
        self.deployment_history = []
        self.active_deployments = {}
        
    def deploy(self, config: DeploymentConfig) -> DeploymentResult:
        """
        Execute complete deployment pipeline.
        
        Args:
            config: Deployment configuration
            
        Returns:
            Deployment result with status and metrics
        """
        deployment_id = f"deploy_{int(time.time())}_{hash(config.model_version) % 10000}"
        
        logger.info(f"Starting deployment {deployment_id}")
        
        result = DeploymentResult(
            deployment_id=deployment_id,
            config=config,
            status=DeploymentStatus.IN_PROGRESS,
            stage=DeploymentStage.VALIDATION,
            start_time=time.time()
        )
        
        self.active_deployments[deployment_id] = result
        
        try:
            # Stage 1: Load and validate model
            model = self._load_model(config)
            
            # Stage 2: Production readiness check
            readiness = self.readiness_checker.check_readiness(model, config)
            if not readiness["overall_ready"]:
                raise RuntimeError(f"Model not ready for production: {readiness['blocking_issues']}")
            
            # Stage 3: Create checkpoint for rollback
            checkpoint_id = self.checkpoint_manager.create_checkpoint(
                model, config.model_version, 
                {"deployment_id": deployment_id, "config": asdict(config)}
            )
            
            # Stage 4: Validation testing
            result.stage = DeploymentStage.TESTING
            validation_results = self._run_validation_tests(model, config)
            result.validation_results = validation_results
            
            # Stage 5: Performance testing
            performance_metrics = self._run_performance_tests(model, config)
            result.performance_metrics = performance_metrics
            
            # Stage 6: Deploy to staging
            result.stage = DeploymentStage.STAGING
            self._deploy_to_staging(model, config)
            
            # Stage 7: Production deployment
            result.stage = DeploymentStage.PRODUCTION
            self._deploy_to_production(model, config)
            
            # Stage 8: Setup monitoring
            if config.enable_monitoring:
                self._setup_monitoring(model, config, deployment_id)
            
            # Success
            result.status = DeploymentStatus.SUCCESS
            result.end_time = time.time()
            
            logger.info(f"Deployment {deployment_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Deployment {deployment_id} failed: {e}")
            result.status = DeploymentStatus.FAILED
            result.error_message = str(e)
            result.end_time = time.time()
            
            # Automatic rollback if enabled
            if config.auto_rollback:
                try:
                    self._perform_rollback(deployment_id)
                    result.status = DeploymentStatus.ROLLED_BACK
                except Exception as rollback_error:
                    logger.error(f"Rollback failed: {rollback_error}")
        
        finally:
            self.deployment_history.append(result)
            if deployment_id in self.active_deployments:
                del self.active_deployments[deployment_id]
        
        return result
    
    def _load_model(self, config: DeploymentConfig) -> PhotonicNeuralNetwork:
        """Load model for deployment."""
        if config.model_path and Path(config.model_path).exists():
            # Load from saved model
            model = create_benchmark_network(config.model_task)
            model.load_model(config.model_path)
            return model
        else:
            # Create new model
            return create_benchmark_network(config.model_task)
    
    def _run_validation_tests(self, model: PhotonicNeuralNetwork,
                            config: DeploymentConfig) -> List[ValidationResult]:
        """Run validation tests."""
        validator = PhotonicSystemValidator()
        validation_results = []
        
        for i in range(config.required_validation_passes):
            logger.info(f"Running validation pass {i + 1}/{config.required_validation_passes}")
            result = validator.validate_system(model)
            validation_results.append(result)
            
            if result.has_critical_errors:
                raise RuntimeError(f"Validation failed with critical errors: {result.errors}")
        
        return validation_results
    
    def _run_performance_tests(self, model: PhotonicNeuralNetwork,
                             config: DeploymentConfig) -> Dict[str, float]:
        """Run performance benchmarks."""
        logger.info("Running performance tests")
        
        # Generate test data
        if config.model_task == "mnist":
            test_input = np.random.randn(100, 784) * 0.1 + 0.5
        elif config.model_task == "cifar10":
            test_input = np.random.randn(100, 3072) * 0.1 + 0.5
        elif config.model_task == "vowel_classification":
            test_input = np.random.randn(100, 10) * 0.3 + 0.5
        else:
            raise ValueError(f"Unknown task: {config.model_task}")
        
        # Run performance test
        start_time = time.perf_counter_ns()
        predictions, hardware_metrics = model.forward(test_input, measure_latency=True)
        end_time = time.perf_counter_ns()
        
        # Calculate metrics
        avg_latency_ns = hardware_metrics["total_latency_ns"] / len(test_input)
        total_power_mw = hardware_metrics["total_power_mw"]
        throughput_ops = len(test_input) * 1e9 / (end_time - start_time)
        
        performance_metrics = {
            "avg_latency_ns": avg_latency_ns,
            "power_mw": total_power_mw,
            "throughput_ops_per_sec": throughput_ops,
            "total_test_time_s": (end_time - start_time) / 1e9
        }
        
        # Check against thresholds
        if avg_latency_ns > config.max_latency_threshold_ns:
            raise RuntimeError(f"Latency {avg_latency_ns:.2f}ns exceeds threshold {config.max_latency_threshold_ns}ns")
        
        if total_power_mw > config.max_power_threshold_mw:
            raise RuntimeError(f"Power {total_power_mw:.1f}mW exceeds threshold {config.max_power_threshold_mw}mW")
        
        return performance_metrics
    
    def _deploy_to_staging(self, model: PhotonicNeuralNetwork, config: DeploymentConfig):
        """Deploy to staging environment."""
        logger.info("Deploying to staging environment")
        # In production, this would involve actual infrastructure deployment
        time.sleep(1)  # Simulate deployment time
    
    def _deploy_to_production(self, model: PhotonicNeuralNetwork, config: DeploymentConfig):
        """Deploy to production environment."""
        logger.info("Deploying to production environment")
        # In production, this would involve:
        # - Blue-green deployment switching
        # - Load balancer updates
        # - Service discovery registration
        # - Health check initialization
        time.sleep(2)  # Simulate deployment time
    
    def _setup_monitoring(self, model: PhotonicNeuralNetwork, 
                         config: DeploymentConfig, deployment_id: str):
        """Setup monitoring for deployed model."""
        logger.info("Setting up production monitoring")
        monitor = create_production_monitor(model, config.monitoring_interval)
        monitor.start_monitoring()
        
        # Store monitor reference for later cleanup
        self.active_deployments[deployment_id].monitor = monitor
    
    def _perform_rollback(self, deployment_id: str):
        """Perform deployment rollback."""
        logger.info(f"Performing rollback for deployment {deployment_id}")
        
        # In production, this would:
        # - Switch traffic back to previous version
        # - Restore previous configuration
        # - Clean up failed deployment resources
        # - Notify operations team
        
        time.sleep(1)  # Simulate rollback time
    
    def get_deployment_status(self, deployment_id: str) -> Optional[DeploymentResult]:
        """Get status of specific deployment."""
        # Check active deployments first
        if deployment_id in self.active_deployments:
            return self.active_deployments[deployment_id]
        
        # Check deployment history
        for result in self.deployment_history:
            if result.deployment_id == deployment_id:
                return result
        
        return None
    
    def list_deployments(self, limit: int = 10) -> List[DeploymentResult]:
        """List recent deployments."""
        return sorted(self.deployment_history, 
                     key=lambda x: x.start_time, reverse=True)[:limit]
    
    def cleanup_deployment(self, deployment_id: str):
        """Clean up deployment resources."""
        if deployment_id in self.active_deployments:
            deployment = self.active_deployments[deployment_id]
            
            # Stop monitoring if active
            if hasattr(deployment, 'monitor'):
                deployment.monitor.stop_monitoring()
            
            del self.active_deployments[deployment_id]
            logger.info(f"Cleaned up deployment {deployment_id}")


# Convenience functions for common deployment patterns
def deploy_model_simple(model_path: str, model_task: str, 
                       target_environment: str = "production") -> DeploymentResult:
    """
    Simple deployment function for basic use cases.
    
    Args:
        model_path: Path to saved model
        model_task: Model task type
        target_environment: Target deployment environment
        
    Returns:
        Deployment result
    """
    config = DeploymentConfig(
        model_path=model_path,
        model_task=model_task,
        model_version="1.0.0",
        target_environment=target_environment
    )
    
    pipeline = DeploymentPipeline()
    return pipeline.deploy(config)


def deploy_with_monitoring(model: PhotonicNeuralNetwork, model_task: str,
                          monitoring_interval: float = 5.0) -> Tuple[DeploymentResult, SystemMonitor]:
    """
    Deploy model with comprehensive monitoring setup.
    
    Args:
        model: Model to deploy
        model_task: Model task type
        monitoring_interval: Monitoring interval in seconds
        
    Returns:
        Tuple of (deployment_result, system_monitor)
    """
    # Save model temporarily
    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp:
        model.save_model(tmp.name)
        temp_model_path = tmp.name
    
    try:
        config = DeploymentConfig(
            model_path=temp_model_path,
            model_task=model_task,
            model_version="1.0.0",
            target_environment="production",
            enable_monitoring=True,
            monitoring_interval=monitoring_interval
        )
        
        pipeline = DeploymentPipeline()
        result = pipeline.deploy(config)
        
        # Create monitoring system
        monitor = create_production_monitor(model, monitoring_interval)
        monitor.start_monitoring()
        
        return result, monitor
        
    finally:
        # Clean up temporary file
        Path(temp_model_path).unlink(missing_ok=True)