"""
Robust validation and error handling for photonic neural networks.

Implements comprehensive error detection, hardware failure simulation,
and graceful degradation strategies for production photonic systems.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from dataclasses import dataclass
from enum import Enum
import warnings
from functools import wraps

try:
    from .core import PhotonicProcessor, WavelengthConfig, ThermalConfig, FabricationConfig
    from .models import PhotonicNeuralNetwork
except ImportError:
    from core import PhotonicProcessor, WavelengthConfig, ThermalConfig, FabricationConfig
    from models import PhotonicNeuralNetwork


logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """Types of errors in photonic systems."""
    THERMAL_DRIFT = "thermal_drift"
    FABRICATION_DEFECT = "fabrication_defect" 
    OPTICAL_LOSS = "optical_loss"
    PHASE_SHIFTER_FAILURE = "phase_shifter_failure"
    WAVELENGTH_INSTABILITY = "wavelength_instability"
    CROSSTALK = "crosstalk"
    NONLINEAR_DISTORTION = "nonlinear_distortion"


class SeverityLevel(Enum):
    """Severity levels for system issues."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high" 
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Results from system validation."""
    is_valid: bool
    errors: List[Dict[str, Any]]
    warnings: List[Dict[str, Any]]
    performance_degradation: float
    recommended_actions: List[str]
    
    @property
    def has_critical_errors(self) -> bool:
        """Check if any critical errors exist."""
        return any(error["severity"] == SeverityLevel.CRITICAL for error in self.errors)


class PhotonicSystemValidator:
    """
    Comprehensive validation system for photonic neural networks.
    
    Implements robust error detection and health monitoring based on
    real-world hardware constraints and failure modes.
    """
    
    def __init__(self, 
                 tolerance_config: Optional[Dict[str, float]] = None,
                 enable_monitoring: bool = True):
        """
        Initialize system validator.
        
        Args:
            tolerance_config: Custom tolerance thresholds for validation
            enable_monitoring: Enable continuous system monitoring
        """
        self.tolerance_config = tolerance_config or self._default_tolerances()
        self.enable_monitoring = enable_monitoring
        self.validation_history = []
        self.error_counts = {error_type: 0 for error_type in ErrorType}
        
    def _default_tolerances(self) -> Dict[str, float]:
        """Default tolerance values based on demonstrated hardware."""
        return {
            "max_thermal_drift_pm": 20.0,      # ±20pm acceptable drift
            "min_coupling_efficiency": 0.7,    # 70% minimum coupling
            "max_propagation_loss_db": 1.0,    # 1dB/cm maximum loss
            "min_extinction_ratio_db": 10.0,   # 10dB minimum extinction
            "max_crosstalk_db": -20.0,         # -20dB maximum crosstalk
            "min_phase_shifter_range_pi": 2.0, # 2π minimum phase range
            "max_power_variation_db": 0.5,     # ±0.5dB power stability
            "min_accuracy_threshold": 0.8,     # 80% minimum accuracy
        }
    
    def validate_system(self, model: PhotonicNeuralNetwork) -> ValidationResult:
        """
        Perform comprehensive system validation.
        
        Args:
            model: Photonic neural network to validate
            
        Returns:
            Detailed validation results with errors and recommendations
        """
        logger.info("Starting comprehensive system validation")
        
        errors = []
        warnings = []
        performance_degradation = 0.0
        
        # Validate each layer
        for layer_idx, layer in enumerate(model.layers):
            layer_validation = self._validate_layer(layer, layer_idx)
            errors.extend(layer_validation["errors"])
            warnings.extend(layer_validation["warnings"])
            performance_degradation = max(performance_degradation, 
                                        layer_validation["degradation"])
        
        # System-level validations
        system_validation = self._validate_system_level(model)
        errors.extend(system_validation["errors"])
        warnings.extend(system_validation["warnings"])
        
        # Generate recommendations
        recommendations = self._generate_recommendations(errors, warnings)
        
        # Determine overall validity
        is_valid = not any(error["severity"] == SeverityLevel.CRITICAL for error in errors)
        
        result = ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            performance_degradation=performance_degradation,
            recommended_actions=recommendations
        )
        
        # Store validation history
        self.validation_history.append(result)
        
        logger.info(f"Validation completed: {len(errors)} errors, {len(warnings)} warnings")
        
        return result
    
    def _validate_layer(self, layer, layer_idx: int) -> Dict[str, Any]:
        """Validate individual layer for hardware constraints."""
        errors = []
        warnings = []
        degradation = 0.0
        
        # Check thermal stability
        thermal_validation = self._check_thermal_stability(layer, layer_idx)
        errors.extend(thermal_validation["errors"])
        warnings.extend(thermal_validation["warnings"])
        degradation = max(degradation, thermal_validation["degradation"])
        
        # Check fabrication constraints
        fab_validation = self._check_fabrication_constraints(layer, layer_idx)
        errors.extend(fab_validation["errors"])
        warnings.extend(fab_validation["warnings"])
        
        # Check optical power levels
        power_validation = self._check_optical_power_levels(layer, layer_idx)
        errors.extend(power_validation["errors"])
        warnings.extend(power_validation["warnings"])
        
        # Check phase shifter functionality
        phase_validation = self._check_phase_shifters(layer, layer_idx)
        errors.extend(phase_validation["errors"])
        warnings.extend(phase_validation["warnings"])
        
        return {
            "errors": errors,
            "warnings": warnings,
            "degradation": degradation
        }
    
    def _check_thermal_stability(self, layer, layer_idx: int) -> Dict[str, Any]:
        """Check thermal stability and compensation effectiveness."""
        errors = []
        warnings = []
        degradation = 0.0
        
        processor = layer.processor
        current_temp = processor.current_temperature
        base_temp = processor.thermal_config.operating_temperature
        
        # Check temperature deviation
        temp_deviation = abs(current_temp - base_temp)
        max_temp_var = processor.thermal_config.max_temperature_variation
        
        if temp_deviation > max_temp_var:
            severity = SeverityLevel.HIGH if temp_deviation > max_temp_var * 1.5 else SeverityLevel.MEDIUM
            errors.append({
                "type": ErrorType.THERMAL_DRIFT,
                "severity": severity,
                "layer": layer_idx,
                "message": f"Temperature deviation {temp_deviation:.2f}K exceeds limit {max_temp_var}K",
                "current_value": temp_deviation,
                "threshold": max_temp_var
            })
            
            # Calculate performance degradation
            degradation = min(temp_deviation / max_temp_var * 0.1, 0.3)  # Up to 30% degradation
        
        # Check thermal compensation power
        compensation_power = processor.power_consumption
        max_power = len(processor.phase_shifts) * processor.thermal_config.power_per_heater
        
        if compensation_power > max_power * 0.9:
            warnings.append({
                "type": ErrorType.THERMAL_DRIFT,
                "severity": SeverityLevel.MEDIUM,
                "layer": layer_idx,
                "message": f"Thermal compensation power {compensation_power:.1f}mW near limit {max_power:.1f}mW",
                "current_value": compensation_power,
                "threshold": max_power
            })
        
        return {"errors": errors, "warnings": warnings, "degradation": degradation}
    
    def _check_fabrication_constraints(self, layer, layer_idx: int) -> Dict[str, Any]:
        """Check fabrication-related constraints and tolerances."""
        errors = []
        warnings = []
        
        # Check weight quantization precision
        weights = layer.weights
        precision_bits = layer.config.weight_precision
        
        # Analyze quantization noise
        original_weights = weights
        quantized_weights = layer._quantize_weights(np.abs(weights), precision_bits)
        quantization_error = np.mean(np.abs(np.abs(original_weights) - quantized_weights))
        
        if quantization_error > 0.1:  # 10% quantization error threshold
            errors.append({
                "type": ErrorType.FABRICATION_DEFECT,
                "severity": SeverityLevel.MEDIUM,
                "layer": layer_idx,
                "message": f"High quantization error {quantization_error:.3f} with {precision_bits}-bit precision",
                "current_value": quantization_error,
                "threshold": 0.1
            })
        
        # Check for extreme weight values
        weight_magnitudes = np.abs(weights)
        max_magnitude = np.max(weight_magnitudes)
        
        if max_magnitude > 2.0:  # Reasonable upper bound for photonic systems
            warnings.append({
                "type": ErrorType.FABRICATION_DEFECT,
                "severity": SeverityLevel.LOW,
                "layer": layer_idx,
                "message": f"Large weight magnitude {max_magnitude:.3f} may be difficult to implement",
                "current_value": max_magnitude,
                "threshold": 2.0
            })
        
        return {"errors": errors, "warnings": warnings}
    
    def _check_optical_power_levels(self, layer, layer_idx: int) -> Dict[str, Any]:
        """Check optical power levels and signal integrity."""
        errors = []
        warnings = []
        
        # Check for power imbalances
        weights = layer.weights
        power_levels = np.abs(weights) ** 2
        
        # Check dynamic range
        max_power = np.max(power_levels)
        min_power = np.min(power_levels[power_levels > 0])
        
        if max_power / min_power > 1000:  # 30dB dynamic range limit
            warnings.append({
                "type": ErrorType.OPTICAL_LOSS,
                "severity": SeverityLevel.MEDIUM,
                "layer": layer_idx,
                "message": f"High dynamic range {max_power/min_power:.1f} may cause SNR issues",
                "current_value": max_power/min_power,
                "threshold": 1000
            })
        
        # Check for zero or near-zero weights
        near_zero_count = np.sum(np.abs(weights) < 1e-6)
        total_weights = weights.size
        
        if near_zero_count / total_weights > 0.5:  # More than 50% near-zero weights
            warnings.append({
                "type": ErrorType.OPTICAL_LOSS,
                "severity": SeverityLevel.LOW,
                "layer": layer_idx,
                "message": f"{near_zero_count}/{total_weights} weights near zero - potential underutilization",
                "current_value": near_zero_count / total_weights,
                "threshold": 0.5
            })
        
        return {"errors": errors, "warnings": warnings}
    
    def _check_phase_shifters(self, layer, layer_idx: int) -> Dict[str, Any]:
        """Check phase shifter functionality and range."""
        errors = []
        warnings = []
        
        # Check phase range utilization
        weights = layer.weights
        phases = np.angle(weights)
        
        phase_range = np.max(phases) - np.min(phases)
        max_range = 2 * np.pi
        
        if phase_range < np.pi:  # Under-utilizing phase range
            warnings.append({
                "type": ErrorType.PHASE_SHIFTER_FAILURE,
                "severity": SeverityLevel.LOW,
                "layer": layer_idx,
                "message": f"Phase range {phase_range:.2f} rad under-utilizing available {max_range:.2f} rad",
                "current_value": phase_range,
                "threshold": np.pi
            })
        
        # Check for phase discontinuities
        phase_diff = np.diff(np.sort(phases.flatten()))
        large_jumps = np.sum(phase_diff > np.pi)
        
        if large_jumps > len(phases.flatten()) * 0.1:  # More than 10% large phase jumps
            warnings.append({
                "type": ErrorType.PHASE_SHIFTER_FAILURE,
                "severity": SeverityLevel.LOW,
                "layer": layer_idx,
                "message": f"Many large phase jumps detected - may indicate control issues",
                "current_value": large_jumps,
                "threshold": len(phases.flatten()) * 0.1
            })
        
        return {"errors": errors, "warnings": warnings}
    
    def _validate_system_level(self, model: PhotonicNeuralNetwork) -> Dict[str, Any]:
        """Perform system-level validations."""
        errors = []
        warnings = []
        
        # Check wavelength channel utilization
        wavelength_validation = self._check_wavelength_utilization(model)
        errors.extend(wavelength_validation["errors"])
        warnings.extend(wavelength_validation["warnings"])
        
        # Check inter-layer compatibility
        compatibility_validation = self._check_layer_compatibility(model)
        errors.extend(compatibility_validation["errors"])
        warnings.extend(compatibility_validation["warnings"])
        
        # Check power budget
        power_validation = self._check_power_budget(model)
        errors.extend(power_validation["errors"])
        warnings.extend(power_validation["warnings"])
        
        return {"errors": errors, "warnings": warnings}
    
    def _check_wavelength_utilization(self, model: PhotonicNeuralNetwork) -> Dict[str, Any]:
        """Check wavelength-division multiplexing utilization."""
        errors = []
        warnings = []
        
        num_channels = model.wavelength_config.num_channels
        
        # Check if all layers use the same wavelength configuration
        for i, layer in enumerate(model.layers):
            layer_channels = layer.processor.wavelength_config.num_channels
            if layer_channels != num_channels:
                errors.append({
                    "type": ErrorType.WAVELENGTH_INSTABILITY,
                    "severity": SeverityLevel.HIGH,
                    "layer": i,
                    "message": f"Wavelength channel mismatch: layer has {layer_channels}, expected {num_channels}",
                    "current_value": layer_channels,
                    "threshold": num_channels
                })
        
        # Check for wavelength spacing issues
        wavelength_spacing = model.wavelength_config.wavelength_spacing
        if wavelength_spacing < 0.4:  # Minimum spacing for crosstalk avoidance
            warnings.append({
                "type": ErrorType.CROSSTALK,
                "severity": SeverityLevel.MEDIUM,
                "layer": -1,  # System-level
                "message": f"Wavelength spacing {wavelength_spacing}nm may cause crosstalk",
                "current_value": wavelength_spacing,
                "threshold": 0.4
            })
        
        return {"errors": errors, "warnings": warnings}
    
    def _check_layer_compatibility(self, model: PhotonicNeuralNetwork) -> Dict[str, Any]:
        """Check compatibility between adjacent layers."""
        errors = []
        warnings = []
        
        for i in range(len(model.layers) - 1):
            current_layer = model.layers[i]
            next_layer = model.layers[i + 1]
            
            # Check dimension compatibility
            if current_layer.config.output_dim != next_layer.config.input_dim:
                errors.append({
                    "type": ErrorType.FABRICATION_DEFECT,
                    "severity": SeverityLevel.CRITICAL,
                    "layer": i,
                    "message": f"Dimension mismatch: layer {i} output {current_layer.config.output_dim} "
                              f"!= layer {i+1} input {next_layer.config.input_dim}",
                    "current_value": current_layer.config.output_dim,
                    "threshold": next_layer.config.input_dim
                })
        
        return {"errors": errors, "warnings": warnings}
    
    def _check_power_budget(self, model: PhotonicNeuralNetwork) -> Dict[str, Any]:
        """Check total system power consumption."""
        errors = []
        warnings = []
        
        total_power = 0.0
        for layer in model.layers:
            layer_power = len(layer.processor.phase_shifts) * layer.processor.thermal_config.power_per_heater
            total_power += layer_power
        
        # Reasonable power budget for integrated systems
        max_system_power = 1000.0  # 1W total budget
        
        if total_power > max_system_power:
            errors.append({
                "type": ErrorType.THERMAL_DRIFT,
                "severity": SeverityLevel.HIGH,
                "layer": -1,  # System-level
                "message": f"Total power consumption {total_power:.1f}mW exceeds budget {max_system_power:.1f}mW",
                "current_value": total_power,
                "threshold": max_system_power
            })
        elif total_power > max_system_power * 0.8:
            warnings.append({
                "type": ErrorType.THERMAL_DRIFT,
                "severity": SeverityLevel.MEDIUM,
                "layer": -1,
                "message": f"Power consumption {total_power:.1f}mW approaching budget limit",
                "current_value": total_power,
                "threshold": max_system_power * 0.8
            })
        
        return {"errors": errors, "warnings": warnings}
    
    def _generate_recommendations(self, errors: List[Dict], warnings: List[Dict]) -> List[str]:
        """Generate actionable recommendations based on validation results."""
        recommendations = []
        
        # Analyze error patterns
        error_types = [error["type"] for error in errors]
        warning_types = [warning["type"] for warning in warnings]
        
        # Thermal-related recommendations
        if ErrorType.THERMAL_DRIFT in error_types:
            recommendations.append("Increase thermal compensation power or improve cooling system")
            recommendations.append("Consider reducing operating temperature or improving thermal isolation")
        
        # Fabrication-related recommendations
        if ErrorType.FABRICATION_DEFECT in error_types:
            recommendations.append("Increase weight precision bits or implement post-fabrication trimming")
            recommendations.append("Consider process optimization for tighter fabrication tolerances")
        
        # Power-related recommendations
        power_errors = [e for e in errors if "power" in e["message"].lower()]
        if power_errors:
            recommendations.append("Optimize weight sparsity or implement power gating")
            recommendations.append("Consider voltage scaling or more efficient phase shifter designs")
        
        # Wavelength-related recommendations
        if ErrorType.WAVELENGTH_INSTABILITY in error_types:
            recommendations.append("Verify wavelength reference stability and calibration")
            recommendations.append("Implement active wavelength locking system")
        
        # General recommendations based on severity
        critical_errors = [e for e in errors if e["severity"] == SeverityLevel.CRITICAL]
        if critical_errors:
            recommendations.append("URGENT: Address critical errors before deployment")
        
        if len(warnings) > 10:
            recommendations.append("Consider design optimization to reduce warning count")
        
        return recommendations


def robust_execution(retry_attempts: int = 3, fallback_strategy: str = "degrade"):
    """
    Decorator for robust execution with error handling and fallback strategies.
    
    Args:
        retry_attempts: Number of retry attempts on failure
        fallback_strategy: Strategy on persistent failure ("degrade", "fail", "bypass")
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(retry_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}")
                    
                    if attempt < retry_attempts - 1:
                        # Implement backoff strategy
                        import time
                        time.sleep(0.1 * (2 ** attempt))  # Exponential backoff
            
            # All attempts failed - implement fallback strategy
            if fallback_strategy == "degrade":
                logger.error(f"Degraded mode activated for {func.__name__}")
                return _degraded_execution(func, args, kwargs)
            elif fallback_strategy == "bypass":
                logger.error(f"Bypassing {func.__name__} due to persistent failures")
                return None
            else:  # "fail"
                raise last_exception
        
        return wrapper
    return decorator


def _degraded_execution(func, args, kwargs):
    """Execute function in degraded mode with reduced precision/functionality."""
    # This is a simplified fallback - in practice, would implement
    # specific degradation strategies per function
    try:
        # Try with reduced precision or simplified computation
        if hasattr(args[0], 'enable_noise'):
            args[0].enable_noise = False  # Disable noise simulation
        
        return func(*args, **kwargs)
    except Exception as e:
        logger.error(f"Degraded execution also failed: {e}")
        return None


class HealthMonitor:
    """Continuous health monitoring for photonic neural networks."""
    
    def __init__(self, model: PhotonicNeuralNetwork, check_interval: float = 1.0):
        """
        Initialize health monitor.
        
        Args:
            model: Photonic neural network to monitor
            check_interval: Monitoring interval in seconds
        """
        self.model = model
        self.check_interval = check_interval
        self.validator = PhotonicSystemValidator(enable_monitoring=True)
        self.health_history = []
        self.is_monitoring = False
        
    def start_monitoring(self):
        """Start continuous health monitoring."""
        self.is_monitoring = True
        logger.info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop continuous health monitoring."""
        self.is_monitoring = False
        logger.info("Health monitoring stopped")
    
    def check_health(self) -> ValidationResult:
        """Perform health check and return current system status."""
        validation_result = self.validator.validate_system(self.model)
        
        # Add to history
        self.health_history.append({
            "timestamp": time.time(),
            "validation_result": validation_result
        })
        
        # Alert on critical errors
        if validation_result.has_critical_errors:
            logger.critical("CRITICAL: System health check failed")
        
        return validation_result
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get summary of recent health status."""
        if not self.health_history:
            return {"status": "No monitoring data available"}
        
        recent_results = [entry["validation_result"] for entry in self.health_history[-10:]]
        
        return {
            "current_status": "HEALTHY" if recent_results[-1].is_valid else "DEGRADED",
            "recent_error_rate": sum(1 for r in recent_results if not r.is_valid) / len(recent_results),
            "avg_performance_degradation": np.mean([r.performance_degradation for r in recent_results]),
            "monitoring_duration_hours": (self.health_history[-1]["timestamp"] - 
                                        self.health_history[0]["timestamp"]) / 3600
        }