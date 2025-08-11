"""
Advanced monitoring and alerting system for photonic neural networks.

Implements real-time system monitoring, performance tracking, alert management,
and automated diagnostics for production photonic AI systems.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
import logging
import time
import json
import threading
from dataclasses import dataclass, asdict
from enum import Enum
from abc import ABC, abstractmethod
from collections import deque, defaultdict
import queue

try:
    from ..validation import ValidationResult, PhotonicSystemValidator
    from ..models import PhotonicNeuralNetwork
except ImportError:
    from validation import ValidationResult, PhotonicSystemValidator
    from models import PhotonicNeuralNetwork

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning" 
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of monitored metrics."""
    PERFORMANCE = "performance"
    HARDWARE = "hardware"
    SYSTEM_HEALTH = "system_health"
    ACCURACY = "accuracy"
    POWER = "power"
    THERMAL = "thermal"


@dataclass
class Alert:
    """System alert with metadata."""
    id: str
    level: AlertLevel
    metric_type: MetricType
    title: str
    message: str
    timestamp: float
    source_component: str
    current_value: Optional[float] = None
    threshold_value: Optional[float] = None
    additional_data: Dict[str, Any] = None
    resolved: bool = False
    resolution_timestamp: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return asdict(self)
    
    def resolve(self, resolution_message: str = ""):
        """Mark alert as resolved."""
        self.resolved = True
        self.resolution_timestamp = time.time()
        if resolution_message:
            if self.additional_data is None:
                self.additional_data = {}
            self.additional_data["resolution_message"] = resolution_message


@dataclass  
class SystemMetrics:
    """System performance and health metrics."""
    timestamp: float
    accuracy: Optional[float] = None
    latency_ns: Optional[float] = None
    power_mw: Optional[float] = None
    temperature_k: Optional[float] = None
    thermal_drift_pm: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    cpu_utilization: Optional[float] = None
    gpu_utilization: Optional[float] = None
    inference_count: Optional[int] = None
    error_count: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {k: v for k, v in asdict(self).items() if v is not None}


class MetricCollector(ABC):
    """Abstract base class for metric collectors."""
    
    @abstractmethod
    def collect(self) -> SystemMetrics:
        """Collect system metrics."""
        pass


class PhotonicMetricCollector(MetricCollector):
    """Metric collector for photonic neural networks."""
    
    def __init__(self, model: PhotonicNeuralNetwork):
        """Initialize collector with model reference."""
        self.model = model
        self.inference_count = 0
        self.error_count = 0
        self.last_accuracy = None
        
    def collect(self) -> SystemMetrics:
        """Collect photonic system metrics."""
        timestamp = time.time()
        
        # Collect hardware metrics from model layers
        total_power = 0.0
        temperatures = []
        
        for layer in self.model.layers:
            processor = layer.processor
            total_power += processor.power_consumption
            temperatures.append(processor.current_temperature)
        
        avg_temperature = np.mean(temperatures) if temperatures else None
        
        # Estimate thermal drift
        base_temp = 300.0  # Reference temperature
        thermal_drift_pm = None
        if avg_temperature:
            temp_diff = avg_temperature - base_temp
            thermal_drift_pm = temp_diff * 10.0  # 10pm/K typical drift
        
        # Memory usage estimation (simplified)
        memory_usage_mb = 0.0
        for layer in self.model.layers:
            if hasattr(layer.processor, '_estimate_memory_usage'):
                memory_usage_mb += layer.processor._estimate_memory_usage()
        
        # Get performance metrics if available
        performance_metrics = {}
        if hasattr(self.model.layers[0].processor, 'get_performance_metrics'):
            performance_metrics = self.model.layers[0].processor.get_performance_metrics()
        
        return SystemMetrics(
            timestamp=timestamp,
            accuracy=self.last_accuracy,
            latency_ns=performance_metrics.get('avg_latency_ns'),
            power_mw=total_power,
            temperature_k=avg_temperature,
            thermal_drift_pm=thermal_drift_pm,
            memory_usage_mb=memory_usage_mb,
            inference_count=self.inference_count,
            error_count=self.error_count
        )
    
    def record_inference(self, accuracy: Optional[float] = None):
        """Record an inference operation."""
        self.inference_count += 1
        if accuracy is not None:
            self.last_accuracy = accuracy
    
    def record_error(self):
        """Record an error occurrence."""
        self.error_count += 1


class AlertManager:
    """Manages system alerts with intelligent filtering and escalation."""
    
    def __init__(self, max_alerts: int = 1000):
        """Initialize alert manager."""
        self.alerts = deque(maxlen=max_alerts)
        self.alert_handlers = defaultdict(list)
        self.alert_counter = 0
        self.suppression_rules = {}
        self._lock = threading.Lock()
        
    def register_handler(self, alert_level: AlertLevel, handler: Callable[[Alert], None]):
        """Register handler for specific alert levels."""
        self.alert_handlers[alert_level].append(handler)
        
    def create_alert(self, level: AlertLevel, metric_type: MetricType,
                    title: str, message: str, source_component: str,
                    current_value: Optional[float] = None,
                    threshold_value: Optional[float] = None,
                    additional_data: Optional[Dict[str, Any]] = None) -> Alert:
        """Create new system alert."""
        
        with self._lock:
            self.alert_counter += 1
            alert_id = f"alert_{self.alert_counter}_{int(time.time())}"
            
            alert = Alert(
                id=alert_id,
                level=level,
                metric_type=metric_type,
                title=title,
                message=message,
                timestamp=time.time(),
                source_component=source_component,
                current_value=current_value,
                threshold_value=threshold_value,
                additional_data=additional_data or {}
            )
            
            # Check suppression rules
            if not self._is_suppressed(alert):
                self.alerts.append(alert)
                self._notify_handlers(alert)
                logger.log(self._alert_level_to_log_level(level), 
                          f"[{level.value.upper()}] {title}: {message}")
        
        return alert
    
    def _is_suppressed(self, alert: Alert) -> bool:
        """Check if alert should be suppressed based on rules."""
        suppression_key = f"{alert.metric_type.value}_{alert.source_component}"
        
        if suppression_key in self.suppression_rules:
            rule = self.suppression_rules[suppression_key]
            time_since_last = time.time() - rule.get("last_alert_time", 0)
            
            if time_since_last < rule.get("min_interval_seconds", 60):
                return True  # Suppress due to rate limit
        
        # Update suppression tracking
        self.suppression_rules[suppression_key] = {
            "last_alert_time": time.time(),
            "min_interval_seconds": 60  # Default 1 minute
        }
        
        return False
    
    def _notify_handlers(self, alert: Alert):
        """Notify registered handlers of new alert."""
        handlers = self.alert_handlers.get(alert.level, [])
        handlers.extend(self.alert_handlers.get(AlertLevel.INFO, []))  # Global handlers
        
        for handler in handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")
    
    def _alert_level_to_log_level(self, alert_level: AlertLevel) -> int:
        """Convert alert level to logging level."""
        mapping = {
            AlertLevel.INFO: logging.INFO,
            AlertLevel.WARNING: logging.WARNING,
            AlertLevel.ERROR: logging.ERROR,
            AlertLevel.CRITICAL: logging.CRITICAL
        }
        return mapping.get(alert_level, logging.INFO)
    
    def get_active_alerts(self, level: Optional[AlertLevel] = None) -> List[Alert]:
        """Get list of active (unresolved) alerts."""
        with self._lock:
            alerts = [alert for alert in self.alerts if not alert.resolved]
            
            if level:
                alerts = [alert for alert in alerts if alert.level == level]
            
            return sorted(alerts, key=lambda a: a.timestamp, reverse=True)
    
    def resolve_alert(self, alert_id: str, resolution_message: str = ""):
        """Resolve an alert by ID."""
        with self._lock:
            for alert in self.alerts:
                if alert.id == alert_id:
                    alert.resolve(resolution_message)
                    logger.info(f"Alert {alert_id} resolved: {resolution_message}")
                    return True
            return False
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of alert status."""
        with self._lock:
            active_alerts = self.get_active_alerts()
            
            summary = {
                "total_alerts": len(self.alerts),
                "active_alerts": len(active_alerts),
                "critical_alerts": len([a for a in active_alerts if a.level == AlertLevel.CRITICAL]),
                "error_alerts": len([a for a in active_alerts if a.level == AlertLevel.ERROR]),
                "warning_alerts": len([a for a in active_alerts if a.level == AlertLevel.WARNING]),
                "alert_rate_per_hour": len([a for a in self.alerts 
                                          if time.time() - a.timestamp < 3600]),
            }
            
            if active_alerts:
                summary["latest_critical"] = next(
                    (a.to_dict() for a in active_alerts if a.level == AlertLevel.CRITICAL),
                    None
                )
            
            return summary


class SystemMonitor:
    """
    Comprehensive system monitor for photonic neural networks.
    
    Provides real-time monitoring, alerting, and automated diagnostics
    for production photonic AI systems.
    """
    
    def __init__(self, model: PhotonicNeuralNetwork, 
                 monitoring_interval: float = 5.0):
        """
        Initialize system monitor.
        
        Args:
            model: Photonic neural network to monitor
            monitoring_interval: Monitoring interval in seconds
        """
        self.model = model
        self.monitoring_interval = monitoring_interval
        self.metric_collector = PhotonicMetricCollector(model)
        self.alert_manager = AlertManager()
        self.validator = PhotonicSystemValidator()
        
        # Metric history
        self.metrics_history = deque(maxlen=1000)
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_thread = None
        
        # Alert thresholds
        self.thresholds = self._default_thresholds()
        
        # Register default alert handlers
        self._setup_default_handlers()
        
    def _default_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Default alert thresholds."""
        return {
            "accuracy": {"warning": 0.8, "critical": 0.5},
            "latency_ns": {"warning": 2.0, "critical": 10.0},
            "power_mw": {"warning": 800.0, "critical": 1000.0},
            "temperature_k": {"warning": 350.0, "critical": 400.0},
            "thermal_drift_pm": {"warning": 50.0, "critical": 100.0},
            "memory_usage_mb": {"warning": 1000.0, "critical": 2000.0},
            "error_rate": {"warning": 0.01, "critical": 0.05}
        }
    
    def _setup_default_handlers(self):
        """Set up default alert handlers."""
        
        def log_alert(alert: Alert):
            """Default logging handler."""
            logger.info(f"Alert: {alert.title} - {alert.message}")
        
        def critical_alert_handler(alert: Alert):
            """Handler for critical alerts."""
            logger.critical(f"CRITICAL ALERT: {alert.title}")
            # In production, this might trigger additional actions like:
            # - Sending notifications
            # - Initiating failover procedures  
            # - Creating support tickets
            
        self.alert_manager.register_handler(AlertLevel.INFO, log_alert)
        self.alert_manager.register_handler(AlertLevel.WARNING, log_alert)
        self.alert_manager.register_handler(AlertLevel.ERROR, log_alert)
        self.alert_manager.register_handler(AlertLevel.CRITICAL, critical_alert_handler)
    
    def start_monitoring(self):
        """Start continuous monitoring."""
        if self.is_monitoring:
            logger.warning("Monitoring already started")
            return
            
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        logger.info(f"System monitoring started (interval: {self.monitoring_interval}s)")
    
    def stop_monitoring(self):
        """Stop continuous monitoring."""
        if not self.is_monitoring:
            return
            
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)
            
        logger.info("System monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Collect metrics
                metrics = self.metric_collector.collect()
                self.metrics_history.append(metrics)
                
                # Check thresholds and generate alerts
                self._check_thresholds(metrics)
                
                # Periodic system validation
                if len(self.metrics_history) % 10 == 0:  # Every 10th cycle
                    self._perform_system_validation()
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(1.0)  # Brief pause before retry
    
    def _check_thresholds(self, metrics: SystemMetrics):
        """Check metrics against alert thresholds."""
        
        # Accuracy alerts
        if metrics.accuracy is not None:
            if metrics.accuracy < self.thresholds["accuracy"]["critical"]:
                self.alert_manager.create_alert(
                    AlertLevel.CRITICAL, MetricType.ACCURACY,
                    "Critical Accuracy Degradation",
                    f"Model accuracy {metrics.accuracy:.3f} below critical threshold",
                    "accuracy_monitor",
                    current_value=metrics.accuracy,
                    threshold_value=self.thresholds["accuracy"]["critical"]
                )
            elif metrics.accuracy < self.thresholds["accuracy"]["warning"]:
                self.alert_manager.create_alert(
                    AlertLevel.WARNING, MetricType.ACCURACY,
                    "Accuracy Degradation Warning", 
                    f"Model accuracy {metrics.accuracy:.3f} below warning threshold",
                    "accuracy_monitor",
                    current_value=metrics.accuracy,
                    threshold_value=self.thresholds["accuracy"]["warning"]
                )
        
        # Latency alerts
        if metrics.latency_ns is not None:
            if metrics.latency_ns > self.thresholds["latency_ns"]["critical"]:
                self.alert_manager.create_alert(
                    AlertLevel.CRITICAL, MetricType.PERFORMANCE,
                    "Critical Latency Increase",
                    f"Inference latency {metrics.latency_ns:.2f}ns exceeds critical threshold",
                    "latency_monitor",
                    current_value=metrics.latency_ns,
                    threshold_value=self.thresholds["latency_ns"]["critical"]
                )
            elif metrics.latency_ns > self.thresholds["latency_ns"]["warning"]:
                self.alert_manager.create_alert(
                    AlertLevel.WARNING, MetricType.PERFORMANCE,
                    "Latency Performance Warning",
                    f"Inference latency {metrics.latency_ns:.2f}ns exceeds warning threshold", 
                    "latency_monitor",
                    current_value=metrics.latency_ns,
                    threshold_value=self.thresholds["latency_ns"]["warning"]
                )
        
        # Power alerts
        if metrics.power_mw is not None:
            if metrics.power_mw > self.thresholds["power_mw"]["critical"]:
                self.alert_manager.create_alert(
                    AlertLevel.CRITICAL, MetricType.POWER,
                    "Critical Power Consumption",
                    f"Power consumption {metrics.power_mw:.1f}mW exceeds safe limits",
                    "power_monitor",
                    current_value=metrics.power_mw,
                    threshold_value=self.thresholds["power_mw"]["critical"]
                )
        
        # Thermal alerts
        if metrics.temperature_k is not None:
            if metrics.temperature_k > self.thresholds["temperature_k"]["critical"]:
                self.alert_manager.create_alert(
                    AlertLevel.CRITICAL, MetricType.THERMAL,
                    "Critical Temperature Alert",
                    f"System temperature {metrics.temperature_k:.1f}K exceeds safe operating range",
                    "thermal_monitor",
                    current_value=metrics.temperature_k,
                    threshold_value=self.thresholds["temperature_k"]["critical"]
                )
            elif metrics.temperature_k > self.thresholds["temperature_k"]["warning"]:
                self.alert_manager.create_alert(
                    AlertLevel.WARNING, MetricType.THERMAL,
                    "Temperature Warning",
                    f"System temperature {metrics.temperature_k:.1f}K approaching limits",
                    "thermal_monitor",
                    current_value=metrics.temperature_k,
                    threshold_value=self.thresholds["temperature_k"]["warning"]
                )
        
        # Error rate calculation and alerts
        if len(self.metrics_history) >= 10:
            recent_metrics = list(self.metrics_history)[-10:]
            total_inferences = sum(m.inference_count or 0 for m in recent_metrics)
            total_errors = sum(m.error_count or 0 for m in recent_metrics)
            
            if total_inferences > 0:
                error_rate = total_errors / total_inferences
                
                if error_rate > self.thresholds["error_rate"]["critical"]:
                    self.alert_manager.create_alert(
                        AlertLevel.CRITICAL, MetricType.SYSTEM_HEALTH,
                        "High Error Rate Detected",
                        f"Error rate {error_rate:.1%} indicates system instability",
                        "error_monitor",
                        current_value=error_rate,
                        threshold_value=self.thresholds["error_rate"]["critical"]
                    )
    
    def _perform_system_validation(self):
        """Perform periodic comprehensive system validation."""
        try:
            validation_result = self.validator.validate_system(self.model)
            
            if validation_result.has_critical_errors:
                self.alert_manager.create_alert(
                    AlertLevel.CRITICAL, MetricType.SYSTEM_HEALTH,
                    "System Validation Failed",
                    f"Critical validation errors detected: {len(validation_result.errors)} errors",
                    "system_validator",
                    additional_data={"validation_errors": len(validation_result.errors)}
                )
            elif validation_result.errors:
                self.alert_manager.create_alert(
                    AlertLevel.WARNING, MetricType.SYSTEM_HEALTH,
                    "System Validation Warnings",
                    f"System validation found issues: {len(validation_result.errors)} errors, "
                    f"{len(validation_result.warnings)} warnings",
                    "system_validator"
                )
                
        except Exception as e:
            logger.error(f"System validation failed: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status report."""
        if not self.metrics_history:
            return {"status": "No monitoring data available"}
        
        latest_metrics = self.metrics_history[-1]
        recent_metrics = list(self.metrics_history)[-10:] if len(self.metrics_history) >= 10 else list(self.metrics_history)
        
        # Calculate trends
        def calculate_trend(metric_name: str) -> Optional[str]:
            values = [getattr(m, metric_name) for m in recent_metrics if getattr(m, metric_name) is not None]
            if len(values) < 2:
                return None
            
            first_half = np.mean(values[:len(values)//2])
            second_half = np.mean(values[len(values)//2:])
            
            change_percent = ((second_half - first_half) / first_half) * 100 if first_half != 0 else 0
            
            if abs(change_percent) < 5:
                return "stable"
            elif change_percent > 0:
                return "increasing"
            else:
                return "decreasing"
        
        status_report = {
            "monitoring_status": "active" if self.is_monitoring else "stopped",
            "last_update": latest_metrics.timestamp,
            "current_metrics": latest_metrics.to_dict(),
            "trends": {
                "accuracy": calculate_trend("accuracy"),
                "latency_ns": calculate_trend("latency_ns"),
                "power_mw": calculate_trend("power_mw"),
                "temperature_k": calculate_trend("temperature_k")
            },
            "alerts": self.alert_manager.get_alert_summary(),
            "system_health": self._assess_system_health(),
            "monitoring_duration_hours": (
                (time.time() - self.metrics_history[0].timestamp) / 3600 
                if self.metrics_history else 0
            )
        }
        
        return status_report
    
    def _assess_system_health(self) -> Dict[str, Any]:
        """Assess overall system health."""
        active_alerts = self.alert_manager.get_active_alerts()
        critical_alerts = [a for a in active_alerts if a.level == AlertLevel.CRITICAL]
        error_alerts = [a for a in active_alerts if a.level == AlertLevel.ERROR]
        
        if critical_alerts:
            health_status = "critical"
            health_score = 0.2
        elif error_alerts:
            health_status = "degraded"
            health_score = 0.6
        elif len(active_alerts) > 5:
            health_status = "warning"
            health_score = 0.8
        else:
            health_status = "healthy"
            health_score = 1.0
        
        return {
            "status": health_status,
            "score": health_score,
            "active_issues": len(active_alerts),
            "critical_issues": len(critical_alerts),
            "last_assessment": time.time()
        }
    
    def export_metrics(self, filepath: str, format: str = "json"):
        """Export metrics history to file."""
        if not self.metrics_history:
            logger.warning("No metrics data to export")
            return
        
        export_data = {
            "export_timestamp": time.time(),
            "monitoring_interval": self.monitoring_interval,
            "metrics_count": len(self.metrics_history),
            "metrics": [metrics.to_dict() for metrics in self.metrics_history]
        }
        
        if format.lower() == "json":
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Metrics exported to {filepath} ({len(self.metrics_history)} records)")


def create_production_monitor(model: PhotonicNeuralNetwork,
                            monitoring_interval: float = 5.0,
                            custom_thresholds: Optional[Dict[str, Dict[str, float]]] = None) -> SystemMonitor:
    """
    Create production-ready system monitor with optimal configuration.
    
    Args:
        model: Photonic neural network to monitor
        monitoring_interval: Monitoring interval in seconds
        custom_thresholds: Custom alert thresholds
        
    Returns:
        Configured system monitor
    """
    monitor = SystemMonitor(model, monitoring_interval)
    
    if custom_thresholds:
        monitor.thresholds.update(custom_thresholds)
    
    # Register additional production handlers
    def critical_alert_handler(alert: Alert):
        """Production critical alert handler."""
        logger.critical(f"PRODUCTION CRITICAL: {alert.title}")
        # In real production:
        # - Send email/SMS notifications
        # - Create PagerDuty incidents
        # - Trigger automatic failover
        # - Log to central monitoring system
    
    monitor.alert_manager.register_handler(AlertLevel.CRITICAL, critical_alert_handler)
    
    return monitor