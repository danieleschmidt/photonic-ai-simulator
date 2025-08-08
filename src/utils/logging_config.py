"""
Logging configuration for photonic AI simulator.

Provides structured logging with performance monitoring,
hardware event tracking, and debug capabilities.
"""

import logging
import logging.config
import sys
import os
from typing import Dict, Any, Optional
import json
from datetime import datetime


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    enable_performance_logging: bool = True,
    enable_hardware_logging: bool = True,
    json_format: bool = False
) -> None:
    """
    Set up comprehensive logging configuration.
    
    Args:
        level: Logging level ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
        log_file: Optional file path for log output
        enable_performance_logging: Enable performance metric logging
        enable_hardware_logging: Enable hardware event logging
        json_format: Use structured JSON logging format
    """
    
    # Create logs directory if logging to file
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure logging format
    if json_format:
        formatter_class = JSONFormatter
        format_string = None  # JSONFormatter doesn't use format string
    else:
        formatter_class = PhotonicFormatter
        format_string = (
            "%(asctime)s | %(name)-20s | %(levelname)-8s | "
            "%(funcName)-15s | %(message)s"
        )
    
    # Build logging configuration
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "detailed": {
                "()": formatter_class,
                "format": format_string
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": level,
                "formatter": "detailed",
                "stream": "ext://sys.stdout"
            }
        },
        "loggers": {
            "photonic_ai_simulator": {
                "level": level,
                "handlers": ["console"],
                "propagate": False
            }
        },
        "root": {
            "level": "WARNING",
            "handlers": ["console"]
        }
    }
    
    # Add file handler if specified
    if log_file:
        config["handlers"]["file"] = {
            "class": "logging.FileHandler",
            "level": level,
            "formatter": "detailed",
            "filename": log_file,
            "mode": "a"
        }
        config["loggers"]["photonic_ai_simulator"]["handlers"].append("file")
        
        # Add rotation for large files
        config["handlers"]["file"]["class"] = "logging.handlers.RotatingFileHandler"
        config["handlers"]["file"]["maxBytes"] = 10 * 1024 * 1024  # 10MB
        config["handlers"]["file"]["backupCount"] = 5
    
    # Add performance logger if enabled
    if enable_performance_logging:
        config["loggers"]["photonic_ai_simulator.performance"] = {
            "level": "DEBUG",
            "handlers": ["console"] + (["file"] if log_file else []),
            "propagate": False
        }
    
    # Add hardware logger if enabled
    if enable_hardware_logging:
        config["loggers"]["photonic_ai_simulator.hardware"] = {
            "level": "DEBUG", 
            "handlers": ["console"] + (["file"] if log_file else []),
            "propagate": False
        }
    
    # Apply configuration
    logging.config.dictConfig(config)


class PhotonicFormatter(logging.Formatter):
    """Custom formatter for photonic neural network logging."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Color codes for different log levels
        self.colors = {
            'DEBUG': '\033[36m',    # Cyan
            'INFO': '\033[32m',     # Green  
            'WARNING': '\033[33m',  # Yellow
            'ERROR': '\033[31m',    # Red
            'CRITICAL': '\033[35m', # Magenta
            'RESET': '\033[0m'      # Reset
        }
    
    def format(self, record):
        # Add color if outputting to terminal
        if hasattr(sys.stdout, 'isatty') and sys.stdout.isatty():
            color = self.colors.get(record.levelname, '')
            reset = self.colors['RESET']
            record.levelname = f"{color}{record.levelname}{reset}"
        
        # Add performance metrics if available
        if hasattr(record, 'latency_ns'):
            record.message = f"{record.getMessage()} (latency: {record.latency_ns}ns)"
        elif hasattr(record, 'power_mw'):
            record.message = f"{record.getMessage()} (power: {record.power_mw}mW)"
        else:
            record.message = record.getMessage()
        
        return super().format(record)


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
        }
        
        # Add performance metrics if available
        performance_fields = ['latency_ns', 'power_mw', 'accuracy', 'throughput_ops']
        for field in performance_fields:
            if hasattr(record, field):
                log_entry[field] = getattr(record, field)
        
        # Add hardware metrics if available
        hardware_fields = ['temperature_k', 'wavelength_nm', 'phase_rad', 'thermal_drift_pm']
        for field in hardware_fields:
            if hasattr(record, field):
                log_entry[field] = getattr(record, field)
        
        # Add exception information if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)


def get_logger(name: str) -> logging.Logger:
    """
    Get logger with photonic AI simulator namespace.
    
    Args:
        name: Logger name (will be prefixed with 'photonic_ai_simulator')
        
    Returns:
        Configured logger instance
    """
    full_name = f"photonic_ai_simulator.{name}"
    return logging.getLogger(full_name)


class PerformanceLogger:
    """Specialized logger for performance metrics."""
    
    def __init__(self):
        self.logger = get_logger("performance")
        self.metrics_history = []
    
    def log_inference(self, latency_ns: float, accuracy: float, power_mw: float):
        """Log inference performance metrics."""
        self.logger.info(
            "Inference completed",
            extra={
                "latency_ns": latency_ns,
                "accuracy": accuracy, 
                "power_mw": power_mw
            }
        )
        
        # Store for analysis
        self.metrics_history.append({
            "timestamp": datetime.utcnow().timestamp(),
            "latency_ns": latency_ns,
            "accuracy": accuracy,
            "power_mw": power_mw
        })
    
    def log_training_epoch(self, epoch: int, loss: float, accuracy: float, 
                          epoch_time_s: float):
        """Log training epoch metrics."""
        self.logger.info(
            f"Epoch {epoch} completed",
            extra={
                "epoch": epoch,
                "loss": loss,
                "accuracy": accuracy,
                "epoch_time_s": epoch_time_s
            }
        )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of recent performance metrics."""
        if not self.metrics_history:
            return {}
        
        recent_metrics = self.metrics_history[-100:]  # Last 100 entries
        
        return {
            "avg_latency_ns": sum(m["latency_ns"] for m in recent_metrics) / len(recent_metrics),
            "avg_accuracy": sum(m["accuracy"] for m in recent_metrics) / len(recent_metrics),
            "avg_power_mw": sum(m["power_mw"] for m in recent_metrics) / len(recent_metrics),
            "total_inferences": len(recent_metrics)
        }


class HardwareLogger:
    """Specialized logger for hardware events and status."""
    
    def __init__(self):
        self.logger = get_logger("hardware")
        self.thermal_events = []
        self.error_events = []
    
    def log_thermal_event(self, temperature_k: float, drift_pm: float, 
                         compensation_power_mw: float):
        """Log thermal management events."""
        self.logger.debug(
            "Thermal compensation applied",
            extra={
                "temperature_k": temperature_k,
                "thermal_drift_pm": drift_pm,
                "power_mw": compensation_power_mw
            }
        )
        
        self.thermal_events.append({
            "timestamp": datetime.utcnow().timestamp(),
            "temperature_k": temperature_k,
            "drift_pm": drift_pm,
            "power_mw": compensation_power_mw
        })
    
    def log_wavelength_event(self, wavelength_nm: float, stability_pm: float):
        """Log wavelength stability events."""
        self.logger.debug(
            "Wavelength monitoring",
            extra={
                "wavelength_nm": wavelength_nm,
                "stability_pm": stability_pm
            }
        )
    
    def log_phase_shifter_event(self, phase_rad: float, voltage_v: float, 
                               power_mw: float):
        """Log phase shifter operation events."""
        self.logger.debug(
            "Phase shifter adjustment",
            extra={
                "phase_rad": phase_rad,
                "voltage_v": voltage_v,
                "power_mw": power_mw
            }
        )
    
    def log_error_event(self, error_type: str, severity: str, message: str):
        """Log hardware error events."""
        self.logger.error(
            f"Hardware error: {message}",
            extra={
                "error_type": error_type,
                "severity": severity
            }
        )
        
        self.error_events.append({
            "timestamp": datetime.utcnow().timestamp(),
            "error_type": error_type,
            "severity": severity,
            "message": message
        })
    
    def get_hardware_summary(self) -> Dict[str, Any]:
        """Get summary of hardware status."""
        return {
            "thermal_events_last_hour": len([
                e for e in self.thermal_events 
                if datetime.utcnow().timestamp() - e["timestamp"] < 3600
            ]),
            "error_events_last_hour": len([
                e for e in self.error_events
                if datetime.utcnow().timestamp() - e["timestamp"] < 3600
            ]),
            "recent_avg_temperature_k": (
                sum(e["temperature_k"] for e in self.thermal_events[-10:]) / 
                min(len(self.thermal_events), 10)
            ) if self.thermal_events else 0.0
        }


# Global logger instances for convenience
performance_logger = PerformanceLogger()
hardware_logger = HardwareLogger()


def log_function_performance(func):
    """Decorator to log function performance automatically."""
    import time
    from functools import wraps
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = time.perf_counter_ns()
        
        try:
            result = func(*args, **kwargs)
            end_time = time.perf_counter_ns()
            latency_ns = end_time - start_time
            
            logger.debug(
                f"Function {func.__name__} completed",
                extra={"latency_ns": latency_ns}
            )
            
            return result
            
        except Exception as e:
            end_time = time.perf_counter_ns()
            latency_ns = end_time - start_time
            
            logger.error(
                f"Function {func.__name__} failed after {latency_ns}ns: {e}",
                extra={"latency_ns": latency_ns}
            )
            raise
            
    return wrapper