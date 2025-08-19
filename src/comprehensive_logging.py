"""
Comprehensive Logging System for Photonic AI.

Provides structured, performance-aware logging with security auditing,
distributed tracing, and production-grade observability for photonic
neural networks and research systems.
"""

import logging
import logging.handlers
import json
import time
import threading
import uuid
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime
import traceback
import functools
import sys
import os
from pathlib import Path
import queue
import atexit


class LogLevel(Enum):
    """Extended log levels for photonic AI."""
    TRACE = 5
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50
    SECURITY = 55
    PERFORMANCE = 15


class EventType(Enum):
    """Types of events for structured logging."""
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    QUANTUM_OPERATION = "quantum_operation"
    WAVELENGTH_MANAGEMENT = "wavelength_management"
    FEDERATED_COMMUNICATION = "federated_communication"
    SECURITY_EVENT = "security_event"
    PERFORMANCE_METRIC = "performance_metric"
    ERROR_RECOVERY = "error_recovery"
    EXPERIMENT_START = "experiment_start"
    EXPERIMENT_END = "experiment_end"
    MODEL_TRAINING = "model_training"
    MODEL_INFERENCE = "model_inference"


@dataclass
class LogContext:
    """Context information for log entries."""
    request_id: str
    session_id: str
    user_id: str
    component: str
    operation: str
    timestamp: float
    thread_id: int
    process_id: int


@dataclass
class PerformanceMetrics:
    """Performance metrics for logging."""
    execution_time_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    gpu_usage_percent: Optional[float]
    throughput_ops_per_sec: Optional[float]
    latency_p50_ms: Optional[float]
    latency_p95_ms: Optional[float]
    latency_p99_ms: Optional[float]


class StructuredLogger:
    """
    High-performance structured logger for photonic AI systems.
    
    Provides JSON-structured logging with performance metrics,
    security auditing, and distributed tracing capabilities.
    """
    
    def __init__(self, 
                 component_name: str,
                 log_level: LogLevel = LogLevel.INFO,
                 log_directory: str = "logs",
                 enable_console: bool = True,
                 enable_file: bool = True,
                 enable_security_audit: bool = True,
                 max_file_size_mb: int = 100,
                 backup_count: int = 10):
        
        self.component_name = component_name
        self.log_level = log_level
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(f"photonic_ai.{component_name}")
        self.logger.setLevel(log_level.value)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Context tracking
        self.context_stack = threading.local()
        self.request_contexts = {}
        self.context_lock = threading.Lock()
        
        # Performance tracking
        self.performance_metrics = {}
        self.metric_lock = threading.Lock()
        
        # Async logging queue
        self.log_queue = queue.Queue(maxsize=10000)
        self.log_thread = threading.Thread(target=self._log_worker, daemon=True)
        self.log_thread.start()
        
        # Setup handlers
        self._setup_handlers(enable_console, enable_file, enable_security_audit,
                           max_file_size_mb, backup_count)
        
        # Register cleanup
        atexit.register(self._cleanup)
        
        self.info("Structured logger initialized", extra={
            "component": component_name,
            "log_level": log_level.value,
            "log_directory": str(log_directory)
        })
    
    def _setup_handlers(self, enable_console: bool, enable_file: bool,
                       enable_security_audit: bool, max_file_size_mb: int,
                       backup_count: int):
        """Setup logging handlers."""
        formatter = JsonFormatter()
        
        # Console handler
        if enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.log_level.value)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # File handler with rotation
        if enable_file:
            file_path = self.log_directory / f"{self.component_name}.log"
            file_handler = logging.handlers.RotatingFileHandler(
                file_path,
                maxBytes=max_file_size_mb * 1024 * 1024,
                backupCount=backup_count
            )
            file_handler.setLevel(self.log_level.value)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        # Security audit handler
        if enable_security_audit:
            security_path = self.log_directory / f"{self.component_name}_security.log"
            security_handler = logging.handlers.RotatingFileHandler(
                security_path,
                maxBytes=max_file_size_mb * 1024 * 1024,
                backupCount=backup_count
            )
            security_handler.setLevel(LogLevel.SECURITY.value)
            security_handler.addFilter(SecurityLogFilter())
            security_handler.setFormatter(formatter)
            self.logger.addHandler(security_handler)
    
    def _log_worker(self):
        """Background worker for async logging."""
        while True:
            try:
                log_record = self.log_queue.get(timeout=1)
                if log_record is None:  # Shutdown signal
                    break
                self.logger.handle(log_record)
                self.log_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                # Avoid infinite recursion in error logging
                print(f"Logging worker error: {e}", file=sys.stderr)
    
    def set_context(self, **context_data):
        """Set logging context for current thread."""
        if not hasattr(self.context_stack, 'context'):
            self.context_stack.context = {}
        
        self.context_stack.context.update(context_data)
    
    def clear_context(self):
        """Clear logging context for current thread."""
        if hasattr(self.context_stack, 'context'):
            self.context_stack.context.clear()
    
    def get_context(self) -> Dict[str, Any]:
        """Get current logging context."""
        if hasattr(self.context_stack, 'context'):
            return self.context_stack.context.copy()
        return {}
    
    def _create_log_record(self, level: int, message: str, 
                          event_type: EventType = None,
                          extra_data: Dict[str, Any] = None) -> logging.LogRecord:
        """Create structured log record."""
        # Get context
        context = self.get_context()
        
        # Base record data
        record_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": logging.getLevelName(level),
            "component": self.component_name,
            "message": message,
            "thread_id": threading.get_ident(),
            "process_id": os.getpid()
        }
        
        # Add context
        record_data.update(context)
        
        # Add event type
        if event_type:
            record_data["event_type"] = event_type.value
        
        # Add extra data
        if extra_data:
            record_data.update(extra_data)
        
        # Create log record
        record = logging.LogRecord(
            name=self.logger.name,
            level=level,
            pathname="",
            lineno=0,
            msg=message,
            args=(),
            exc_info=None
        )
        
        # Add structured data
        record.structured_data = record_data
        
        return record
    
    def log(self, level: LogLevel, message: str, 
           event_type: EventType = None,
           **extra_data):
        """Log message with structured data."""
        record = self._create_log_record(level.value, message, event_type, extra_data)
        
        # Use async logging for high-frequency logs
        if level.value <= LogLevel.INFO.value:
            try:
                self.log_queue.put_nowait(record)
            except queue.Full:
                # Fall back to synchronous logging
                self.logger.handle(record)
        else:
            # Synchronous logging for warnings and errors
            self.logger.handle(record)
    
    def trace(self, message: str, **extra_data):
        """Log trace level message."""
        self.log(LogLevel.TRACE, message, **extra_data)
    
    def debug(self, message: str, **extra_data):
        """Log debug level message."""
        self.log(LogLevel.DEBUG, message, **extra_data)
    
    def info(self, message: str, **extra_data):
        """Log info level message."""
        self.log(LogLevel.INFO, message, **extra_data)
    
    def warning(self, message: str, **extra_data):
        """Log warning level message."""
        self.log(LogLevel.WARNING, message, **extra_data)
    
    def error(self, message: str, exception: Exception = None, **extra_data):
        """Log error level message."""
        if exception:
            extra_data["exception"] = {
                "type": exception.__class__.__name__,
                "message": str(exception),
                "traceback": traceback.format_exc()
            }
        self.log(LogLevel.ERROR, message, **extra_data)
    
    def critical(self, message: str, exception: Exception = None, **extra_data):
        """Log critical level message."""
        if exception:
            extra_data["exception"] = {
                "type": exception.__class__.__name__,
                "message": str(exception),
                "traceback": traceback.format_exc()
            }
        self.log(LogLevel.CRITICAL, message, **extra_data)
    
    def security(self, message: str, **extra_data):
        """Log security event."""
        extra_data["security_event"] = True
        self.log(LogLevel.SECURITY, message, EventType.SECURITY_EVENT, **extra_data)
    
    def performance(self, message: str, metrics: PerformanceMetrics, **extra_data):
        """Log performance metrics."""
        extra_data["performance_metrics"] = asdict(metrics)
        self.log(LogLevel.PERFORMANCE, message, EventType.PERFORMANCE_METRIC, **extra_data)
    
    def _cleanup(self):
        """Cleanup logging resources."""
        # Signal worker thread to stop
        self.log_queue.put(None)
        self.log_thread.join(timeout=5)
        
        # Close handlers
        for handler in self.logger.handlers:
            handler.close()


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record):
        """Format log record as JSON."""
        if hasattr(record, 'structured_data'):
            log_data = record.structured_data
        else:
            # Fallback for non-structured records
            log_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "level": record.levelname,
                "message": record.getMessage(),
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno
            }
            
            if record.exc_info:
                log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data, default=str)


class SecurityLogFilter(logging.Filter):
    """Filter for security-related log entries."""
    
    def filter(self, record):
        """Filter security events."""
        if hasattr(record, 'structured_data'):
            return record.structured_data.get('security_event', False)
        return record.levelno >= LogLevel.SECURITY.value


class PerformanceLogger:
    """
    Specialized logger for performance metrics and profiling.
    
    Tracks execution times, resource usage, and system performance
    for photonic AI operations.
    """
    
    def __init__(self, base_logger: StructuredLogger):
        self.base_logger = base_logger
        self.active_operations = {}
        self.operation_lock = threading.Lock()
        
        # Performance history
        self.performance_history = {}
        self.history_lock = threading.Lock()
    
    def start_operation(self, operation_name: str, **context) -> str:
        """Start tracking an operation."""
        operation_id = str(uuid.uuid4())
        
        start_data = {
            "operation_id": operation_id,
            "operation_name": operation_name,
            "start_time": time.perf_counter(),
            "start_memory": self._get_memory_usage(),
            "context": context
        }
        
        with self.operation_lock:
            self.active_operations[operation_id] = start_data
        
        self.base_logger.debug(
            f"Started operation: {operation_name}",
            event_type=EventType.PERFORMANCE_METRIC,
            operation_id=operation_id,
            operation_name=operation_name,
            **context
        )
        
        return operation_id
    
    def end_operation(self, operation_id: str, 
                     success: bool = True,
                     result_data: Dict[str, Any] = None):
        """End tracking an operation."""
        end_time = time.perf_counter()
        end_memory = self._get_memory_usage()
        
        with self.operation_lock:
            start_data = self.active_operations.pop(operation_id, None)
        
        if not start_data:
            self.base_logger.warning(f"Unknown operation ID: {operation_id}")
            return
        
        # Calculate metrics
        execution_time = (end_time - start_data["start_time"]) * 1000  # ms
        memory_delta = end_memory - start_data["start_memory"]
        
        metrics = PerformanceMetrics(
            execution_time_ms=execution_time,
            memory_usage_mb=end_memory,
            cpu_usage_percent=self._get_cpu_usage(),
            gpu_usage_percent=None,  # Would be implemented with GPU monitoring
            throughput_ops_per_sec=None,
            latency_p50_ms=None,
            latency_p95_ms=None,
            latency_p99_ms=None
        )
        
        # Store in history
        operation_name = start_data["operation_name"]
        with self.history_lock:
            if operation_name not in self.performance_history:
                self.performance_history[operation_name] = []
            
            self.performance_history[operation_name].append({
                "timestamp": time.time(),
                "execution_time_ms": execution_time,
                "memory_delta_mb": memory_delta,
                "success": success
            })
            
            # Keep only recent history
            if len(self.performance_history[operation_name]) > 1000:
                self.performance_history[operation_name] = self.performance_history[operation_name][-1000:]
        
        # Log completion
        log_data = {
            "operation_id": operation_id,
            "operation_name": operation_name,
            "execution_time_ms": execution_time,
            "memory_delta_mb": memory_delta,
            "success": success
        }
        
        if result_data:
            log_data["result"] = result_data
        
        self.base_logger.performance(
            f"Completed operation: {operation_name}",
            metrics,
            **log_data
        )
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        try:
            import psutil
            return psutil.cpu_percent(interval=None)
        except ImportError:
            return 0.0
    
    def get_operation_stats(self, operation_name: str) -> Dict[str, Any]:
        """Get statistics for an operation."""
        with self.history_lock:
            history = self.performance_history.get(operation_name, [])
        
        if not history:
            return {"operation_name": operation_name, "no_data": True}
        
        execution_times = [h["execution_time_ms"] for h in history]
        memory_deltas = [h["memory_delta_mb"] for h in history]
        success_rate = sum(h["success"] for h in history) / len(history)
        
        return {
            "operation_name": operation_name,
            "total_executions": len(history),
            "success_rate": success_rate,
            "avg_execution_time_ms": sum(execution_times) / len(execution_times),
            "min_execution_time_ms": min(execution_times),
            "max_execution_time_ms": max(execution_times),
            "avg_memory_delta_mb": sum(memory_deltas) / len(memory_deltas),
            "recent_executions": len([h for h in history if time.time() - h["timestamp"] < 3600])
        }


def log_function_performance(logger: StructuredLogger = None):
    """Decorator to automatically log function performance."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get or create logger
            if logger is None:
                component_name = func.__module__.split('.')[-1]
                func_logger = StructuredLogger(component_name)
            else:
                func_logger = logger
            
            perf_logger = PerformanceLogger(func_logger)
            
            # Start tracking
            operation_id = perf_logger.start_operation(
                f"{func.__module__}.{func.__name__}",
                function=func.__name__,
                module=func.__module__
            )
            
            try:
                result = func(*args, **kwargs)
                perf_logger.end_operation(operation_id, success=True)
                return result
            
            except Exception as e:
                perf_logger.end_operation(operation_id, success=False)
                func_logger.error(
                    f"Function {func.__name__} failed",
                    exception=e,
                    operation_id=operation_id
                )
                raise
        
        return wrapper
    return decorator


class DistributedTracer:
    """
    Distributed tracing for federated photonic AI systems.
    
    Tracks requests across multiple components and systems
    for comprehensive observability.
    """
    
    def __init__(self, service_name: str, logger: StructuredLogger):
        self.service_name = service_name
        self.logger = logger
        self.active_traces = {}
        self.trace_lock = threading.Lock()
    
    def start_trace(self, operation_name: str, 
                   parent_trace_id: str = None,
                   **context) -> str:
        """Start a new trace."""
        trace_id = str(uuid.uuid4())
        span_id = str(uuid.uuid4())
        
        trace_data = {
            "trace_id": trace_id,
            "span_id": span_id,
            "parent_span_id": parent_trace_id,
            "service_name": self.service_name,
            "operation_name": operation_name,
            "start_time": time.time(),
            "context": context
        }
        
        with self.trace_lock:
            self.active_traces[trace_id] = trace_data
        
        self.logger.debug(
            f"Started trace: {operation_name}",
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_trace_id,
            operation_name=operation_name,
            **context
        )
        
        return trace_id
    
    def end_trace(self, trace_id: str, status: str = "success", **result_data):
        """End a trace."""
        end_time = time.time()
        
        with self.trace_lock:
            trace_data = self.active_traces.pop(trace_id, None)
        
        if not trace_data:
            self.logger.warning(f"Unknown trace ID: {trace_id}")
            return
        
        duration_ms = (end_time - trace_data["start_time"]) * 1000
        
        self.logger.info(
            f"Completed trace: {trace_data['operation_name']}",
            trace_id=trace_id,
            span_id=trace_data["span_id"],
            duration_ms=duration_ms,
            status=status,
            **result_data
        )
    
    def add_trace_event(self, trace_id: str, event_name: str, **event_data):
        """Add an event to an active trace."""
        with self.trace_lock:
            trace_data = self.active_traces.get(trace_id)
        
        if not trace_data:
            self.logger.warning(f"Unknown trace ID for event: {trace_id}")
            return
        
        self.logger.debug(
            f"Trace event: {event_name}",
            trace_id=trace_id,
            span_id=trace_data["span_id"],
            event_name=event_name,
            **event_data
        )


# Global logger registry
_logger_registry = {}
_registry_lock = threading.Lock()


def get_logger(component_name: str, **kwargs) -> StructuredLogger:
    """Get or create a logger for a component."""
    with _registry_lock:
        if component_name not in _logger_registry:
            _logger_registry[component_name] = StructuredLogger(component_name, **kwargs)
        return _logger_registry[component_name]


# Example usage and testing
if __name__ == "__main__":
    # Test logging system
    logger = StructuredLogger("test_component")
    
    # Set context
    logger.set_context(
        request_id="req_123",
        user_id="user_456"
    )
    
    # Test different log levels
    logger.info("System starting up", system_version="1.0.0")
    logger.warning("High memory usage detected", memory_usage_mb=850)
    logger.error("Failed to connect to quantum processor", retry_count=3)
    
    # Test performance logging
    perf_logger = PerformanceLogger(logger)
    
    @log_function_performance(logger)
    def test_function():
        time.sleep(0.1)
        return "test_result"
    
    result = test_function()
    
    # Test distributed tracing
    tracer = DistributedTracer("test_service", logger)
    trace_id = tracer.start_trace("test_operation", context="test")
    tracer.add_trace_event(trace_id, "processing_data")
    tracer.end_trace(trace_id, status="success")
    
    print("Logging system test completed")