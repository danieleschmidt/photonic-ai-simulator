"""
Resilience and fault tolerance module for photonic AI systems.

Implements circuit breakers, retry mechanisms, graceful degradation,
and automated recovery for production photonic neural networks.
"""

import asyncio
import logging
import time
import random
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
import threading
from collections import deque, defaultdict
import json

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, blocking requests
    HALF_OPEN = "half_open"  # Testing recovery


class FailureType(Enum):
    """Types of system failures."""
    HARDWARE_ERROR = "hardware_error"
    THERMAL_FAILURE = "thermal_failure"
    INFERENCE_TIMEOUT = "inference_timeout"
    MEMORY_OVERFLOW = "memory_overflow"
    NETWORK_ERROR = "network_error"
    VALIDATION_ERROR = "validation_error"
    RESOURCE_EXHAUSTION = "resource_exhaustion"


@dataclass
class FailureRecord:
    """Record of a system failure."""
    timestamp: float
    failure_type: FailureType
    component: str
    error_message: str
    severity: str  # "low", "medium", "high", "critical"
    recovery_time_s: Optional[float] = None
    mitigation_applied: Optional[str] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    success_threshold: int = 3
    timeout_seconds: float = 60.0
    max_concurrent_requests: int = 100
    failure_rate_threshold: float = 0.5  # 50% failure rate triggers open
    monitoring_window_seconds: float = 300.0  # 5 minute window


class CircuitBreaker:
    """
    Circuit breaker for protecting photonic system components.
    
    Implements the circuit breaker pattern to prevent cascading failures
    and allow systems to recover gracefully from errors.
    """
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        """
        Initialize circuit breaker.
        
        Args:
            name: Name of the protected component
            config: Circuit breaker configuration
        """
        self.name = name
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        self.concurrent_requests = 0
        self.request_history = deque(maxlen=1000)
        self._lock = threading.Lock()
        
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerOpenError: If circuit is open
            ResourceExhaustionError: If too many concurrent requests
        """
        with self._lock:
            # Check concurrent request limit
            if self.concurrent_requests >= self.config.max_concurrent_requests:
                raise ResourceExhaustionError(
                    f"Too many concurrent requests: {self.concurrent_requests}"
                )
            
            # Check circuit state
            if self.state == CircuitState.OPEN:
                if time.time() - self.last_failure_time < self.config.timeout_seconds:
                    raise CircuitBreakerOpenError(f"Circuit breaker {self.name} is open")
                else:
                    # Try to transition to half-open
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    logger.info(f"Circuit breaker {self.name} transitioning to half-open")
            
            self.concurrent_requests += 1
        
        start_time = time.time()
        try:
            # Execute function
            result = func(*args, **kwargs)
            
            # Record success
            execution_time = time.time() - start_time
            self._record_success(execution_time)
            
            return result
            
        except Exception as e:
            # Record failure
            execution_time = time.time() - start_time
            self._record_failure(e, execution_time)
            raise
            
        finally:
            with self._lock:
                self.concurrent_requests -= 1
    
    def _record_success(self, execution_time: float):
        """Record successful request."""
        with self._lock:
            current_time = time.time()
            
            self.request_history.append({
                "timestamp": current_time,
                "success": True,
                "execution_time": execution_time
            })
            
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    logger.info(f"Circuit breaker {self.name} closed (recovered)")
            
            elif self.state == CircuitState.CLOSED:
                # Reset failure count on success
                self.failure_count = max(0, self.failure_count - 1)
    
    def _record_failure(self, error: Exception, execution_time: float):
        """Record failed request."""
        with self._lock:
            current_time = time.time()
            
            self.request_history.append({
                "timestamp": current_time,
                "success": False,
                "error": str(error),
                "execution_time": execution_time
            })
            
            self.failure_count += 1
            self.last_failure_time = current_time
            
            # Check if we should open the circuit
            if self.state == CircuitState.CLOSED:
                failure_rate = self._calculate_failure_rate()
                
                if (self.failure_count >= self.config.failure_threshold or 
                    failure_rate >= self.config.failure_rate_threshold):
                    self.state = CircuitState.OPEN
                    logger.warning(
                        f"Circuit breaker {self.name} opened due to failures "
                        f"(count: {self.failure_count}, rate: {failure_rate:.2%})"
                    )
            
            elif self.state == CircuitState.HALF_OPEN:
                # Return to open state on any failure in half-open
                self.state = CircuitState.OPEN
                logger.warning(f"Circuit breaker {self.name} returned to open state")
    
    def _calculate_failure_rate(self) -> float:
        """Calculate failure rate over monitoring window."""
        current_time = time.time()
        window_start = current_time - self.config.monitoring_window_seconds
        
        recent_requests = [
            req for req in self.request_history 
            if req["timestamp"] >= window_start
        ]
        
        if not recent_requests:
            return 0.0
        
        failed_requests = [req for req in recent_requests if not req["success"]]
        return len(failed_requests) / len(recent_requests)
    
    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status."""
        with self._lock:
            failure_rate = self._calculate_failure_rate()
            
            return {
                "name": self.name,
                "state": self.state.value,
                "failure_count": self.failure_count,
                "success_count": self.success_count,
                "failure_rate": failure_rate,
                "concurrent_requests": self.concurrent_requests,
                "total_requests": len(self.request_history),
                "last_failure_time": self.last_failure_time
            }


class RetryConfig:
    """Configuration for retry mechanisms."""
    
    def __init__(self,
                 max_attempts: int = 3,
                 base_delay: float = 1.0,
                 max_delay: float = 60.0,
                 exponential_base: float = 2.0,
                 jitter: bool = True,
                 retryable_exceptions: List[type] = None):
        """Initialize retry configuration."""
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions or [
            ConnectionError, TimeoutError, ResourceExhaustionError
        ]


def with_retry(config: RetryConfig):
    """Decorator for adding retry logic to functions."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(config.max_attempts):
                try:
                    return func(*args, **kwargs)
                    
                except Exception as e:
                    last_exception = e
                    
                    # Check if exception is retryable
                    if not any(isinstance(e, exc_type) for exc_type in config.retryable_exceptions):
                        raise  # Not retryable, re-raise immediately
                    
                    # Don't retry on last attempt
                    if attempt == config.max_attempts - 1:
                        break
                    
                    # Calculate delay
                    delay = min(
                        config.base_delay * (config.exponential_base ** attempt),
                        config.max_delay
                    )
                    
                    # Add jitter to prevent thundering herd
                    if config.jitter:
                        delay *= (0.5 + random.random() * 0.5)
                    
                    logger.warning(
                        f"Attempt {attempt + 1}/{config.max_attempts} failed for {func.__name__}: {e}. "
                        f"Retrying in {delay:.2f}s"
                    )
                    
                    time.sleep(delay)
            
            # All attempts failed
            logger.error(f"All {config.max_attempts} attempts failed for {func.__name__}")
            raise last_exception
        
        return wrapper
    return decorator


class GracefulDegradation:
    """
    Graceful degradation strategies for photonic systems.
    
    Implements fallback mechanisms when primary systems fail.
    """
    
    def __init__(self):
        """Initialize degradation manager."""
        self.degradation_strategies = {}
        self.active_degradations = set()
        self.performance_impact = {}
        
    def register_fallback(self, component: str, fallback_func: Callable,
                         performance_impact: float = 0.5):
        """
        Register fallback strategy for component.
        
        Args:
            component: Component name
            fallback_func: Fallback function
            performance_impact: Performance impact (0.0-1.0, where 1.0 is no impact)
        """
        self.degradation_strategies[component] = {
            "fallback": fallback_func,
            "performance_impact": performance_impact
        }
        
    def execute_with_fallback(self, component: str, primary_func: Callable,
                            *args, **kwargs) -> Any:
        """
        Execute function with fallback on failure.
        
        Args:
            component: Component name
            primary_func: Primary function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result (from primary or fallback)
        """
        try:
            # Try primary function
            result = primary_func(*args, **kwargs)
            
            # If we were degraded, try to recover
            if component in self.active_degradations:
                self._attempt_recovery(component)
            
            return result
            
        except Exception as e:
            logger.warning(f"Primary function failed for {component}: {e}")
            
            # Try fallback
            if component in self.degradation_strategies:
                self.active_degradations.add(component)
                strategy = self.degradation_strategies[component]
                
                logger.info(f"Activating degraded mode for {component}")
                
                try:
                    return strategy["fallback"](*args, **kwargs)
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed for {component}: {fallback_error}")
                    raise
            else:
                raise  # No fallback available
    
    def _attempt_recovery(self, component: str):
        """Attempt to recover from degraded mode."""
        # Simple recovery attempt - in production would be more sophisticated
        if random.random() < 0.1:  # 10% chance to attempt recovery
            self.active_degradations.discard(component)
            logger.info(f"Recovered from degraded mode for {component}")
    
    def get_degradation_status(self) -> Dict[str, Any]:
        """Get current degradation status."""
        overall_impact = 1.0
        for component in self.active_degradations:
            if component in self.degradation_strategies:
                impact = self.degradation_strategies[component]["performance_impact"]
                overall_impact *= impact
        
        return {
            "degraded_components": list(self.active_degradations),
            "total_degraded": len(self.active_degradations),
            "overall_performance_impact": overall_impact,
            "estimated_performance": f"{overall_impact:.1%}"
        }


class HealthChecker:
    """
    Continuous health monitoring for photonic systems.
    """
    
    def __init__(self, check_interval: float = 30.0):
        """Initialize health checker."""
        self.check_interval = check_interval
        self.health_checks = {}
        self.health_status = {}
        self.is_running = False
        self.monitor_thread = None
        
    def register_health_check(self, component: str, check_func: Callable,
                             critical: bool = False):
        """
        Register health check for component.
        
        Args:
            component: Component name
            check_func: Function that returns True if healthy
            critical: Whether failure is critical to system operation
        """
        self.health_checks[component] = {
            "check_func": check_func,
            "critical": critical,
            "last_check": 0.0,
            "consecutive_failures": 0
        }
        
    def start_monitoring(self):
        """Start continuous health monitoring."""
        if self.is_running:
            return
            
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        logger.info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        self.is_running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        
        logger.info("Health monitoring stopped")
    
    def _monitoring_loop(self):
        """Main health monitoring loop."""
        while self.is_running:
            try:
                self._perform_health_checks()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                time.sleep(5.0)  # Brief pause before retry
    
    def _perform_health_checks(self):
        """Perform all registered health checks."""
        current_time = time.time()
        
        for component, check_info in self.health_checks.items():
            try:
                is_healthy = check_info["check_func"]()
                
                if is_healthy:
                    self.health_status[component] = {
                        "status": "healthy",
                        "last_check": current_time,
                        "consecutive_failures": 0
                    }
                    check_info["consecutive_failures"] = 0
                else:
                    check_info["consecutive_failures"] += 1
                    self.health_status[component] = {
                        "status": "unhealthy",
                        "last_check": current_time,
                        "consecutive_failures": check_info["consecutive_failures"]
                    }
                    
                    # Log health issues
                    if check_info["critical"]:
                        logger.error(f"Critical component {component} is unhealthy")
                    else:
                        logger.warning(f"Component {component} is unhealthy")
                        
            except Exception as e:
                check_info["consecutive_failures"] += 1
                self.health_status[component] = {
                    "status": "error",
                    "last_check": current_time,
                    "consecutive_failures": check_info["consecutive_failures"],
                    "error": str(e)
                }
                logger.error(f"Health check failed for {component}: {e}")
    
    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        if not self.health_status:
            return {"status": "unknown", "components": {}}
        
        healthy_components = sum(1 for status in self.health_status.values() 
                               if status["status"] == "healthy")
        total_components = len(self.health_status)
        
        critical_unhealthy = any(
            status["status"] != "healthy" and 
            self.health_checks[comp]["critical"]
            for comp, status in self.health_status.items()
        )
        
        if critical_unhealthy:
            overall_status = "critical"
        elif healthy_components == total_components:
            overall_status = "healthy"
        elif healthy_components >= total_components * 0.8:
            overall_status = "degraded"
        else:
            overall_status = "unhealthy"
        
        return {
            "status": overall_status,
            "healthy_components": healthy_components,
            "total_components": total_components,
            "health_percentage": healthy_components / total_components * 100,
            "components": self.health_status
        }


class ResilienceManager:
    """
    Comprehensive resilience management for photonic AI systems.
    
    Coordinates circuit breakers, retry mechanisms, graceful degradation,
    and health monitoring for robust system operation.
    """
    
    def __init__(self):
        """Initialize resilience manager."""
        self.circuit_breakers = {}
        self.degradation_manager = GracefulDegradation()
        self.health_checker = HealthChecker()
        self.failure_history = deque(maxlen=1000)
        self.recovery_strategies = {}
        
    def create_circuit_breaker(self, name: str, 
                             config: CircuitBreakerConfig = None) -> CircuitBreaker:
        """Create and register circuit breaker."""
        config = config or CircuitBreakerConfig()
        breaker = CircuitBreaker(name, config)
        self.circuit_breakers[name] = breaker
        return breaker
    
    def register_fallback_strategy(self, component: str, fallback_func: Callable,
                                 performance_impact: float = 0.5):
        """Register fallback strategy."""
        self.degradation_manager.register_fallback(
            component, fallback_func, performance_impact
        )
    
    def register_health_check(self, component: str, check_func: Callable,
                            critical: bool = False):
        """Register health check."""
        self.health_checker.register_health_check(component, check_func, critical)
    
    def start_monitoring(self):
        """Start all monitoring systems."""
        self.health_checker.start_monitoring()
        logger.info("Resilience monitoring started")
    
    def stop_monitoring(self):
        """Stop all monitoring systems."""
        self.health_checker.stop_monitoring()
        logger.info("Resilience monitoring stopped")
    
    def record_failure(self, component: str, failure_type: FailureType,
                      error_message: str, severity: str = "medium"):
        """Record system failure."""
        failure = FailureRecord(
            timestamp=time.time(),
            failure_type=failure_type,
            component=component,
            error_message=error_message,
            severity=severity
        )
        
        self.failure_history.append(failure)
        logger.warning(f"Failure recorded: {component} - {error_message}")
        
        # Trigger recovery if needed
        self._trigger_recovery(failure)
    
    def _trigger_recovery(self, failure: FailureRecord):
        """Trigger appropriate recovery strategies."""
        component = failure.component
        
        if component in self.recovery_strategies:
            strategy = self.recovery_strategies[component]
            try:
                strategy(failure)
                logger.info(f"Recovery strategy executed for {component}")
            except Exception as e:
                logger.error(f"Recovery strategy failed for {component}: {e}")
    
    def get_resilience_status(self) -> Dict[str, Any]:
        """Get comprehensive resilience status."""
        circuit_status = {
            name: breaker.get_status() 
            for name, breaker in self.circuit_breakers.items()
        }
        
        degradation_status = self.degradation_manager.get_degradation_status()
        health_status = self.health_checker.get_overall_health()
        
        recent_failures = [
            f for f in self.failure_history 
            if time.time() - f.timestamp < 3600  # Last hour
        ]
        
        return {
            "overall_health": health_status["status"],
            "circuit_breakers": circuit_status,
            "degradation": degradation_status,
            "health_checks": health_status,
            "recent_failures": len(recent_failures),
            "failure_rate_per_hour": len(recent_failures),
            "resilience_score": self._calculate_resilience_score()
        }
    
    def _calculate_resilience_score(self) -> float:
        """Calculate overall resilience score (0.0-1.0)."""
        score = 1.0
        
        # Factor in circuit breaker states
        for breaker in self.circuit_breakers.values():
            if breaker.state == CircuitState.OPEN:
                score *= 0.7  # 30% penalty for open circuits
            elif breaker.state == CircuitState.HALF_OPEN:
                score *= 0.9  # 10% penalty for recovering circuits
        
        # Factor in degradations
        degradation_impact = self.degradation_manager.get_degradation_status()["overall_performance_impact"]
        score *= degradation_impact
        
        # Factor in health status
        health = self.health_checker.get_overall_health()
        if health["status"] == "critical":
            score *= 0.3
        elif health["status"] == "unhealthy":
            score *= 0.6
        elif health["status"] == "degraded":
            score *= 0.8
        
        return max(0.0, min(1.0, score))


# Custom exceptions
class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


class ResourceExhaustionError(Exception):
    """Raised when system resources are exhausted."""
    pass


def create_resilient_photonic_system(base_system, enable_monitoring: bool = True):
    """
    Create resilience-enhanced photonic system.
    
    Args:
        base_system: Base photonic neural network system
        enable_monitoring: Whether to start monitoring immediately
        
    Returns:
        Resilience-enhanced system
    """
    resilience_manager = ResilienceManager()
    
    # Create circuit breakers for key components
    inference_breaker = resilience_manager.create_circuit_breaker(
        "inference", CircuitBreakerConfig(failure_threshold=3, timeout_seconds=30.0)
    )
    
    training_breaker = resilience_manager.create_circuit_breaker(
        "training", CircuitBreakerConfig(failure_threshold=2, timeout_seconds=60.0)
    )
    
    # Register health checks
    def check_model_health():
        """Check if model is functioning correctly."""
        try:
            # Simple health check - in production would be more comprehensive
            return hasattr(base_system, 'forward') and callable(base_system.forward)
        except Exception:
            return False
    
    resilience_manager.register_health_check("model", check_model_health, critical=True)
    
    # Register fallback strategies
    def fallback_inference(*args, **kwargs):
        """Fallback inference using simplified model."""
        # Simplified fallback - return zero predictions
        import numpy as np
        if args:
            batch_size = args[0].shape[0] if hasattr(args[0], 'shape') else 1
            return np.zeros((batch_size, 10)), {"total_latency_ns": 1000, "total_power_mw": 0}
        return np.zeros((1, 10)), {"total_latency_ns": 1000, "total_power_mw": 0}
    
    resilience_manager.register_fallback_strategy("inference", fallback_inference, 0.3)
    
    if enable_monitoring:
        resilience_manager.start_monitoring()
    
    # Wrap base system methods with resilience
    original_forward = base_system.forward
    
    @with_retry(RetryConfig(max_attempts=2, base_delay=0.5))
    def resilient_forward(*args, **kwargs):
        return inference_breaker.call(original_forward, *args, **kwargs)
    
    base_system.forward = resilient_forward
    base_system.resilience_manager = resilience_manager
    
    return base_system