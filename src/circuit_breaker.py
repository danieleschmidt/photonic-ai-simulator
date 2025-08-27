"""
Circuit Breaker and Health Check System for Photonic AI

Implements fault tolerance patterns including circuit breakers, health checks,
and automatic failover for robust production photonic AI systems.
"""

import time
import threading
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import deque, defaultdict
import asyncio
import functools

try:
    from .robust_error_handling import PhotonicError, ErrorSeverity
    from .utils.monitoring import SystemMetrics
except ImportError:
    from robust_error_handling import PhotonicError, ErrorSeverity
    from utils.monitoring import SystemMetrics

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"           # Normal operation
    OPEN = "open"              # Circuit is open, requests failing
    HALF_OPEN = "half_open"    # Testing if service has recovered


class HealthStatus(Enum):
    """Health check status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5          # Failures needed to open circuit
    recovery_timeout: float = 60.0      # Seconds before trying half-open
    success_threshold: int = 3          # Successes needed to close circuit
    timeout: float = 30.0               # Request timeout in seconds
    monitoring_window: int = 100        # Number of requests to monitor
    failure_rate_threshold: float = 0.5 # Failure rate to open circuit


@dataclass
class HealthCheckConfig:
    """Configuration for health checks."""
    check_interval: float = 30.0        # Seconds between health checks
    timeout: float = 5.0               # Health check timeout
    unhealthy_threshold: int = 3       # Failed checks to mark unhealthy
    recovery_threshold: int = 2        # Successful checks to mark healthy


@dataclass
class CircuitMetrics:
    """Circuit breaker metrics."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rejected_requests: int = 0
    current_state: CircuitState = CircuitState.CLOSED
    last_failure_time: Optional[float] = None
    failure_rate: float = 0.0
    avg_response_time: float = 0.0


class CircuitBreaker:
    """
    Circuit breaker implementation for photonic AI systems.
    
    Provides automatic fault tolerance and prevents cascading failures
    in distributed photonic neural network deployments.
    """
    
    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        """Initialize circuit breaker."""
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.request_history = deque(maxlen=self.config.monitoring_window)
        self.metrics = CircuitMetrics()
        self._lock = threading.Lock()
        
        logger.info(f"Circuit breaker '{name}' initialized with config: {self.config}")
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap functions with circuit breaker."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args, **kwargs: Function arguments
            
        Returns:
            Function result or raises CircuitBreakerError
        """
        with self._lock:
            # Check if circuit should reject request
            if self._should_reject_request():
                self.metrics.rejected_requests += 1
                raise CircuitBreakerError(
                    f"Circuit breaker '{self.name}' is {self.state.value}",
                    ErrorSeverity.HIGH,
                    self.name
                )
            
            self.metrics.total_requests += 1
        
        # Execute the function
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Record success
            self._record_success(execution_time)
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Record failure
            self._record_failure(e, execution_time)
            raise
    
    def _should_reject_request(self) -> bool:
        """Check if request should be rejected based on circuit state."""
        current_time = time.time()
        
        if self.state == CircuitState.OPEN:
            # Check if we should try half-open
            if (self.last_failure_time and 
                current_time - self.last_failure_time >= self.config.recovery_timeout):
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                logger.info(f"Circuit breaker '{self.name}' moving to HALF_OPEN state")
                return False
            return True
        
        return False
    
    def _record_success(self, execution_time: float):
        """Record successful request."""
        with self._lock:
            self.request_history.append({
                'timestamp': time.time(),
                'success': True,
                'execution_time': execution_time
            })
            
            self.metrics.successful_requests += 1
            
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    logger.info(f"Circuit breaker '{self.name}' moving to CLOSED state")
            
            self._update_metrics()
    
    def _record_failure(self, error: Exception, execution_time: float):
        """Record failed request."""
        with self._lock:
            self.request_history.append({
                'timestamp': time.time(),
                'success': False,
                'execution_time': execution_time,
                'error': str(error)
            })
            
            self.metrics.failed_requests += 1
            self.failure_count += 1
            self.last_failure_time = time.time()
            self.metrics.last_failure_time = self.last_failure_time
            
            # Check if we should open the circuit
            if self.state == CircuitState.CLOSED:
                failure_rate = self._calculate_failure_rate()
                if (self.failure_count >= self.config.failure_threshold or 
                    failure_rate >= self.config.failure_rate_threshold):
                    self.state = CircuitState.OPEN
                    logger.warning(f"Circuit breaker '{self.name}' OPENED due to failures")
            
            elif self.state == CircuitState.HALF_OPEN:
                # Any failure in half-open state reopens circuit
                self.state = CircuitState.OPEN
                logger.warning(f"Circuit breaker '{self.name}' reopened from HALF_OPEN")
            
            self._update_metrics()
    
    def _calculate_failure_rate(self) -> float:
        """Calculate current failure rate."""
        if not self.request_history:
            return 0.0
        
        recent_requests = list(self.request_history)[-50:]  # Last 50 requests
        if not recent_requests:
            return 0.0
        
        failures = sum(1 for req in recent_requests if not req['success'])
        return failures / len(recent_requests)
    
    def _update_metrics(self):
        """Update internal metrics."""
        self.metrics.current_state = self.state
        self.metrics.failure_rate = self._calculate_failure_rate()
        
        # Calculate average response time
        if self.request_history:
            recent_times = [req['execution_time'] for req in list(self.request_history)[-10:]]
            self.metrics.avg_response_time = sum(recent_times) / len(recent_times)
    
    def get_metrics(self) -> CircuitMetrics:
        """Get current circuit breaker metrics."""
        with self._lock:
            return self.metrics
    
    def reset(self):
        """Reset circuit breaker to closed state."""
        with self._lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.request_history.clear()
            self.metrics = CircuitMetrics()
            logger.info(f"Circuit breaker '{self.name}' manually reset")


class CircuitBreakerError(PhotonicError):
    """Circuit breaker rejection error."""
    pass


class HealthCheck:
    """Individual health check for a system component."""
    
    def __init__(self, name: str, check_func: Callable[[], bool], 
                 config: HealthCheckConfig = None):
        """
        Initialize health check.
        
        Args:
            name: Health check name
            check_func: Function that returns True if healthy
            config: Health check configuration
        """
        self.name = name
        self.check_func = check_func
        self.config = config or HealthCheckConfig()
        self.status = HealthStatus.UNKNOWN
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        self.last_check_time = None
        self.last_error = None
        
    def execute_check(self) -> bool:
        """Execute the health check."""
        self.last_check_time = time.time()
        
        try:
            # Execute health check with timeout
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("Health check timed out")
            
            # Set timeout (Unix systems only)
            if hasattr(signal, 'SIGALRM'):
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(int(self.config.timeout))
            
            try:
                result = self.check_func()
                
                if hasattr(signal, 'SIGALRM'):
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
                
                if result:
                    self._record_success()
                    return True
                else:
                    self._record_failure("Health check returned False")
                    return False
                    
            except Exception as e:
                if hasattr(signal, 'SIGALRM'):
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
                
                self._record_failure(f"Health check error: {e}")
                return False
                
        except Exception as e:
            self._record_failure(f"Health check execution error: {e}")
            return False
    
    def _record_success(self):
        """Record successful health check."""
        self.consecutive_failures = 0
        self.consecutive_successes += 1
        self.last_error = None
        
        if self.consecutive_successes >= self.config.recovery_threshold:
            if self.status != HealthStatus.HEALTHY:
                logger.info(f"Health check '{self.name}' is now HEALTHY")
            self.status = HealthStatus.HEALTHY
    
    def _record_failure(self, error: str):
        """Record failed health check."""
        self.consecutive_successes = 0
        self.consecutive_failures += 1
        self.last_error = error
        
        if self.consecutive_failures >= self.config.unhealthy_threshold:
            if self.status != HealthStatus.UNHEALTHY:
                logger.error(f"Health check '{self.name}' is now UNHEALTHY: {error}")
            self.status = HealthStatus.UNHEALTHY
        elif self.status == HealthStatus.HEALTHY:
            self.status = HealthStatus.DEGRADED
            logger.warning(f"Health check '{self.name}' is DEGRADED: {error}")


class HealthMonitor:
    """
    Comprehensive health monitoring system for photonic AI.
    
    Manages multiple health checks and provides system-wide health status.
    """
    
    def __init__(self):
        """Initialize health monitor."""
        self.health_checks: Dict[str, HealthCheck] = {}
        self.system_metrics = None
        self.is_monitoring = False
        self.monitor_task = None
        self._lock = threading.Lock()
        
    def register_health_check(self, check: HealthCheck):
        """Register a health check."""
        with self._lock:
            self.health_checks[check.name] = check
            logger.info(f"Registered health check: {check.name}")
    
    def add_database_health_check(self, db_connection_func: Callable[[], bool]):
        """Add database connectivity health check."""
        def db_check():
            return db_connection_func()
        
        check = HealthCheck("database", db_check)
        self.register_health_check(check)
    
    def add_memory_health_check(self, max_memory_mb: float = 8192):
        """Add memory usage health check."""
        def memory_check():
            import psutil
            memory_mb = psutil.virtual_memory().used / (1024 * 1024)
            return memory_mb < max_memory_mb
        
        check = HealthCheck("memory", memory_check)
        self.register_health_check(check)
    
    def add_disk_health_check(self, max_disk_usage_percent: float = 90.0):
        """Add disk usage health check."""
        def disk_check():
            import psutil
            disk_usage = psutil.disk_usage('/').percent
            return disk_usage < max_disk_usage_percent
        
        check = HealthCheck("disk", disk_check)
        self.register_health_check(check)
    
    def add_photonic_system_health_check(self, photonic_network):
        """Add photonic neural network health check."""
        def photonic_check():
            try:
                # Test basic inference
                import numpy as np
                test_input = np.random.randn(1, 784)  # MNIST-sized input
                outputs, metrics = photonic_network.optimized_forward(test_input)
                
                # Check if outputs are reasonable
                if outputs is None or np.any(np.isnan(outputs)):
                    return False
                
                # Check latency is reasonable (< 100ms)
                if metrics.get('total_latency_ns', 0) > 100_000_000:
                    return False
                
                return True
            except Exception:
                return False
        
        check = HealthCheck("photonic_network", photonic_check)
        self.register_health_check(check)
    
    async def start_monitoring(self):
        """Start continuous health monitoring."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        logger.info("Starting health monitoring")
        
        while self.is_monitoring:
            await self._run_health_checks()
            await asyncio.sleep(30)  # Check every 30 seconds
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        self.is_monitoring = False
        logger.info("Stopping health monitoring")
    
    async def _run_health_checks(self):
        """Run all registered health checks."""
        for name, check in self.health_checks.items():
            try:
                check.execute_check()
            except Exception as e:
                logger.error(f"Error executing health check '{name}': {e}")
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status."""
        with self._lock:
            health_statuses = {}
            overall_status = HealthStatus.HEALTHY
            
            for name, check in self.health_checks.items():
                health_statuses[name] = {
                    "status": check.status.value,
                    "last_check": check.last_check_time,
                    "consecutive_failures": check.consecutive_failures,
                    "last_error": check.last_error
                }
                
                # Determine overall status
                if check.status == HealthStatus.UNHEALTHY:
                    overall_status = HealthStatus.UNHEALTHY
                elif (check.status == HealthStatus.DEGRADED and 
                      overall_status == HealthStatus.HEALTHY):
                    overall_status = HealthStatus.DEGRADED
            
            return {
                "overall_status": overall_status.value,
                "checks": health_statuses,
                "healthy_checks": sum(1 for c in self.health_checks.values() 
                                    if c.status == HealthStatus.HEALTHY),
                "total_checks": len(self.health_checks),
                "timestamp": time.time()
            }
    
    def is_healthy(self) -> bool:
        """Check if system is healthy overall."""
        health = self.get_system_health()
        return health["overall_status"] == HealthStatus.HEALTHY.value


class FaultToleranceManager:
    """
    Manages fault tolerance for photonic AI systems.
    
    Coordinates circuit breakers, health checks, and automatic recovery.
    """
    
    def __init__(self):
        """Initialize fault tolerance manager."""
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.health_monitor = HealthMonitor()
        self.failover_handlers: Dict[str, Callable] = {}
        self.is_active = False
        
    def create_circuit_breaker(self, name: str, 
                             config: CircuitBreakerConfig = None) -> CircuitBreaker:
        """Create and register a circuit breaker."""
        breaker = CircuitBreaker(name, config)
        self.circuit_breakers[name] = breaker
        return breaker
    
    def register_failover_handler(self, component: str, handler: Callable):
        """Register failover handler for a component."""
        self.failover_handlers[component] = handler
        logger.info(f"Registered failover handler for: {component}")
    
    async def start_fault_tolerance(self):
        """Start fault tolerance monitoring."""
        if self.is_active:
            return
        
        self.is_active = True
        logger.info("Starting fault tolerance management")
        
        # Start health monitoring
        await self.health_monitor.start_monitoring()
    
    def stop_fault_tolerance(self):
        """Stop fault tolerance monitoring."""
        self.is_active = False
        self.health_monitor.stop_monitoring()
        logger.info("Stopped fault tolerance management")
    
    def get_fault_tolerance_status(self) -> Dict[str, Any]:
        """Get comprehensive fault tolerance status."""
        circuit_status = {}
        for name, breaker in self.circuit_breakers.items():
            metrics = breaker.get_metrics()
            circuit_status[name] = {
                "state": metrics.current_state.value,
                "total_requests": metrics.total_requests,
                "failure_rate": metrics.failure_rate,
                "rejected_requests": metrics.rejected_requests
            }
        
        return {
            "health_status": self.health_monitor.get_system_health(),
            "circuit_breakers": circuit_status,
            "is_active": self.is_active,
            "timestamp": time.time()
        }


# Factory functions
def create_circuit_breaker(name: str, 
                          failure_threshold: int = 5,
                          recovery_timeout: float = 60.0,
                          **kwargs) -> CircuitBreaker:
    """Create a circuit breaker with specified configuration."""
    config = CircuitBreakerConfig(
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
        **kwargs
    )
    return CircuitBreaker(name, config)


def create_fault_tolerance_manager() -> FaultToleranceManager:
    """Create a fault tolerance manager for photonic AI systems."""
    return FaultToleranceManager()