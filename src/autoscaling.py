"""
Auto-scaling and resource management for photonic AI systems.

Implements intelligent scaling decisions, resource allocation,
and distributed processing for production photonic neural networks.
"""

import asyncio
import logging
import time
import threading
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import json
import math
import statistics

logger = logging.getLogger(__name__)


class ScalingAction(Enum):
    """Types of scaling actions."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    SCALE_OUT = "scale_out"
    SCALE_IN = "scale_in"
    NO_ACTION = "no_action"


class ResourceType(Enum):
    """Types of system resources."""
    cpu = "cpu"
    memory = "memory"
    gpu = "gpu"
    network = "network"
    storage = "storage"
    photonic_cores = "photonic_cores"


class WorkloadPattern(Enum):
    """Workload patterns for prediction."""
    steady = "steady"
    periodic = "periodic"
    burst = "burst"
    trending = "trending"
    random = "random"


@dataclass
class ResourceMetrics:
    """System resource utilization metrics."""
    timestamp: float
    cpu_utilization: float
    memory_utilization: float
    gpu_utilization: float
    network_throughput: float
    storage_iops: float
    photonic_core_utilization: float
    request_rate: float
    response_latency: float
    error_rate: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "cpu_utilization": self.cpu_utilization,
            "memory_utilization": self.memory_utilization,
            "gpu_utilization": self.gpu_utilization,
            "network_throughput": self.network_throughput,
            "storage_iops": self.storage_iops,
            "photonic_core_utilization": self.photonic_core_utilization,
            "request_rate": self.request_rate,
            "response_latency": self.response_latency,
            "error_rate": self.error_rate
        }


@dataclass
class ScalingPolicy:
    """Auto-scaling policy configuration."""
    name: str
    resource_type: ResourceType
    target_utilization: float = 0.7
    scale_up_threshold: float = 0.8
    scale_down_threshold: float = 0.3
    cooldown_period_seconds: float = 300.0
    min_instances: int = 1
    max_instances: int = 10
    scale_up_increment: int = 1
    scale_down_increment: int = 1
    
    # Advanced settings
    prediction_window_seconds: float = 900.0  # 15 minutes
    confidence_threshold: float = 0.8
    enable_predictive_scaling: bool = True
    enable_cost_optimization: bool = True


@dataclass
class ScalingDecision:
    """Scaling decision with reasoning."""
    timestamp: float
    action: ScalingAction
    resource_type: ResourceType
    current_instances: int
    target_instances: int
    confidence: float
    reasoning: str
    predicted_impact: Dict[str, float]
    cost_impact: Optional[float] = None


class WorkloadPredictor:
    """
    Workload prediction for proactive scaling.
    
    Uses time series analysis and pattern recognition
    to predict future resource demands.
    """
    
    def __init__(self, history_window: int = 1000):
        """Initialize workload predictor."""
        self.history_window = history_window
        self.metrics_history = deque(maxlen=history_window)
        self.patterns = {}
        self.prediction_accuracy = {}
        
    def add_metrics(self, metrics: ResourceMetrics):
        """Add new metrics to history."""
        self.metrics_history.append(metrics)
        
        # Update pattern analysis periodically
        if len(self.metrics_history) >= 100 and len(self.metrics_history) % 50 == 0:
            self._analyze_patterns()
    
    def predict_demand(self, resource_type: ResourceType, 
                      prediction_horizon_seconds: float) -> Tuple[float, float]:
        """
        Predict resource demand for given horizon.
        
        Args:
            resource_type: Resource to predict
            prediction_horizon_seconds: How far ahead to predict
            
        Returns:
            Tuple of (predicted_utilization, confidence)
        """
        if len(self.metrics_history) < 10:
            # Not enough data for prediction
            return 0.5, 0.1
        
        # Get recent values for the resource
        recent_values = self._get_resource_values(resource_type, lookback_count=50)
        
        if not recent_values:
            return 0.5, 0.1
        
        # Detect pattern type
        pattern = self._detect_pattern(recent_values)
        
        # Make prediction based on pattern
        if pattern == WorkloadPattern.steady:
            prediction = statistics.mean(recent_values[-10:])
            confidence = 0.8
        
        elif pattern == WorkloadPattern.periodic:
            prediction = self._predict_periodic(recent_values, prediction_horizon_seconds)
            confidence = 0.7
        
        elif pattern == WorkloadPattern.trending:
            prediction = self._predict_trending(recent_values)
            confidence = 0.6
        
        elif pattern == WorkloadPattern.burst:
            prediction = self._predict_burst(recent_values)
            confidence = 0.4
        
        else:  # RANDOM
            prediction = statistics.mean(recent_values)
            confidence = 0.2
        
        # Clamp prediction to valid range
        prediction = max(0.0, min(1.0, prediction))
        
        return prediction, confidence
    
    def _get_resource_values(self, resource_type: ResourceType, 
                           lookback_count: int) -> List[float]:
        """Get recent values for specific resource type."""
        values = []
        for metrics in list(self.metrics_history)[-lookback_count:]:
            if resource_type == ResourceType.cpu:
                values.append(metrics.cpu_utilization)
            elif resource_type == ResourceType.memory:
                values.append(metrics.memory_utilization)
            elif resource_type == ResourceType.gpu:
                values.append(metrics.gpu_utilization)
            elif resource_type == ResourceType.photonic_cores:
                values.append(metrics.photonic_core_utilization)
            else:
                values.append(0.5)  # Default fallback
        return values
    
    def _detect_pattern(self, values: List[float]) -> WorkloadPattern:
        """Detect workload pattern from time series."""
        if len(values) < 10:
            return WorkloadPattern.random
        
        # Calculate statistics
        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values) if len(values) > 1 else 0
        
        # Check for steady pattern (low variance)
        if std_val < 0.1:
            return WorkloadPattern.steady
        
        # Check for trending pattern
        first_half_mean = statistics.mean(values[:len(values)//2])
        second_half_mean = statistics.mean(values[len(values)//2:])
        
        if abs(second_half_mean - first_half_mean) > 0.2:
            return WorkloadPattern.trending
        
        # Check for periodic pattern using autocorrelation
        if self._detect_periodicity(values):
            return WorkloadPattern.periodic
        
        # Check for burst pattern (occasional high spikes)
        high_values = [v for v in values if v > mean_val + 2 * std_val]
        if len(high_values) > 0 and len(high_values) < len(values) * 0.2:
            return WorkloadPattern.burst
        
        return WorkloadPattern.random
    
    def _detect_periodicity(self, values: List[float]) -> bool:
        """Detect if values show periodic behavior."""
        # Simple periodicity detection using autocorrelation
        # In production, would use FFT or more sophisticated methods
        
        if len(values) < 20:
            return False
        
        # Check for patterns at common intervals (5, 10, 15, 30 minutes)
        common_periods = [5, 10, 15, 30]
        
        for period in common_periods:
            if period >= len(values):
                continue
            
            correlation = 0.0
            for i in range(len(values) - period):
                correlation += values[i] * values[i + period]
            
            correlation /= (len(values) - period)
            
            if correlation > 0.7:  # Strong correlation
                return True
        
        return False
    
    def _predict_periodic(self, values: List[float], 
                         horizon_seconds: float) -> float:
        """Predict based on periodic pattern."""
        # Find the period and predict based on cycle
        # Simplified implementation
        period_length = 15  # Assume 15-sample period
        
        if len(values) >= period_length:
            cycle_position = len(values) % period_length
            return values[-period_length + cycle_position]
        
        return statistics.mean(values)
    
    def _predict_trending(self, values: List[float]) -> float:
        """Predict based on trending pattern."""
        # Simple linear trend extrapolation
        if len(values) < 2:
            return values[-1] if values else 0.5
        
        # Calculate trend slope
        x_values = list(range(len(values)))
        n = len(values)
        
        sum_x = sum(x_values)
        sum_y = sum(values)
        sum_xy = sum(x * y for x, y in zip(x_values, values))
        sum_x_squared = sum(x * x for x in x_values)
        
        # Linear regression slope
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x * sum_x)
        intercept = (sum_y - slope * sum_x) / n
        
        # Predict next value
        next_x = len(values)
        prediction = slope * next_x + intercept
        
        return prediction
    
    def _predict_burst(self, values: List[float]) -> float:
        """Predict based on burst pattern."""
        # For burst patterns, predict conservatively but account for spikes
        baseline = statistics.median(values)
        max_spike = max(values)
        
        # Return value between baseline and spike based on recent activity
        recent_activity = statistics.mean(values[-5:])
        
        if recent_activity > baseline * 1.5:
            # Recent high activity, predict continued elevation
            return min(max_spike, recent_activity * 1.2)
        else:
            # Predict return to baseline with some buffer
            return baseline * 1.1
    
    def _analyze_patterns(self):
        """Analyze historical patterns for improved prediction."""
        # Analyze different time windows for pattern recognition
        # This would be expanded in production with ML models
        pass


class AutoScaler:
    """
    Intelligent auto-scaling system for photonic AI workloads.
    
    Implements predictive scaling, cost optimization, and
    adaptive resource management.
    """
    
    def __init__(self):
        """Initialize auto-scaler."""
        self.policies = {}
        self.predictor = WorkloadPredictor()
        self.current_instances = defaultdict(int)
        self.scaling_history = deque(maxlen=1000)
        self.last_scaling_time = defaultdict(float)
        
        # Performance tracking
        self.scaling_effectiveness = {}
        self.cost_savings = 0.0
        
        # Threading
        self.is_running = False
        self.scaling_thread = None
        self.scaling_interval = 60.0  # Check every minute
        
    def add_scaling_policy(self, policy: ScalingPolicy):
        """Add scaling policy for resource type."""
        self.policies[policy.resource_type] = policy
        self.current_instances[policy.resource_type] = policy.min_instances
        
        logger.info(f"Added scaling policy for {policy.resource_type.value}: "
                   f"target={policy.target_utilization}, "
                   f"range=[{policy.min_instances}, {policy.max_instances}]")
    
    def start_auto_scaling(self):
        """Start auto-scaling monitoring."""
        if self.is_running:
            return
        
        self.is_running = True
        self.scaling_thread = threading.Thread(target=self._scaling_loop)
        self.scaling_thread.daemon = True
        self.scaling_thread.start()
        
        logger.info("Auto-scaling started")
    
    def stop_auto_scaling(self):
        """Stop auto-scaling monitoring."""
        self.is_running = False
        if self.scaling_thread:
            self.scaling_thread.join(timeout=5.0)
        
        logger.info("Auto-scaling stopped")
    
    def process_metrics(self, metrics: ResourceMetrics) -> List[ScalingDecision]:
        """
        Process new metrics and make scaling decisions.
        
        Args:
            metrics: Current resource metrics
            
        Returns:
            List of scaling decisions
        """
        # Add metrics to predictor
        self.predictor.add_metrics(metrics)
        
        decisions = []
        current_time = time.time()
        
        for resource_type, policy in self.policies.items():
            # Check cooldown period
            last_scaling = self.last_scaling_time.get(resource_type, 0)
            if current_time - last_scaling < policy.cooldown_period_seconds:
                continue
            
            # Get current utilization
            current_utilization = self._get_current_utilization(metrics, resource_type)
            current_instances = self.current_instances[resource_type]
            
            # Make scaling decision
            decision = self._make_scaling_decision(
                resource_type, policy, current_utilization, current_instances
            )
            
            if decision.action != ScalingAction.NO_ACTION:
                decisions.append(decision)
                self.scaling_history.append(decision)
                self.last_scaling_time[resource_type] = current_time
                
                # Update instance count
                self.current_instances[resource_type] = decision.target_instances
                
                logger.info(f"Scaling decision: {decision.action.value} "
                           f"{resource_type.value} from {decision.current_instances} "
                           f"to {decision.target_instances} (confidence: {decision.confidence:.2f})")
        
        return decisions
    
    def _scaling_loop(self):
        """Main auto-scaling monitoring loop."""
        while self.is_running:
            try:
                # In production, would collect real metrics
                # For now, simulate with reasonable values
                simulated_metrics = self._simulate_metrics()
                self.process_metrics(simulated_metrics)
                
                time.sleep(self.scaling_interval)
                
            except Exception as e:
                logger.error(f"Auto-scaling loop error: {e}")
                time.sleep(10.0)  # Brief pause before retry
    
    def _make_scaling_decision(self, resource_type: ResourceType,
                             policy: ScalingPolicy, current_utilization: float,
                             current_instances: int) -> ScalingDecision:
        """Make scaling decision for resource type."""
        current_time = time.time()
        
        # Get prediction if enabled
        predicted_utilization = current_utilization
        confidence = 0.5
        
        if policy.enable_predictive_scaling:
            predicted_utilization, prediction_confidence = self.predictor.predict_demand(
                resource_type, policy.prediction_window_seconds
            )
            
            # Combine current and predicted utilization
            weight = prediction_confidence if prediction_confidence > policy.confidence_threshold else 0.3
            effective_utilization = (1 - weight) * current_utilization + weight * predicted_utilization
            confidence = (confidence + prediction_confidence) / 2
        else:
            effective_utilization = current_utilization
        
        # Determine scaling action
        action = ScalingAction.NO_ACTION
        target_instances = current_instances
        reasoning = f"Current utilization: {current_utilization:.2f}, Effective: {effective_utilization:.2f}"
        
        if effective_utilization > policy.scale_up_threshold:
            if current_instances < policy.max_instances:
                action = ScalingAction.SCALE_OUT
                target_instances = min(
                    current_instances + policy.scale_up_increment,
                    policy.max_instances
                )
                reasoning += f" - Scaling up due to high utilization"
            else:
                reasoning += f" - At maximum instances"
        
        elif effective_utilization < policy.scale_down_threshold:
            if current_instances > policy.min_instances:
                action = ScalingAction.SCALE_IN
                target_instances = max(
                    current_instances - policy.scale_down_increment,
                    policy.min_instances
                )
                reasoning += f" - Scaling down due to low utilization"
            else:
                reasoning += f" - At minimum instances"
        
        # Calculate predicted impact
        predicted_impact = self._calculate_impact(
            resource_type, current_instances, target_instances, effective_utilization
        )
        
        # Calculate cost impact if enabled
        cost_impact = None
        if policy.enable_cost_optimization:
            cost_impact = self._calculate_cost_impact(
                resource_type, current_instances, target_instances
            )
        
        return ScalingDecision(
            timestamp=current_time,
            action=action,
            resource_type=resource_type,
            current_instances=current_instances,
            target_instances=target_instances,
            confidence=confidence,
            reasoning=reasoning,
            predicted_impact=predicted_impact,
            cost_impact=cost_impact
        )
    
    def _get_current_utilization(self, metrics: ResourceMetrics, 
                               resource_type: ResourceType) -> float:
        """Get current utilization for resource type."""
        if resource_type == ResourceType.cpu:
            return metrics.cpu_utilization
        elif resource_type == ResourceType.memory:
            return metrics.memory_utilization
        elif resource_type == ResourceType.gpu:
            return metrics.gpu_utilization
        elif resource_type == ResourceType.photonic_cores:
            return metrics.photonic_core_utilization
        else:
            return 0.5  # Default fallback
    
    def _calculate_impact(self, resource_type: ResourceType,
                        current_instances: int, target_instances: int,
                        current_utilization: float) -> Dict[str, float]:
        """Calculate predicted impact of scaling decision."""
        if target_instances == current_instances:
            return {"latency_change": 0.0, "throughput_change": 0.0}
        
        scaling_factor = target_instances / current_instances
        
        # Simple impact model - would be more sophisticated in production
        if scaling_factor > 1:  # Scaling up
            latency_improvement = -(1 - 1/scaling_factor) * 100  # Negative = improvement
            throughput_increase = (scaling_factor - 1) * 100
        else:  # Scaling down
            latency_degradation = (1/scaling_factor - 1) * 100  # Positive = degradation
            throughput_decrease = -(1 - scaling_factor) * 100  # Negative = decrease
            latency_improvement = latency_degradation
            throughput_increase = throughput_decrease
        
        return {
            "latency_change_percent": latency_improvement,
            "throughput_change_percent": throughput_increase,
            "utilization_change": -((target_instances - current_instances) / target_instances) * current_utilization
        }
    
    def _calculate_cost_impact(self, resource_type: ResourceType,
                             current_instances: int, target_instances: int) -> float:
        """Calculate cost impact of scaling decision."""
        # Simple cost model - would use real pricing in production
        cost_per_instance_per_hour = {
            ResourceType.cpu: 0.50,
            ResourceType.memory: 0.10,
            ResourceType.gpu: 2.00,
            ResourceType.photonic_cores: 1.50
        }
        
        base_cost = cost_per_instance_per_hour.get(resource_type, 1.0)
        cost_change = (target_instances - current_instances) * base_cost
        
        return cost_change
    
    def _simulate_metrics(self) -> ResourceMetrics:
        """Simulate realistic metrics for demonstration."""
        # Generate realistic metrics with some patterns
        current_time = time.time()
        
        # Simulate daily pattern
        hour_of_day = (current_time % 86400) / 3600
        daily_factor = 0.5 + 0.4 * math.sin((hour_of_day - 6) * math.pi / 12)
        
        # Add some randomness
        import random
        noise = random.uniform(0.8, 1.2)
        
        base_utilization = daily_factor * noise
        
        return ResourceMetrics(
            timestamp=current_time,
            cpu_utilization=min(1.0, base_utilization + random.uniform(-0.1, 0.1)),
            memory_utilization=min(1.0, base_utilization * 0.8 + random.uniform(-0.05, 0.05)),
            gpu_utilization=min(1.0, base_utilization * 1.2 + random.uniform(-0.2, 0.2)),
            network_throughput=base_utilization * 1000 + random.uniform(-100, 100),
            storage_iops=base_utilization * 500 + random.uniform(-50, 50),
            photonic_core_utilization=min(1.0, base_utilization + random.uniform(-0.15, 0.15)),
            request_rate=base_utilization * 100 + random.uniform(-10, 10),
            response_latency=1.0 / max(0.1, base_utilization) + random.uniform(-0.5, 0.5),
            error_rate=max(0.0, (1 - base_utilization) * 0.05 + random.uniform(-0.01, 0.01))
        )
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get comprehensive auto-scaling status."""
        recent_decisions = list(self.scaling_history)[-10:] if self.scaling_history else []
        
        # Calculate scaling effectiveness
        total_decisions = len(self.scaling_history)
        successful_scalings = sum(1 for d in self.scaling_history if d.confidence > 0.6)
        effectiveness = (successful_scalings / total_decisions * 100) if total_decisions > 0 else 0
        
        return {
            "is_running": self.is_running,
            "active_policies": len(self.policies),
            "current_instances": dict(self.current_instances),
            "total_scaling_decisions": total_decisions,
            "scaling_effectiveness": effectiveness,
            "recent_decisions": [
                {
                    "timestamp": d.timestamp,
                    "action": d.action.value,
                    "resource": d.resource_type.value,
                    "instances": f"{d.current_instances} -> {d.target_instances}",
                    "confidence": d.confidence
                }
                for d in recent_decisions
            ],
            "cost_impact_total": sum(
                d.cost_impact for d in self.scaling_history
                if d.cost_impact is not None
            ),
            "predictor_patterns": self.predictor.patterns
        }


class LoadBalancer:
    """
    Intelligent load balancer for distributed photonic AI systems.
    
    Distributes inference requests across multiple instances
    with latency-aware routing and health monitoring.
    """
    
    def __init__(self):
        """Initialize load balancer."""
        self.instances = {}
        self.routing_algorithm = "least_latency"
        self.health_check_interval = 30.0
        self.request_history = deque(maxlen=10000)
        
    def register_instance(self, instance_id: str, endpoint: str, 
                         capacity: float = 1.0):
        """Register new instance for load balancing."""
        self.instances[instance_id] = {
            "endpoint": endpoint,
            "capacity": capacity,
            "current_load": 0.0,
            "avg_latency": 0.0,
            "health_status": "healthy",
            "last_health_check": time.time(),
            "request_count": 0,
            "error_count": 0
        }
        
        logger.info(f"Registered instance {instance_id} with capacity {capacity}")
    
    def route_request(self, request_data: Any) -> str:
        """
        Route request to optimal instance.
        
        Args:
            request_data: Request data for routing decisions
            
        Returns:
            Selected instance ID
        """
        healthy_instances = {
            k: v for k, v in self.instances.items()
            if v["health_status"] == "healthy"
        }
        
        if not healthy_instances:
            raise RuntimeError("No healthy instances available")
        
        # Select instance based on routing algorithm
        if self.routing_algorithm == "round_robin":
            selected_id = self._round_robin_selection(healthy_instances)
        elif self.routing_algorithm == "least_connections":
            selected_id = self._least_connections_selection(healthy_instances)
        elif self.routing_algorithm == "least_latency":
            selected_id = self._least_latency_selection(healthy_instances)
        elif self.routing_algorithm == "weighted_capacity":
            selected_id = self._weighted_capacity_selection(healthy_instances)
        else:
            # Default to least latency
            selected_id = self._least_latency_selection(healthy_instances)
        
        # Update instance load
        self.instances[selected_id]["current_load"] += 1
        
        # Record routing decision
        self.request_history.append({
            "timestamp": time.time(),
            "instance_id": selected_id,
            "routing_algorithm": self.routing_algorithm
        })
        
        return selected_id
    
    def _round_robin_selection(self, instances: Dict[str, Any]) -> str:
        """Select instance using round-robin algorithm."""
        # Simple round-robin based on request count
        return min(instances.keys(), key=lambda k: instances[k]["request_count"])
    
    def _least_connections_selection(self, instances: Dict[str, Any]) -> str:
        """Select instance with least current connections."""
        return min(instances.keys(), key=lambda k: instances[k]["current_load"])
    
    def _least_latency_selection(self, instances: Dict[str, Any]) -> str:
        """Select instance with lowest average latency."""
        return min(instances.keys(), key=lambda k: instances[k]["avg_latency"])
    
    def _weighted_capacity_selection(self, instances: Dict[str, Any]) -> str:
        """Select instance based on capacity-weighted load."""
        def load_ratio(instance_id):
            instance = instances[instance_id]
            return instance["current_load"] / instance["capacity"]
        
        return min(instances.keys(), key=load_ratio)
    
    def update_instance_metrics(self, instance_id: str, latency: float,
                              success: bool = True):
        """Update instance performance metrics."""
        if instance_id not in self.instances:
            return
        
        instance = self.instances[instance_id]
        
        # Update latency (exponential moving average)
        alpha = 0.1  # Smoothing factor
        instance["avg_latency"] = (
            alpha * latency + (1 - alpha) * instance["avg_latency"]
        )
        
        # Update counters
        instance["request_count"] += 1
        instance["current_load"] = max(0, instance["current_load"] - 1)
        
        if not success:
            instance["error_count"] += 1
            
            # Mark as unhealthy if error rate too high
            error_rate = instance["error_count"] / instance["request_count"]
            if error_rate > 0.1:  # 10% error rate threshold
                instance["health_status"] = "unhealthy"
                logger.warning(f"Instance {instance_id} marked unhealthy due to high error rate")
    
    def get_load_balancer_status(self) -> Dict[str, Any]:
        """Get load balancer status and statistics."""
        total_requests = len(self.request_history)
        recent_requests = [
            r for r in self.request_history
            if time.time() - r["timestamp"] < 3600  # Last hour
        ]
        
        # Calculate distribution
        distribution = defaultdict(int)
        for request in recent_requests:
            distribution[request["instance_id"]] += 1
        
        return {
            "routing_algorithm": self.routing_algorithm,
            "total_instances": len(self.instances),
            "healthy_instances": sum(
                1 for i in self.instances.values()
                if i["health_status"] == "healthy"
            ),
            "total_requests": total_requests,
            "recent_requests": len(recent_requests),
            "request_distribution": dict(distribution),
            "instance_status": {
                k: {
                    "health": v["health_status"],
                    "load": v["current_load"],
                    "latency": v["avg_latency"],
                    "requests": v["request_count"],
                    "errors": v["error_count"]
                }
                for k, v in self.instances.items()
            }
        }


def create_auto_scaling_photonic_system(base_system, 
                                       enable_predictive_scaling: bool = True,
                                       enable_load_balancing: bool = True):
    """
    Create auto-scaling enhanced photonic system.
    
    Args:
        base_system: Base photonic neural network system
        enable_predictive_scaling: Enable predictive scaling
        enable_load_balancing: Enable load balancing
        
    Returns:
        Auto-scaling enhanced system
    """
    # Create auto-scaler
    auto_scaler = AutoScaler()
    
    # Add default scaling policies
    cpu_policy = ScalingPolicy(
        name="cpu_scaling",
        resource_type=ResourceType.cpu,
        target_utilization=0.7,
        scale_up_threshold=0.8,
        scale_down_threshold=0.3,
        enable_predictive_scaling=enable_predictive_scaling
    )
    
    gpu_policy = ScalingPolicy(
        name="gpu_scaling",
        resource_type=ResourceType.gpu,
        target_utilization=0.75,
        scale_up_threshold=0.85,
        scale_down_threshold=0.25,
        min_instances=1,
        max_instances=8,
        enable_predictive_scaling=enable_predictive_scaling
    )
    
    photonic_policy = ScalingPolicy(
        name="photonic_scaling",
        resource_type=ResourceType.photonic_cores,
        target_utilization=0.8,
        scale_up_threshold=0.9,
        scale_down_threshold=0.4,
        min_instances=2,
        max_instances=16,
        enable_predictive_scaling=enable_predictive_scaling
    )
    
    auto_scaler.add_scaling_policy(cpu_policy)
    auto_scaler.add_scaling_policy(gpu_policy)
    auto_scaler.add_scaling_policy(photonic_policy)
    
    # Create load balancer if enabled
    load_balancer = None
    if enable_load_balancing:
        load_balancer = LoadBalancer()
        
        # Register initial instances
        for i in range(2):
            load_balancer.register_instance(
                f"photonic_instance_{i}",
                f"http://localhost:800{i}",
                capacity=1.0
            )
    
    # Start auto-scaling
    auto_scaler.start_auto_scaling()
    
    # Attach to base system
    base_system.auto_scaler = auto_scaler
    base_system.load_balancer = load_balancer
    
    # Wrap forward method with scaling awareness
    original_forward = base_system.forward
    
    def scaling_aware_forward(*args, **kwargs):
        start_time = time.time()
        
        try:
            result = original_forward(*args, **kwargs)
            end_time = time.time()
            latency = end_time - start_time
            
            # Create metrics for auto-scaler
            metrics = ResourceMetrics(
                timestamp=start_time,
                cpu_utilization=0.6,  # Would get from real monitoring
                memory_utilization=0.5,
                gpu_utilization=0.7,
                network_throughput=100.0,
                storage_iops=50.0,
                photonic_core_utilization=0.8,
                request_rate=10.0,
                response_latency=latency,
                error_rate=0.01
            )
            
            # Process metrics for scaling decisions
            auto_scaler.process_metrics(metrics)
            
            return result
            
        except Exception as e:
            # Record error for scaling decisions
            end_time = time.time()
            logger.error(f"Forward pass failed: {e}")
            raise
    
    base_system.forward = scaling_aware_forward
    
    return base_system