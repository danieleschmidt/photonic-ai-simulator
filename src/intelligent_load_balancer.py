"""
Intelligent Load Balancing and Auto-Scaling for Photonic AI Systems.

Implements advanced load balancing algorithms with predictive scaling,
health monitoring, and intelligent request routing for optimal performance
across distributed photonic neural network deployments.
"""

import numpy as np
import asyncio
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import logging
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import aiohttp
from datetime import datetime, timedelta
from enum import Enum
import psutil
import math

try:
    from .concurrent_optimization import InferencePipeline, ConcurrencyConfig
    from .models import PhotonicNeuralNetwork
    from .scaling import AutoScalingManager, ScalingConfig
except ImportError:
    from concurrent_optimization import InferencePipeline, ConcurrencyConfig
    from models import PhotonicNeuralNetwork
    from scaling import AutoScalingManager, ScalingConfig

logger = logging.getLogger(__name__)


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_RESPONSE_TIME = "least_response_time"
    ADAPTIVE_WEIGHTED = "adaptive_weighted"
    QUANTUM_AWARE = "quantum_aware"
    PREDICTIVE = "predictive"


class HealthStatus(Enum):
    """Node health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"


@dataclass
class NodeMetrics:
    """Comprehensive node performance metrics."""
    node_id: str
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    gpu_utilization: float = 0.0
    active_connections: int = 0
    avg_response_time_ms: float = 0.0
    error_rate: float = 0.0
    throughput_rps: float = 0.0
    queue_length: int = 0
    health_status: HealthStatus = HealthStatus.HEALTHY
    last_update: float = 0.0
    
    # Photonic-specific metrics
    photonic_processor_temp: float = 0.0
    wavelength_channel_utilization: Dict[int, float] = field(default_factory=dict)
    quantum_coherence_quality: float = 1.0
    optical_power_efficiency: float = 1.0


@dataclass 
class LoadBalancerConfig:
    """Configuration for intelligent load balancer."""
    strategy: LoadBalancingStrategy = LoadBalancingStrategy.ADAPTIVE_WEIGHTED
    health_check_interval_s: float = 5.0
    metrics_collection_interval_s: float = 1.0
    prediction_window_s: float = 60.0
    max_queue_size: int = 1000
    circuit_breaker_threshold: float = 0.1  # 10% error rate
    circuit_breaker_timeout_s: float = 30.0
    enable_predictive_scaling: bool = True
    photonic_optimization: bool = True


class PhotonicNode:
    """Represents a photonic neural network node."""
    
    def __init__(self, 
                 node_id: str, 
                 models: List[PhotonicNeuralNetwork],
                 endpoint: str = "localhost",
                 port: int = 8000):
        self.node_id = node_id
        self.models = models
        self.endpoint = endpoint
        self.port = port
        self.base_url = f"http://{endpoint}:{port}"
        
        # Node state
        self.metrics = NodeMetrics(node_id=node_id)
        self.circuit_breaker_open = False
        self.circuit_breaker_last_failure = 0.0
        self.weight = 1.0  # Load balancing weight
        
        # Inference pipeline
        concurrency_config = ConcurrencyConfig()
        self.pipeline = InferencePipeline(models, concurrency_config)
        
        # Request history for prediction
        self.request_history = deque(maxlen=1000)
        self.response_times = deque(maxlen=100)
        
        logger.info(f"Initialized PhotonicNode {node_id} at {self.base_url}")
    
    async def process_inference(self, inputs: np.ndarray, model_idx: int = 0) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process inference request on this node."""
        if self.circuit_breaker_open:
            if time.time() - self.circuit_breaker_last_failure > 30.0:
                self.circuit_breaker_open = False
                logger.info(f"Circuit breaker closed for node {self.node_id}")
            else:
                raise Exception(f"Circuit breaker open for node {self.node_id}")
        
        start_time = time.time()
        try:
            # Submit to inference pipeline
            request_id = await self.pipeline.submit_inference_request(inputs, model_idx)
            
            # Get result (with timeout)
            input_hash = f"{id(inputs)}_{model_idx}"
            result = await self.pipeline.get_inference_result(input_hash, timeout=10.0)
            
            if result is None:
                raise Exception(f"Inference timeout on node {self.node_id}")
            
            # Update metrics
            end_time = time.time()
            response_time = (end_time - start_time) * 1000
            self.response_times.append(response_time)
            self.request_history.append(end_time)
            
            self._update_metrics(response_time, success=True)
            
            return result
            
        except Exception as e:
            self._update_metrics(0, success=False)
            self._trigger_circuit_breaker()
            raise e
    
    def _update_metrics(self, response_time: float, success: bool):
        """Update node performance metrics."""
        current_time = time.time()
        self.metrics.last_update = current_time
        
        # Update response time
        if self.response_times:
            self.metrics.avg_response_time_ms = np.mean(list(self.response_times))
        
        # Update error rate
        if not success:
            self.metrics.error_rate = min(1.0, self.metrics.error_rate + 0.01)
        else:
            self.metrics.error_rate = max(0.0, self.metrics.error_rate - 0.001)
        
        # Update throughput (requests per second in last minute)
        recent_requests = [t for t in self.request_history if current_time - t < 60.0]
        self.metrics.throughput_rps = len(recent_requests) / 60.0
        
        # System metrics
        try:
            self.metrics.cpu_utilization = psutil.cpu_percent()
            self.metrics.memory_utilization = psutil.virtual_memory().percent
            
            # GPU utilization (if available)
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                self.metrics.gpu_utilization = gpu_util.gpu
            except:
                pass
        except:
            pass
        
        # Update health status
        self._update_health_status()
    
    def _update_health_status(self):
        """Update node health status based on metrics."""
        if self.metrics.error_rate > 0.2 or self.metrics.avg_response_time_ms > 5000:
            self.metrics.health_status = HealthStatus.UNHEALTHY
        elif self.metrics.error_rate > 0.05 or self.metrics.avg_response_time_ms > 2000:
            self.metrics.health_status = HealthStatus.DEGRADED
        else:
            self.metrics.health_status = HealthStatus.HEALTHY
    
    def _trigger_circuit_breaker(self):
        """Trigger circuit breaker for node."""
        self.circuit_breaker_open = True
        self.circuit_breaker_last_failure = time.time()
        logger.warning(f"Circuit breaker opened for node {self.node_id}")
    
    def get_load_score(self) -> float:
        """Get current load score (lower is better)."""
        base_score = (
            self.metrics.cpu_utilization * 0.3 +
            self.metrics.memory_utilization * 0.2 +
            self.metrics.avg_response_time_ms / 1000.0 * 0.3 +
            self.metrics.error_rate * 100.0 * 0.2
        )
        
        # Adjust for photonic-specific metrics
        if hasattr(self.metrics, 'photonic_processor_temp') and self.metrics.photonic_processor_temp > 0:
            # Higher temperature = higher load score
            temp_factor = max(0.0, (self.metrics.photonic_processor_temp - 300.0) / 50.0)
            base_score += temp_factor * 0.1
        
        return max(0.1, base_score)  # Minimum score to avoid division by zero


class PredictiveLoadAnalyzer:
    """Analyzes load patterns for predictive scaling."""
    
    def __init__(self):
        self.hourly_patterns = defaultdict(list)  # Hour -> [load_values]
        self.daily_patterns = defaultdict(list)   # Day -> [load_values]
        self.trend_history = deque(maxlen=1440)   # 24 hours of minute data
        
    def record_load(self, timestamp: float, total_load: float):
        """Record load measurement."""
        dt = datetime.fromtimestamp(timestamp)
        hour = dt.hour
        day = dt.weekday()
        
        self.hourly_patterns[hour].append(total_load)
        self.daily_patterns[day].append(total_load)
        self.trend_history.append((timestamp, total_load))
        
        # Keep only recent data
        if len(self.hourly_patterns[hour]) > 7:  # Keep last 7 days
            self.hourly_patterns[hour].pop(0)
        if len(self.daily_patterns[day]) > 4:  # Keep last 4 weeks
            self.daily_patterns[day].pop(0)
    
    def predict_future_load(self, horizon_minutes: int = 15) -> float:
        """Predict load for next horizon_minutes."""
        current_time = time.time()
        current_dt = datetime.fromtimestamp(current_time)
        future_dt = current_dt + timedelta(minutes=horizon_minutes)
        
        # Get historical pattern for this time
        hour_pattern = np.array(self.hourly_patterns.get(future_dt.hour, [1.0]))
        day_pattern = np.array(self.daily_patterns.get(future_dt.weekday(), [1.0]))
        
        # Weighted prediction based on patterns
        hour_pred = np.mean(hour_pattern) if len(hour_pattern) > 0 else 1.0
        day_pred = np.mean(day_pattern) if len(day_pattern) > 0 else 1.0
        
        # Trend analysis
        if len(self.trend_history) >= 30:  # Need at least 30 minutes of data
            recent_loads = [load for _, load in list(self.trend_history)[-30:]]
            trend = np.polyfit(range(len(recent_loads)), recent_loads, 1)[0]
            trend_pred = recent_loads[-1] + trend * horizon_minutes
        else:
            trend_pred = hour_pred
        
        # Combine predictions
        predicted_load = (hour_pred * 0.4 + day_pred * 0.3 + trend_pred * 0.3)
        
        return max(0.1, predicted_load)
    
    def get_scaling_recommendation(self, current_nodes: int, target_utilization: float = 0.7) -> int:
        """Recommend number of nodes based on predicted load."""
        predicted_load = self.predict_future_load()
        
        # Calculate required nodes
        required_nodes = math.ceil(predicted_load / target_utilization)
        
        # Add buffer for safety
        buffer_nodes = max(1, int(required_nodes * 0.2))
        recommended_nodes = required_nodes + buffer_nodes
        
        return max(1, recommended_nodes)


class IntelligentLoadBalancer:
    """Advanced load balancer with predictive scaling."""
    
    def __init__(self, config: LoadBalancerConfig = None):
        self.config = config or LoadBalancerConfig()
        self.nodes: Dict[str, PhotonicNode] = {}
        self.request_counter = 0
        self.load_analyzer = PredictiveLoadAnalyzer()
        
        # Auto-scaling
        scaling_config = ScalingConfig()
        self.auto_scaler = AutoScalingManager(scaling_config)
        
        # Monitoring
        self.monitoring_active = False
        self.monitoring_task = None
        
        logger.info("Initialized IntelligentLoadBalancer")
    
    def add_node(self, node: PhotonicNode):
        """Add node to load balancer."""
        self.nodes[node.node_id] = node
        logger.info(f"Added node {node.node_id} to load balancer")
    
    def remove_node(self, node_id: str):
        """Remove node from load balancer."""
        if node_id in self.nodes:
            del self.nodes[node_id]
            logger.info(f"Removed node {node_id} from load balancer")
    
    async def route_request(self, inputs: np.ndarray, model_idx: int = 0) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Route inference request to optimal node."""
        # Select best node based on strategy
        selected_node = self._select_node()
        
        if selected_node is None:
            raise Exception("No healthy nodes available")
        
        try:
            # Process request
            start_time = time.time()
            result = await selected_node.process_inference(inputs, model_idx)
            end_time = time.time()
            
            # Record metrics for load analysis
            total_load = sum(node.get_load_score() for node in self.nodes.values())
            self.load_analyzer.record_load(end_time, total_load)
            
            return result
            
        except Exception as e:
            logger.error(f"Request failed on node {selected_node.node_id}: {e}")
            # Try failover to another node
            return await self._failover_request(inputs, model_idx, selected_node.node_id)
    
    def _select_node(self) -> Optional[PhotonicNode]:
        """Select optimal node based on load balancing strategy."""
        healthy_nodes = [
            node for node in self.nodes.values() 
            if node.metrics.health_status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]
            and not node.circuit_breaker_open
        ]
        
        if not healthy_nodes:
            return None
        
        if self.config.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return healthy_nodes[self.request_counter % len(healthy_nodes)]
        
        elif self.config.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return min(healthy_nodes, key=lambda n: n.metrics.active_connections)
        
        elif self.config.strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
            return min(healthy_nodes, key=lambda n: n.metrics.avg_response_time_ms)
        
        elif self.config.strategy == LoadBalancingStrategy.ADAPTIVE_WEIGHTED:
            # Select based on inverse of load score
            weights = [1.0 / node.get_load_score() for node in healthy_nodes]
            total_weight = sum(weights)
            
            if total_weight > 0:
                normalized_weights = [w / total_weight for w in weights]
                selected_idx = np.random.choice(len(healthy_nodes), p=normalized_weights)
                return healthy_nodes[selected_idx]
            else:
                return healthy_nodes[0]
        
        elif self.config.strategy == LoadBalancingStrategy.QUANTUM_AWARE:
            # Consider quantum coherence quality in selection
            scores = []
            for node in healthy_nodes:
                base_score = 1.0 / node.get_load_score()
                quantum_bonus = getattr(node.metrics, 'quantum_coherence_quality', 1.0)
                optical_bonus = getattr(node.metrics, 'optical_power_efficiency', 1.0)
                total_score = base_score * quantum_bonus * optical_bonus
                scores.append(total_score)
            
            best_idx = np.argmax(scores)
            return healthy_nodes[best_idx]
        
        elif self.config.strategy == LoadBalancingStrategy.PREDICTIVE:
            # Use predictive analysis for selection
            predicted_loads = []
            for node in healthy_nodes:
                current_load = node.get_load_score()
                # Simple prediction based on trend
                predicted_load = current_load  # Simplified - could use ML model
                predicted_loads.append(predicted_load)
            
            best_idx = np.argmin(predicted_loads)
            return healthy_nodes[best_idx]
        
        # Default to round robin
        return healthy_nodes[self.request_counter % len(healthy_nodes)]
    
    async def _failover_request(self, inputs: np.ndarray, model_idx: int, failed_node_id: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Failover request to another node."""
        # Get healthy nodes excluding the failed one
        available_nodes = [
            node for node in self.nodes.values()
            if (node.node_id != failed_node_id and 
                node.metrics.health_status == HealthStatus.HEALTHY and
                not node.circuit_breaker_open)
        ]
        
        if not available_nodes:
            raise Exception("No healthy nodes available for failover")
        
        # Select node with lowest load
        failover_node = min(available_nodes, key=lambda n: n.get_load_score())
        
        logger.info(f"Failing over request to node {failover_node.node_id}")
        return await failover_node.process_inference(inputs, model_idx)
    
    async def start_monitoring(self):
        """Start continuous monitoring and auto-scaling."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Started load balancer monitoring")
    
    async def stop_monitoring(self):
        """Stop monitoring."""
        self.monitoring_active = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped load balancer monitoring")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect metrics from all nodes
                await self._collect_metrics()
                
                # Perform auto-scaling decisions
                if self.config.enable_predictive_scaling:
                    await self._evaluate_scaling()
                
                # Health checks
                await self._perform_health_checks()
                
                await asyncio.sleep(self.config.metrics_collection_interval_s)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
    
    async def _collect_metrics(self):
        """Collect metrics from all nodes."""
        for node in self.nodes.values():
            # Metrics are updated automatically during request processing
            pass
    
    async def _evaluate_scaling(self):
        """Evaluate and execute scaling decisions."""
        current_nodes = len(self.nodes)
        recommended_nodes = self.load_analyzer.get_scaling_recommendation(current_nodes)
        
        if recommended_nodes > current_nodes:
            logger.info(f"Scaling recommendation: {current_nodes} -> {recommended_nodes}")
            # In a real implementation, this would trigger node provisioning
        elif recommended_nodes < current_nodes:
            logger.info(f"Scale-down recommendation: {current_nodes} -> {recommended_nodes}")
            # In a real implementation, this would trigger node deprovisioning
    
    async def _perform_health_checks(self):
        """Perform health checks on all nodes."""
        for node in self.nodes.values():
            # Health status is updated automatically during metric updates
            if node.metrics.health_status == HealthStatus.UNHEALTHY:
                logger.warning(f"Node {node.node_id} is unhealthy")
    
    def get_cluster_metrics(self) -> Dict[str, Any]:
        """Get overall cluster metrics."""
        if not self.nodes:
            return {}
        
        total_requests = sum(len(node.request_history) for node in self.nodes.values())
        avg_response_time = np.mean([
            node.metrics.avg_response_time_ms for node in self.nodes.values()
        ])
        
        healthy_nodes = sum(
            1 for node in self.nodes.values() 
            if node.metrics.health_status == HealthStatus.HEALTHY
        )
        
        return {
            "total_nodes": len(self.nodes),
            "healthy_nodes": healthy_nodes,
            "total_requests_processed": total_requests,
            "avg_cluster_response_time_ms": avg_response_time,
            "load_balancing_strategy": self.config.strategy.value,
            "predicted_load_next_15min": self.load_analyzer.predict_future_load(15)
        }