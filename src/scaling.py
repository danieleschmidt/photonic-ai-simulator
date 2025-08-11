"""
Advanced scaling and distributed computing for photonic neural networks.

Implements distributed training, inference scaling, load balancing,
and cloud deployment optimization for production photonic AI systems.
"""

import numpy as np
import time
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
from abc import ABC, abstractmethod
import logging
import queue
import asyncio
from pathlib import Path
import json

try:
    from .models import PhotonicNeuralNetwork
    from .training import TrainingConfig, ForwardOnlyTrainer
    from .optimization import OptimizationConfig, create_optimized_network
    from .utils.monitoring import SystemMonitor, SystemMetrics, AlertLevel
    from .utils.logging_config import get_logger
except ImportError:
    from models import PhotonicNeuralNetwork
    from training import TrainingConfig, ForwardOnlyTrainer
    from optimization import OptimizationConfig, create_optimized_network
    from utils.monitoring import SystemMonitor, SystemMetrics, AlertLevel
    from utils.logging_config import get_logger

logger = get_logger(__name__)


class ScalingStrategy(Enum):
    """Scaling strategy types."""
    HORIZONTAL = "horizontal"      # Multiple instances
    VERTICAL = "vertical"          # Larger instances  
    ELASTIC = "elastic"           # Auto-scaling based on load
    HYBRID = "hybrid"             # Combination of strategies


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LATENCY_BASED = "latency_based"
    ADAPTIVE = "adaptive"


@dataclass
class ScalingConfig:
    """Configuration for scaling operations."""
    strategy: ScalingStrategy = ScalingStrategy.ELASTIC
    
    # Horizontal scaling
    min_instances: int = 1
    max_instances: int = 10
    target_cpu_utilization: float = 0.7
    target_latency_ms: float = 1.0
    scale_up_threshold: float = 0.8
    scale_down_threshold: float = 0.3
    cooldown_seconds: int = 300
    
    # Load balancing
    load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.ADAPTIVE
    health_check_interval: float = 30.0
    max_queue_size: int = 1000
    request_timeout_seconds: float = 10.0
    
    # Performance optimization
    batch_processing: bool = True
    optimal_batch_size: int = 32
    batch_timeout_ms: float = 50.0
    enable_caching: bool = True
    cache_size_mb: int = 512
    
    # Monitoring and alerting
    enable_auto_scaling_alerts: bool = True
    metrics_collection_interval: float = 10.0


class WorkerNode:
    """Individual worker node for distributed inference."""
    
    def __init__(self, worker_id: str, model: PhotonicNeuralNetwork,
                 optimization_config: OptimizationConfig):
        """Initialize worker node."""
        self.worker_id = worker_id
        self.model = model
        self.optimization_config = optimization_config
        
        # Worker state
        self.is_healthy = True
        self.current_load = 0.0
        self.total_requests = 0
        self.failed_requests = 0
        self.avg_latency_ms = 0.0
        
        # Request queue
        self.request_queue = queue.Queue(maxsize=1000)
        self.result_cache = {}
        
        # Performance tracking
        self.performance_history = []
        self.last_health_check = time.time()
        
        # Worker thread
        self.worker_thread = None
        self.is_running = False
        
    def start(self):
        """Start worker node processing."""
        if self.is_running:
            return
            
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._worker_loop)
        self.worker_thread.daemon = True
        self.worker_thread.start()
        
        logger.info(f"Worker {self.worker_id} started")
    
    def stop(self):
        """Stop worker node processing."""
        self.is_running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5.0)
        
        logger.info(f"Worker {self.worker_id} stopped")
    
    def _worker_loop(self):
        """Main worker processing loop."""
        batch = []
        batch_start_time = time.time()
        
        while self.is_running:
            try:
                # Check for batch processing opportunity
                current_time = time.time()
                
                # Get request from queue (non-blocking)
                try:
                    request = self.request_queue.get(timeout=0.01)
                    batch.append(request)
                except queue.Empty:
                    pass
                
                # Process batch if conditions are met
                should_process = (
                    len(batch) >= self.optimization_config.batch_size or
                    (batch and (current_time - batch_start_time) > 0.05) or  # 50ms timeout
                    not self.is_running
                )
                
                if should_process and batch:
                    self._process_batch(batch)
                    batch = []
                    batch_start_time = current_time
                
                # Update health status
                self._update_health_status()
                
            except Exception as e:
                logger.error(f"Worker {self.worker_id} error: {e}")
                self.failed_requests += 1
                self.is_healthy = False
    
    def _process_batch(self, batch: List[Dict[str, Any]]):
        """Process batch of inference requests."""
        if not batch:
            return
        
        try:
            start_time = time.perf_counter()
            
            # Extract input data from batch
            batch_inputs = []
            request_ids = []
            
            for request in batch:
                batch_inputs.append(request['input'])
                request_ids.append(request['request_id'])
            
            # Combine inputs for batch processing
            combined_input = np.array(batch_inputs)
            
            # Run inference
            predictions, metrics = self.model.forward(combined_input, measure_latency=True)
            
            # Calculate performance metrics
            end_time = time.perf_counter()
            batch_latency_ms = (end_time - start_time) * 1000
            avg_latency_per_sample = batch_latency_ms / len(batch)
            
            # Update statistics
            self.total_requests += len(batch)
            self._update_avg_latency(avg_latency_per_sample)
            
            # Store results
            for i, request_id in enumerate(request_ids):
                result = {
                    'request_id': request_id,
                    'prediction': predictions[i].tolist(),
                    'latency_ms': avg_latency_per_sample,
                    'worker_id': self.worker_id,
                    'timestamp': time.time()
                }
                
                # Store in cache and notify completion
                self.result_cache[request_id] = result
                
                # Notify request completion (simplified)
                if 'callback' in batch[i]:
                    batch[i]['callback'](result)
            
            # Record performance
            self.performance_history.append({
                'timestamp': time.time(),
                'batch_size': len(batch),
                'latency_ms': batch_latency_ms,
                'avg_latency_per_sample': avg_latency_per_sample
            })
            
            # Maintain history size
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-500:]
                
        except Exception as e:
            logger.error(f"Batch processing failed in worker {self.worker_id}: {e}")
            self.failed_requests += len(batch)
            self.is_healthy = False
    
    def _update_avg_latency(self, new_latency: float):
        """Update running average latency."""
        if self.avg_latency_ms == 0:
            self.avg_latency_ms = new_latency
        else:
            # Exponential moving average
            alpha = 0.1
            self.avg_latency_ms = (1 - alpha) * self.avg_latency_ms + alpha * new_latency
    
    def _update_health_status(self):
        """Update worker health status."""
        current_time = time.time()
        
        # Check error rate
        if self.total_requests > 0:
            error_rate = self.failed_requests / self.total_requests
            if error_rate > 0.1:  # 10% error rate threshold
                self.is_healthy = False
                return
        
        # Check queue size
        queue_size = self.request_queue.qsize()
        self.current_load = queue_size / self.request_queue.maxsize
        
        if self.current_load > 0.9:  # 90% queue utilization
            logger.warning(f"Worker {self.worker_id} queue nearly full")
        
        # Reset health if conditions improve
        if error_rate < 0.05 and self.current_load < 0.8:
            self.is_healthy = True
        
        self.last_health_check = current_time
    
    def submit_request(self, request: Dict[str, Any]) -> bool:
        """Submit inference request to worker."""
        try:
            self.request_queue.put(request, timeout=1.0)
            return True
        except queue.Full:
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get worker status."""
        return {
            'worker_id': self.worker_id,
            'is_healthy': self.is_healthy,
            'current_load': self.current_load,
            'total_requests': self.total_requests,
            'failed_requests': self.failed_requests,
            'error_rate': self.failed_requests / max(self.total_requests, 1),
            'avg_latency_ms': self.avg_latency_ms,
            'queue_size': self.request_queue.qsize(),
            'last_health_check': self.last_health_check
        }


class LoadBalancer:
    """Intelligent load balancer for photonic neural network workers."""
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.ADAPTIVE):
        """Initialize load balancer."""
        self.strategy = strategy
        self.workers = {}
        self.request_counter = 0
        self.routing_weights = {}
        self.performance_history = {}
        
    def register_worker(self, worker: WorkerNode, weight: float = 1.0):
        """Register worker node."""
        self.workers[worker.worker_id] = worker
        self.routing_weights[worker.worker_id] = weight
        self.performance_history[worker.worker_id] = []
        
        logger.info(f"Registered worker {worker.worker_id} with weight {weight}")
    
    def unregister_worker(self, worker_id: str):
        """Unregister worker node."""
        if worker_id in self.workers:
            del self.workers[worker_id]
            del self.routing_weights[worker_id]
            del self.performance_history[worker_id]
            logger.info(f"Unregistered worker {worker_id}")
    
    def route_request(self, request: Dict[str, Any]) -> Optional[str]:
        """Route request to optimal worker."""
        healthy_workers = [
            worker_id for worker_id, worker in self.workers.items()
            if worker.is_healthy and worker.request_queue.qsize() < worker.request_queue.maxsize
        ]
        
        if not healthy_workers:
            return None
        
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_routing(healthy_workers)
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections_routing(healthy_workers)
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_routing(healthy_workers)
        elif self.strategy == LoadBalancingStrategy.LATENCY_BASED:
            return self._latency_based_routing(healthy_workers)
        elif self.strategy == LoadBalancingStrategy.ADAPTIVE:
            return self._adaptive_routing(healthy_workers)
        
        return healthy_workers[0]  # Fallback
    
    def _round_robin_routing(self, healthy_workers: List[str]) -> str:
        """Round robin routing."""
        self.request_counter += 1
        return healthy_workers[self.request_counter % len(healthy_workers)]
    
    def _least_connections_routing(self, healthy_workers: List[str]) -> str:
        """Route to worker with least connections."""
        return min(healthy_workers, key=lambda w: self.workers[w].current_load)
    
    def _weighted_round_robin_routing(self, healthy_workers: List[str]) -> str:
        """Weighted round robin routing."""
        # Simplified weighted selection
        weights = [self.routing_weights.get(w, 1.0) for w in healthy_workers]
        total_weight = sum(weights)
        
        if total_weight == 0:
            return healthy_workers[0]
        
        selection = (self.request_counter % int(total_weight))
        cumulative = 0
        
        for i, weight in enumerate(weights):
            cumulative += weight
            if selection < cumulative:
                self.request_counter += 1
                return healthy_workers[i]
        
        return healthy_workers[0]
    
    def _latency_based_routing(self, healthy_workers: List[str]) -> str:
        """Route to worker with lowest latency."""
        return min(healthy_workers, key=lambda w: self.workers[w].avg_latency_ms)
    
    def _adaptive_routing(self, healthy_workers: List[str]) -> str:
        """Adaptive routing based on multiple factors."""
        scores = {}
        
        for worker_id in healthy_workers:
            worker = self.workers[worker_id]
            
            # Combine multiple factors
            latency_score = 1.0 / (worker.avg_latency_ms + 1.0)  # Lower latency = higher score
            load_score = 1.0 - worker.current_load                # Lower load = higher score
            error_score = 1.0 - (worker.failed_requests / max(worker.total_requests, 1))
            weight_score = self.routing_weights.get(worker_id, 1.0)
            
            # Weighted combination
            scores[worker_id] = (
                0.4 * latency_score +
                0.3 * load_score + 
                0.2 * error_score +
                0.1 * weight_score
            )
        
        return max(scores.keys(), key=lambda w: scores[w])
    
    def get_load_balancer_status(self) -> Dict[str, Any]:
        """Get load balancer status."""
        worker_statuses = {w_id: worker.get_status() 
                          for w_id, worker in self.workers.items()}
        
        healthy_count = sum(1 for w in self.workers.values() if w.is_healthy)
        total_requests = sum(w.total_requests for w in self.workers.values())
        avg_latency = np.mean([w.avg_latency_ms for w in self.workers.values() if w.avg_latency_ms > 0])
        
        return {
            'strategy': self.strategy.value,
            'total_workers': len(self.workers),
            'healthy_workers': healthy_count,
            'total_requests_processed': total_requests,
            'avg_latency_ms': avg_latency if not np.isnan(avg_latency) else 0.0,
            'worker_statuses': worker_statuses
        }


class AutoScaler:
    """Automatic scaling system for photonic neural networks."""
    
    def __init__(self, config: ScalingConfig):
        """Initialize auto-scaler."""
        self.config = config
        self.workers = {}
        self.load_balancer = LoadBalancer(config.load_balancing_strategy)
        
        # Scaling state
        self.current_instances = 0
        self.last_scale_action = 0
        self.scaling_history = []
        
        # Monitoring
        self.metrics_history = []
        self.is_monitoring = False
        self.monitoring_thread = None
        
    def start_auto_scaling(self):
        """Start automatic scaling monitoring."""
        if self.is_monitoring:
            return
            
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._scaling_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        logger.info("Auto-scaling started")
    
    def stop_auto_scaling(self):
        """Stop automatic scaling."""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        logger.info("Auto-scaling stopped")
    
    def _scaling_loop(self):
        """Main auto-scaling loop."""
        while self.is_monitoring:
            try:
                # Collect metrics
                metrics = self._collect_scaling_metrics()
                self.metrics_history.append(metrics)
                
                # Maintain history size
                if len(self.metrics_history) > 100:
                    self.metrics_history = self.metrics_history[-50:]
                
                # Make scaling decisions
                if time.time() - self.last_scale_action > self.config.cooldown_seconds:
                    self._evaluate_scaling_decision(metrics)
                
                time.sleep(self.config.metrics_collection_interval)
                
            except Exception as e:
                logger.error(f"Auto-scaling loop error: {e}")
                time.sleep(5.0)
    
    def _collect_scaling_metrics(self) -> Dict[str, float]:
        """Collect metrics for scaling decisions."""
        if not self.workers:
            return {
                'avg_cpu_utilization': 0.0,
                'avg_latency_ms': 0.0,
                'queue_utilization': 0.0,
                'error_rate': 0.0,
                'healthy_worker_ratio': 0.0
            }
        
        workers = list(self.workers.values())
        
        # Calculate aggregate metrics
        avg_latency = np.mean([w.avg_latency_ms for w in workers if w.avg_latency_ms > 0])
        avg_load = np.mean([w.current_load for w in workers])
        healthy_ratio = sum(1 for w in workers if w.is_healthy) / len(workers)
        total_errors = sum(w.failed_requests for w in workers)
        total_requests = sum(w.total_requests for w in workers)
        error_rate = total_errors / max(total_requests, 1)
        
        return {
            'avg_cpu_utilization': avg_load,  # Using load as proxy for CPU
            'avg_latency_ms': avg_latency if not np.isnan(avg_latency) else 0.0,
            'queue_utilization': avg_load,
            'error_rate': error_rate,
            'healthy_worker_ratio': healthy_ratio,
            'total_workers': len(workers),
            'healthy_workers': sum(1 for w in workers if w.is_healthy)
        }
    
    def _evaluate_scaling_decision(self, metrics: Dict[str, float]):
        """Evaluate whether to scale up or down."""
        current_workers = metrics['healthy_workers']
        
        # Scale up conditions
        should_scale_up = (
            (metrics['avg_cpu_utilization'] > self.config.scale_up_threshold or
             metrics['avg_latency_ms'] > self.config.target_latency_ms or
             metrics['queue_utilization'] > 0.8) and
            current_workers < self.config.max_instances
        )
        
        # Scale down conditions  
        should_scale_down = (
            metrics['avg_cpu_utilization'] < self.config.scale_down_threshold and
            metrics['avg_latency_ms'] < self.config.target_latency_ms * 0.5 and
            current_workers > self.config.min_instances
        )
        
        if should_scale_up:
            self._scale_up()
        elif should_scale_down:
            self._scale_down()
    
    def _scale_up(self):
        """Scale up by adding worker instances."""
        new_worker_id = f"worker_{len(self.workers)}_{int(time.time())}"
        
        try:
            # Create new optimized model instance
            model = create_optimized_network("mnist", "high")  # Default task
            optimization_config = OptimizationConfig(use_gpu=True, batch_size=32)
            
            # Create and start worker
            worker = WorkerNode(new_worker_id, model, optimization_config)
            worker.start()
            
            # Register with load balancer
            self.workers[new_worker_id] = worker
            self.load_balancer.register_worker(worker)
            
            self.current_instances += 1
            self.last_scale_action = time.time()
            
            # Record scaling action
            self.scaling_history.append({
                'timestamp': time.time(),
                'action': 'scale_up',
                'worker_id': new_worker_id,
                'total_workers': self.current_instances
            })
            
            logger.info(f"Scaled up: Added worker {new_worker_id} (total: {self.current_instances})")
            
        except Exception as e:
            logger.error(f"Scale up failed: {e}")
    
    def _scale_down(self):
        """Scale down by removing worker instances."""
        if not self.workers:
            return
        
        # Find worker with lowest utilization
        worker_to_remove = min(self.workers.values(), key=lambda w: w.current_load)
        worker_id = worker_to_remove.worker_id
        
        try:
            # Gracefully stop worker
            worker_to_remove.stop()
            
            # Unregister from load balancer
            self.load_balancer.unregister_worker(worker_id)
            del self.workers[worker_id]
            
            self.current_instances -= 1
            self.last_scale_action = time.time()
            
            # Record scaling action
            self.scaling_history.append({
                'timestamp': time.time(),
                'action': 'scale_down',
                'worker_id': worker_id,
                'total_workers': self.current_instances
            })
            
            logger.info(f"Scaled down: Removed worker {worker_id} (total: {self.current_instances})")
            
        except Exception as e:
            logger.error(f"Scale down failed: {e}")
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get comprehensive scaling status."""
        recent_metrics = self.metrics_history[-10:] if self.metrics_history else []
        
        return {
            'current_instances': self.current_instances,
            'target_range': f"{self.config.min_instances}-{self.config.max_instances}",
            'scaling_strategy': self.config.strategy.value,
            'load_balancing_strategy': self.config.load_balancing_strategy.value,
            'is_auto_scaling': self.is_monitoring,
            'last_scale_action': self.last_scale_action,
            'recent_metrics': recent_metrics,
            'scaling_history': self.scaling_history[-10:],  # Last 10 actions
            'load_balancer_status': self.load_balancer.get_load_balancer_status()
        }


class DistributedInferenceServer:
    """
    High-performance distributed inference server.
    
    Provides production-ready inference serving with auto-scaling,
    load balancing, and comprehensive monitoring.
    """
    
    def __init__(self, model_task: str = "mnist", 
                 scaling_config: Optional[ScalingConfig] = None):
        """Initialize distributed inference server."""
        self.model_task = model_task
        self.scaling_config = scaling_config or ScalingConfig()
        
        # Initialize auto-scaler
        self.auto_scaler = AutoScaler(self.scaling_config)
        
        # Request tracking
        self.request_id_counter = 0
        self.active_requests = {}
        self.request_history = []
        
        # Performance tracking
        self.start_time = time.time()
        self.total_requests_served = 0
        
        # Initialize with minimum instances
        self._initialize_workers()
    
    def _initialize_workers(self):
        """Initialize minimum number of worker instances."""
        for i in range(self.scaling_config.min_instances):
            self._add_initial_worker(i)
    
    def _add_initial_worker(self, worker_index: int):
        """Add initial worker instance."""
        worker_id = f"initial_worker_{worker_index}"
        
        # Create optimized model
        model = create_optimized_network(self.model_task, "high")
        optimization_config = OptimizationConfig(
            use_gpu=True,
            batch_size=self.scaling_config.optimal_batch_size,
            vectorize_operations=True,
            cache_activations=True
        )
        
        # Create and start worker
        worker = WorkerNode(worker_id, model, optimization_config)
        worker.start()
        
        # Register with auto-scaler
        self.auto_scaler.workers[worker_id] = worker
        self.auto_scaler.load_balancer.register_worker(worker)
        self.auto_scaler.current_instances += 1
        
        logger.info(f"Initialized worker {worker_id}")
    
    def start_server(self):
        """Start the distributed inference server."""
        self.auto_scaler.start_auto_scaling()
        logger.info("Distributed inference server started")
    
    def stop_server(self):
        """Stop the distributed inference server."""
        # Stop auto-scaler
        self.auto_scaler.stop_auto_scaling()
        
        # Stop all workers
        for worker in self.auto_scaler.workers.values():
            worker.stop()
        
        logger.info("Distributed inference server stopped")
    
    def submit_inference_request(self, input_data: np.ndarray,
                               callback: Optional[Callable] = None) -> str:
        """
        Submit inference request to the server.
        
        Args:
            input_data: Input data for inference
            callback: Optional callback function for async result handling
            
        Returns:
            Request ID for tracking
        """
        # Generate request ID
        self.request_id_counter += 1
        request_id = f"req_{self.request_id_counter}_{int(time.time() * 1000)}"
        
        # Create request
        request = {
            'request_id': request_id,
            'input': input_data,
            'timestamp': time.time(),
            'callback': callback
        }
        
        # Route request to worker
        worker_id = self.auto_scaler.load_balancer.route_request(request)
        
        if worker_id is None:
            logger.error("No healthy workers available")
            return None
        
        # Submit to worker
        worker = self.auto_scaler.workers[worker_id]
        success = worker.submit_request(request)
        
        if success:
            self.active_requests[request_id] = {
                'worker_id': worker_id,
                'timestamp': time.time()
            }
            self.total_requests_served += 1
            return request_id
        else:
            logger.warning(f"Failed to submit request to worker {worker_id}")
            return None
    
    def get_inference_result(self, request_id: str, timeout: float = 10.0) -> Optional[Dict[str, Any]]:
        """
        Get inference result by request ID.
        
        Args:
            request_id: Request ID to retrieve
            timeout: Timeout in seconds
            
        Returns:
            Inference result or None if not available
        """
        if request_id not in self.active_requests:
            return None
        
        worker_id = self.active_requests[request_id]['worker_id']
        worker = self.auto_scaler.workers.get(worker_id)
        
        if not worker:
            return None
        
        # Wait for result with timeout
        start_time = time.time()
        while time.time() - start_time < timeout:
            if request_id in worker.result_cache:
                result = worker.result_cache.pop(request_id)
                
                # Clean up tracking
                del self.active_requests[request_id]
                
                # Record in history
                self.request_history.append({
                    'request_id': request_id,
                    'worker_id': worker_id,
                    'latency_ms': result['latency_ms'],
                    'timestamp': result['timestamp']
                })
                
                # Maintain history size
                if len(self.request_history) > 1000:
                    self.request_history = self.request_history[-500:]
                
                return result
            
            time.sleep(0.01)  # 10ms polling interval
        
        # Timeout - clean up
        if request_id in self.active_requests:
            del self.active_requests[request_id]
        
        return None
    
    async def async_inference(self, input_data: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Asynchronous inference method.
        
        Args:
            input_data: Input data for inference
            
        Returns:
            Inference result
        """
        request_id = self.submit_inference_request(input_data)
        if not request_id:
            return None
        
        # Async wait for result
        result = None
        for _ in range(1000):  # 10 second timeout
            result = self.get_inference_result(request_id, timeout=0.01)
            if result:
                break
            await asyncio.sleep(0.01)
        
        return result
    
    def get_server_status(self) -> Dict[str, Any]:
        """Get comprehensive server status."""
        uptime_hours = (time.time() - self.start_time) / 3600
        
        # Calculate recent performance metrics
        recent_requests = self.request_history[-100:] if self.request_history else []
        avg_latency = np.mean([r['latency_ms'] for r in recent_requests]) if recent_requests else 0.0
        throughput_rps = len(recent_requests) / min(uptime_hours * 3600, 3600)  # RPS over last hour
        
        return {
            'server_uptime_hours': uptime_hours,
            'total_requests_served': self.total_requests_served,
            'active_requests': len(self.active_requests),
            'avg_latency_ms': avg_latency,
            'throughput_requests_per_sec': throughput_rps,
            'scaling_status': self.auto_scaler.get_scaling_status(),
            'model_task': self.model_task,
            'server_health': 'healthy' if self.auto_scaler.workers else 'no_workers'
        }


# Convenience functions for easy deployment
def create_distributed_server(model_task: str = "mnist",
                            min_instances: int = 1,
                            max_instances: int = 5,
                            target_latency_ms: float = 1.0) -> DistributedInferenceServer:
    """
    Create distributed inference server with sensible defaults.
    
    Args:
        model_task: Model task type
        min_instances: Minimum worker instances
        max_instances: Maximum worker instances
        target_latency_ms: Target latency threshold
        
    Returns:
        Configured distributed inference server
    """
    scaling_config = ScalingConfig(
        min_instances=min_instances,
        max_instances=max_instances,
        target_latency_ms=target_latency_ms,
        enable_auto_scaling_alerts=True
    )
    
    server = DistributedInferenceServer(model_task, scaling_config)
    server.start_server()
    
    return server


def benchmark_distributed_system(server: DistributedInferenceServer,
                                num_requests: int = 1000,
                                concurrent_requests: int = 10) -> Dict[str, float]:
    """
    Benchmark distributed inference system performance.
    
    Args:
        server: Distributed inference server
        num_requests: Total number of requests to send
        concurrent_requests: Number of concurrent requests
        
    Returns:
        Performance benchmark results
    """
    logger.info(f"Starting distributed system benchmark ({num_requests} requests)")
    
    # Generate test data
    if server.model_task == "mnist":
        test_data = [np.random.randn(784) * 0.1 + 0.5 for _ in range(num_requests)]
    elif server.model_task == "cifar10":
        test_data = [np.random.randn(3072) * 0.1 + 0.5 for _ in range(num_requests)]
    else:  # vowel_classification
        test_data = [np.random.randn(10) * 0.3 + 0.5 for _ in range(num_requests)]
    
    # Run benchmark
    start_time = time.perf_counter()
    completed_requests = 0
    latencies = []
    
    def process_requests(request_batch):
        """Process batch of requests."""
        nonlocal completed_requests, latencies
        
        for data in request_batch:
            req_start = time.perf_counter()
            request_id = server.submit_inference_request(data)
            
            if request_id:
                result = server.get_inference_result(request_id, timeout=10.0)
                if result:
                    req_end = time.perf_counter()
                    latencies.append((req_end - req_start) * 1000)  # Convert to ms
                    completed_requests += 1
    
    # Execute requests with concurrency
    with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
        batch_size = max(1, num_requests // concurrent_requests)
        futures = []
        
        for i in range(0, num_requests, batch_size):
            batch = test_data[i:i + batch_size]
            futures.append(executor.submit(process_requests, batch))
        
        # Wait for completion
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Benchmark request failed: {e}")
    
    end_time = time.perf_counter()
    
    # Calculate metrics
    total_time_s = end_time - start_time
    success_rate = completed_requests / num_requests
    avg_latency_ms = np.mean(latencies) if latencies else 0.0
    p95_latency_ms = np.percentile(latencies, 95) if latencies else 0.0
    throughput_rps = completed_requests / total_time_s
    
    results = {
        'total_requests': num_requests,
        'completed_requests': completed_requests,
        'success_rate': success_rate,
        'total_time_s': total_time_s,
        'avg_latency_ms': avg_latency_ms,
        'p95_latency_ms': p95_latency_ms,
        'throughput_requests_per_sec': throughput_rps,
        'concurrent_requests': concurrent_requests
    }
    
    logger.info(f"Benchmark completed: {throughput_rps:.1f} RPS, {avg_latency_ms:.2f}ms avg latency")
    
    return results