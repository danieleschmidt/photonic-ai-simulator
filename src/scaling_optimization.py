"""
Scaling and Performance Optimization Framework.

Implements advanced concurrency, distributed processing, auto-scaling,
and performance optimization techniques for photonic AI systems at scale.
"""

import numpy as np
import time
import threading
import multiprocessing as mp
import asyncio
import concurrent.futures
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import queue
import weakref
import psutil
import logging
from pathlib import Path
import json
import pickle
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import functools


class ScalingStrategy(Enum):
    """Auto-scaling strategies."""
    REACTIVE = "reactive"
    PREDICTIVE = "predictive"
    HYBRID = "hybrid"


class ResourceType(Enum):
    """Types of computational resources."""
    CPU = "cpu"
    GPU = "gpu"
    MEMORY = "memory"
    NETWORK = "network"
    OPTICAL = "optical"


@dataclass
class ResourceMetrics:
    """Real-time resource usage metrics."""
    cpu_usage_percent: float
    memory_usage_percent: float
    gpu_usage_percent: float
    network_bandwidth_mbps: float
    optical_channel_utilization: float
    queue_depth: int
    throughput_ops_per_sec: float
    latency_p50_ms: float
    latency_p95_ms: float
    timestamp: float


@dataclass
class ScalingPolicy:
    """Auto-scaling policy configuration."""
    target_cpu_utilization: float = 70.0
    target_memory_utilization: float = 80.0
    scale_up_threshold: float = 85.0
    scale_down_threshold: float = 30.0
    min_instances: int = 1
    max_instances: int = 100
    scale_up_cooldown_seconds: int = 300
    scale_down_cooldown_seconds: int = 600
    predictive_window_minutes: int = 30


class HighPerformanceProcessingPool:
    """
    High-performance processing pool with advanced optimization.
    
    Implements work-stealing, priority queues, and adaptive load balancing
    for optimal performance under varying workloads.
    """
    
    def __init__(self, 
                 num_workers: int = None,
                 max_queue_size: int = 10000,
                 enable_work_stealing: bool = True,
                 enable_priority_queue: bool = True,
                 worker_type: str = "thread"):  # "thread" or "process"
        
        self.num_workers = num_workers or mp.cpu_count()
        self.max_queue_size = max_queue_size
        self.enable_work_stealing = enable_work_stealing
        self.enable_priority_queue = enable_priority_queue
        self.worker_type = worker_type
        
        # Work queues
        if enable_priority_queue:
            self.work_queue = queue.PriorityQueue(maxsize=max_queue_size)
        else:
            self.work_queue = queue.Queue(maxsize=max_queue_size)
        
        # Worker-specific queues for work stealing
        if enable_work_stealing:
            self.worker_queues = [queue.Queue(maxsize=100) for _ in range(self.num_workers)]
        
        # Workers
        self.workers = []
        self.running = True
        
        # Performance tracking
        self.metrics = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_execution_time": 0.0,
            "worker_utilization": [0.0] * self.num_workers
        }
        self.metrics_lock = threading.Lock()
        
        # Result futures
        self.pending_futures = {}
        self.future_id_counter = 0
        self.future_lock = threading.Lock()
        
        # Start workers
        self._start_workers()
        
        self.logger = logging.getLogger("high_performance_pool")
        self.logger.info(f"Started {self.num_workers} {worker_type} workers")
    
    def _start_workers(self):
        """Start worker threads/processes."""
        if self.worker_type == "thread":
            for i in range(self.num_workers):
                worker = threading.Thread(
                    target=self._worker_loop,
                    args=(i,),
                    daemon=True
                )
                worker.start()
                self.workers.append(worker)
        else:
            # Process-based workers would be implemented here
            # For now, fall back to thread-based
            self._start_workers()
    
    def _worker_loop(self, worker_id: int):
        """Main worker loop."""
        last_activity = time.time()
        
        while self.running:
            try:
                # Try to get work from worker-specific queue first (work stealing)
                task = None
                if self.enable_work_stealing and self.worker_queues:
                    try:
                        task = self.worker_queues[worker_id].get_nowait()
                    except queue.Empty:
                        pass
                
                # If no work in local queue, try global queue
                if task is None:
                    try:
                        task = self.work_queue.get(timeout=1.0)
                    except queue.Empty:
                        # Try stealing work from other workers
                        if self.enable_work_stealing:
                            task = self._steal_work(worker_id)
                        
                        if task is None:
                            continue
                
                # Execute task
                start_time = time.perf_counter()
                self._execute_task(task)
                execution_time = time.perf_counter() - start_time
                
                # Update metrics
                with self.metrics_lock:
                    self.metrics["tasks_completed"] += 1
                    self.metrics["total_execution_time"] += execution_time
                    
                    # Update worker utilization
                    current_time = time.time()
                    work_ratio = min(1.0, execution_time / (current_time - last_activity))
                    self.metrics["worker_utilization"][worker_id] = work_ratio
                    last_activity = current_time
                
            except Exception as e:
                with self.metrics_lock:
                    self.metrics["tasks_failed"] += 1
                self.logger.error(f"Worker {worker_id} error: {e}")
    
    def _steal_work(self, worker_id: int) -> Optional[Any]:
        """Steal work from other workers."""
        for i, worker_queue in enumerate(self.worker_queues):
            if i != worker_id:
                try:
                    return worker_queue.get_nowait()
                except queue.Empty:
                    continue
        return None
    
    def _execute_task(self, task):
        """Execute a task and handle result."""
        task_id, func, args, kwargs, priority = task
        
        try:
            result = func(*args, **kwargs)
            self._complete_task(task_id, result, None)
        except Exception as e:
            self._complete_task(task_id, None, e)
    
    def _complete_task(self, task_id: int, result: Any, exception: Exception):
        """Complete a task and set future result."""
        with self.future_lock:
            future = self.pending_futures.pop(task_id, None)
        
        if future:
            if exception:
                future.set_exception(exception)
            else:
                future.set_result(result)
    
    def submit(self, func: Callable, *args, priority: int = 0, **kwargs) -> concurrent.futures.Future:
        """
        Submit a task for execution.
        
        Args:
            func: Function to execute
            *args: Function arguments
            priority: Task priority (lower = higher priority)
            **kwargs: Function keyword arguments
            
        Returns:
            Future object for result
        """
        with self.future_lock:
            task_id = self.future_id_counter
            self.future_id_counter += 1
            
            future = concurrent.futures.Future()
            self.pending_futures[task_id] = future
        
        task = (task_id, func, args, kwargs, priority)
        
        # Add to appropriate queue
        if self.enable_work_stealing:
            # Use round-robin for work distribution
            worker_queue = self.worker_queues[task_id % len(self.worker_queues)]
            try:
                worker_queue.put_nowait(task)
            except queue.Full:
                # Fall back to global queue
                self.work_queue.put(task)
        else:
            if self.enable_priority_queue:
                self.work_queue.put((priority, task))
            else:
                self.work_queue.put(task)
        
        return future
    
    def map(self, func: Callable, iterable, priority: int = 0) -> List[Any]:
        """
        Apply function to all items in iterable.
        
        Args:
            func: Function to apply
            iterable: Items to process
            priority: Task priority
            
        Returns:
            List of results
        """
        futures = [self.submit(func, item, priority=priority) for item in iterable]
        return [future.result() for future in futures]
    
    def shutdown(self, wait: bool = True):
        """Shutdown the processing pool."""
        self.running = False
        
        if wait:
            for worker in self.workers:
                if hasattr(worker, 'join'):
                    worker.join(timeout=5.0)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get pool performance metrics."""
        with self.metrics_lock:
            return {
                "num_workers": self.num_workers,
                "tasks_completed": self.metrics["tasks_completed"],
                "tasks_failed": self.metrics["tasks_failed"],
                "avg_execution_time": (
                    self.metrics["total_execution_time"] / max(self.metrics["tasks_completed"], 1)
                ),
                "worker_utilization": self.metrics["worker_utilization"].copy(),
                "avg_worker_utilization": sum(self.metrics["worker_utilization"]) / len(self.metrics["worker_utilization"]),
                "queue_size": self.work_queue.qsize()
            }


class AdaptiveLoadBalancer:
    """
    Adaptive load balancer for distributed photonic AI processing.
    
    Dynamically distributes workload across available resources based on
    real-time performance metrics and predicted demand.
    """
    
    def __init__(self, 
                 workers: List[Any],
                 balancing_algorithm: str = "weighted_round_robin"):
        
        self.workers = workers
        self.balancing_algorithm = balancing_algorithm
        
        # Worker metrics
        self.worker_metrics = {i: ResourceMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, time.time()) 
                              for i in range(len(workers))}
        self.metrics_lock = threading.Lock()
        
        # Load balancing state
        self.current_worker = 0
        self.worker_weights = [1.0] * len(workers)
        
        # Performance tracking
        self.response_times = {i: [] for i in range(len(workers))}
        self.error_counts = {i: 0 for i in range(len(workers))}
        
        self.logger = logging.getLogger("adaptive_load_balancer")
    
    def update_worker_metrics(self, worker_id: int, metrics: ResourceMetrics):
        """Update metrics for a worker."""
        with self.metrics_lock:
            self.worker_metrics[worker_id] = metrics
            
            # Update worker weights based on performance
            self._update_worker_weights()
    
    def _update_worker_weights(self):
        """Update worker weights based on current metrics."""
        for worker_id, metrics in self.worker_metrics.items():
            # Calculate weight based on inverse of load
            cpu_factor = max(0.1, 1.0 - metrics.cpu_usage_percent / 100.0)
            memory_factor = max(0.1, 1.0 - metrics.memory_usage_percent / 100.0)
            latency_factor = max(0.1, 1.0 / (1.0 + metrics.latency_p95_ms / 1000.0))
            
            # Recent error rate
            error_rate = self.error_counts[worker_id] / max(1, len(self.response_times[worker_id]))
            error_factor = max(0.1, 1.0 - error_rate)
            
            # Combined weight
            weight = cpu_factor * memory_factor * latency_factor * error_factor
            self.worker_weights[worker_id] = weight
    
    def select_worker(self) -> int:
        """Select optimal worker based on balancing algorithm."""
        if self.balancing_algorithm == "round_robin":
            return self._round_robin()
        elif self.balancing_algorithm == "weighted_round_robin":
            return self._weighted_round_robin()
        elif self.balancing_algorithm == "least_connections":
            return self._least_connections()
        elif self.balancing_algorithm == "least_response_time":
            return self._least_response_time()
        else:
            return self._weighted_round_robin()  # Default
    
    def _round_robin(self) -> int:
        """Simple round-robin selection."""
        worker_id = self.current_worker
        self.current_worker = (self.current_worker + 1) % len(self.workers)
        return worker_id
    
    def _weighted_round_robin(self) -> int:
        """Weighted round-robin based on worker performance."""
        total_weight = sum(self.worker_weights)
        if total_weight == 0:
            return self._round_robin()
        
        # Normalize weights
        normalized_weights = [w / total_weight for w in self.worker_weights]
        
        # Select based on weights
        import random
        return random.choices(range(len(self.workers)), weights=normalized_weights)[0]
    
    def _least_connections(self) -> int:
        """Select worker with least active connections."""
        with self.metrics_lock:
            queue_depths = [metrics.queue_depth for metrics in self.worker_metrics.values()]
        
        return queue_depths.index(min(queue_depths))
    
    def _least_response_time(self) -> int:
        """Select worker with best response time."""
        avg_response_times = []
        
        for worker_id in range(len(self.workers)):
            response_times = self.response_times[worker_id]
            if response_times:
                avg_time = sum(response_times[-10:]) / min(len(response_times), 10)  # Last 10
            else:
                avg_time = float('inf')
            avg_response_times.append(avg_time)
        
        return avg_response_times.index(min(avg_response_times))
    
    def record_response_time(self, worker_id: int, response_time: float, success: bool):
        """Record response time for a worker."""
        self.response_times[worker_id].append(response_time)
        
        # Keep only recent history
        if len(self.response_times[worker_id]) > 100:
            self.response_times[worker_id] = self.response_times[worker_id][-100:]
        
        if not success:
            self.error_counts[worker_id] += 1


class AutoScaler:
    """
    Intelligent auto-scaling system for photonic AI workloads.
    
    Monitors system metrics and automatically scales resources up or down
    based on demand patterns and performance requirements.
    """
    
    def __init__(self, 
                 scaling_policy: ScalingPolicy,
                 resource_manager: 'ResourceManager'):
        
        self.policy = scaling_policy
        self.resource_manager = resource_manager
        
        # Scaling state
        self.current_instances = scaling_policy.min_instances
        self.last_scale_up = 0
        self.last_scale_down = 0
        
        # Metrics history for predictive scaling
        self.metrics_history = []
        self.max_history_size = 1000
        
        # Scaling decisions log
        self.scaling_decisions = []
        
        self.logger = logging.getLogger("auto_scaler")
        
        # Start monitoring thread
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def _monitor_loop(self):
        """Main monitoring loop for auto-scaling."""
        while self.monitoring:
            try:
                # Get current metrics
                metrics = self.resource_manager.get_current_metrics()
                
                # Store metrics history
                self.metrics_history.append(metrics)
                if len(self.metrics_history) > self.max_history_size:
                    self.metrics_history.pop(0)
                
                # Make scaling decision
                scaling_decision = self._make_scaling_decision(metrics)
                
                if scaling_decision != 0:
                    self._execute_scaling(scaling_decision)
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                time.sleep(60)  # Back off on error
    
    def _make_scaling_decision(self, metrics: ResourceMetrics) -> int:
        """
        Make scaling decision based on current metrics.
        
        Returns:
            Positive for scale up, negative for scale down, 0 for no change
        """
        current_time = time.time()
        
        # Check cooldown periods
        if current_time - self.last_scale_up < self.policy.scale_up_cooldown_seconds:
            if metrics.cpu_usage_percent > self.policy.scale_up_threshold:
                return 0  # Still in cooldown
        
        if current_time - self.last_scale_down < self.policy.scale_down_cooldown_seconds:
            if metrics.cpu_usage_percent < self.policy.scale_down_threshold:
                return 0  # Still in cooldown
        
        # Scale up conditions
        if (metrics.cpu_usage_percent > self.policy.scale_up_threshold or
            metrics.memory_usage_percent > self.policy.scale_up_threshold):
            
            if self.current_instances < self.policy.max_instances:
                # Calculate how many instances to add
                load_factor = max(
                    metrics.cpu_usage_percent / 100.0,
                    metrics.memory_usage_percent / 100.0
                )
                
                instances_needed = max(1, int(self.current_instances * load_factor - self.current_instances))
                return min(instances_needed, self.policy.max_instances - self.current_instances)
        
        # Scale down conditions
        elif (metrics.cpu_usage_percent < self.policy.scale_down_threshold and
              metrics.memory_usage_percent < self.policy.scale_down_threshold):
            
            if self.current_instances > self.policy.min_instances:
                # Be conservative with scale down
                target_instances = max(
                    self.policy.min_instances,
                    int(self.current_instances * 0.8)  # Scale down by 20%
                )
                return target_instances - self.current_instances
        
        return 0
    
    def _execute_scaling(self, scaling_decision: int):
        """Execute scaling decision."""
        current_time = time.time()
        
        if scaling_decision > 0:
            # Scale up
            new_instances = min(
                self.current_instances + scaling_decision,
                self.policy.max_instances
            )
            
            if self.resource_manager.scale_up(new_instances - self.current_instances):
                self.current_instances = new_instances
                self.last_scale_up = current_time
                
                self.logger.info(f"Scaled up to {self.current_instances} instances")
                
                self.scaling_decisions.append({
                    "timestamp": current_time,
                    "action": "scale_up",
                    "instances": self.current_instances,
                    "reason": "high_utilization"
                })
        
        elif scaling_decision < 0:
            # Scale down
            new_instances = max(
                self.current_instances + scaling_decision,
                self.policy.min_instances
            )
            
            if self.resource_manager.scale_down(self.current_instances - new_instances):
                self.current_instances = new_instances
                self.last_scale_down = current_time
                
                self.logger.info(f"Scaled down to {self.current_instances} instances")
                
                self.scaling_decisions.append({
                    "timestamp": current_time,
                    "action": "scale_down",
                    "instances": self.current_instances,
                    "reason": "low_utilization"
                })
    
    def stop_monitoring(self):
        """Stop auto-scaling monitoring."""
        self.monitoring = False
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)


class ResourceManager:
    """
    Comprehensive resource management for photonic AI systems.
    
    Manages CPU, GPU, memory, and optical resources with intelligent
    allocation and optimization strategies.
    """
    
    def __init__(self):
        self.active_resources = {}
        self.resource_pools = {
            ResourceType.CPU: HighPerformanceProcessingPool(worker_type="thread"),
            ResourceType.GPU: None,  # Would be implemented with GPU pools
            ResourceType.MEMORY: None,  # Memory pool management
            ResourceType.OPTICAL: None  # Optical channel management
        }
        
        self.resource_lock = threading.Lock()
        self.allocation_history = []
        
        self.logger = logging.getLogger("resource_manager")
    
    def allocate_resources(self, 
                         resource_requirements: Dict[ResourceType, float],
                         priority: int = 0) -> Dict[ResourceType, Any]:
        """
        Allocate resources based on requirements.
        
        Args:
            resource_requirements: Required resources by type
            priority: Allocation priority
            
        Returns:
            Dictionary of allocated resources
        """
        with self.resource_lock:
            allocated = {}
            
            for resource_type, amount in resource_requirements.items():
                if resource_type in self.resource_pools and self.resource_pools[resource_type]:
                    # Allocate from pool
                    allocated[resource_type] = self.resource_pools[resource_type]
                else:
                    # Create new resource allocation
                    allocated[resource_type] = self._create_resource(resource_type, amount)
            
            # Record allocation
            self.allocation_history.append({
                "timestamp": time.time(),
                "requirements": resource_requirements,
                "allocated": allocated,
                "priority": priority
            })
            
            return allocated
    
    def _create_resource(self, resource_type: ResourceType, amount: float) -> Any:
        """Create new resource allocation."""
        if resource_type == ResourceType.CPU:
            return HighPerformanceProcessingPool(
                num_workers=max(1, int(amount * mp.cpu_count()))
            )
        elif resource_type == ResourceType.MEMORY:
            # Memory pool would be implemented here
            return {"memory_mb": amount * 1024}
        else:
            return {"type": resource_type, "amount": amount}
    
    def get_current_metrics(self) -> ResourceMetrics:
        """Get current resource utilization metrics."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # Network metrics (simplified)
            network_stats = psutil.net_io_counters()
            network_mbps = (network_stats.bytes_sent + network_stats.bytes_recv) / (1024 * 1024)
            
            return ResourceMetrics(
                cpu_usage_percent=cpu_percent,
                memory_usage_percent=memory.percent,
                gpu_usage_percent=0.0,  # Would use GPU monitoring library
                network_bandwidth_mbps=network_mbps,
                optical_channel_utilization=0.0,  # Custom metric
                queue_depth=0,  # From processing pools
                throughput_ops_per_sec=0.0,  # Calculated metric
                latency_p50_ms=0.0,  # From performance tracking
                latency_p95_ms=0.0,
                timestamp=time.time()
            )
        except Exception as e:
            self.logger.error(f"Failed to get metrics: {e}")
            return ResourceMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, time.time())
    
    def scale_up(self, additional_instances: int) -> bool:
        """Scale up resources."""
        try:
            # Implementation would add new worker instances
            self.logger.info(f"Scaling up by {additional_instances} instances")
            return True
        except Exception as e:
            self.logger.error(f"Scale up failed: {e}")
            return False
    
    def scale_down(self, instances_to_remove: int) -> bool:
        """Scale down resources."""
        try:
            # Implementation would remove worker instances
            self.logger.info(f"Scaling down by {instances_to_remove} instances")
            return True
        except Exception as e:
            self.logger.error(f"Scale down failed: {e}")
            return False


class DistributedComputeManager:
    """
    Distributed compute management for multi-node photonic AI processing.
    
    Coordinates computation across multiple nodes with fault tolerance
    and optimal work distribution.
    """
    
    def __init__(self, node_addresses: List[str]):
        self.node_addresses = node_addresses
        self.node_status = {addr: "unknown" for addr in node_addresses}
        self.node_capabilities = {}
        
        # Work distribution
        self.pending_work = queue.Queue()
        self.completed_work = {}
        self.work_assignments = {}
        
        # Fault tolerance
        self.failed_nodes = set()
        self.redundancy_factor = 2
        
        self.logger = logging.getLogger("distributed_compute")
        
        # Initialize nodes
        self._initialize_nodes()
    
    def _initialize_nodes(self):
        """Initialize connections to compute nodes."""
        for addr in self.node_addresses:
            try:
                # In a real implementation, this would establish connections
                self.node_status[addr] = "available"
                self.node_capabilities[addr] = {
                    "cpu_cores": 8,
                    "memory_gb": 32,
                    "gpu_count": 1,
                    "optical_channels": 16
                }
                self.logger.info(f"Initialized node: {addr}")
            except Exception as e:
                self.logger.error(f"Failed to initialize node {addr}: {e}")
                self.node_status[addr] = "failed"
                self.failed_nodes.add(addr)
    
    def submit_distributed_task(self, 
                              task_func: Callable,
                              task_data: List[Any],
                              redundancy: bool = True) -> List[concurrent.futures.Future]:
        """
        Submit task for distributed execution.
        
        Args:
            task_func: Function to execute
            task_data: Data to process (will be distributed)
            redundancy: Whether to use redundant execution
            
        Returns:
            List of futures for results
        """
        available_nodes = [addr for addr, status in self.node_status.items() 
                          if status == "available"]
        
        if not available_nodes:
            raise RuntimeError("No available compute nodes")
        
        # Partition data across nodes
        chunks = self._partition_data(task_data, len(available_nodes))
        
        futures = []
        for i, (chunk, node) in enumerate(zip(chunks, available_nodes)):
            future = self._submit_to_node(node, task_func, chunk, redundancy)
            futures.append(future)
        
        return futures
    
    def _partition_data(self, data: List[Any], num_partitions: int) -> List[List[Any]]:
        """Partition data across compute nodes."""
        chunk_size = len(data) // num_partitions
        chunks = []
        
        for i in range(num_partitions):
            start_idx = i * chunk_size
            if i == num_partitions - 1:  # Last chunk gets remaining data
                end_idx = len(data)
            else:
                end_idx = (i + 1) * chunk_size
            
            chunks.append(data[start_idx:end_idx])
        
        return chunks
    
    def _submit_to_node(self, 
                       node_addr: str,
                       task_func: Callable,
                       data_chunk: List[Any],
                       redundancy: bool) -> concurrent.futures.Future:
        """Submit task to specific compute node."""
        # In a real implementation, this would use network communication
        # For now, simulate with local execution
        
        future = concurrent.futures.Future()
        
        def execute_task():
            try:
                result = [task_func(item) for item in data_chunk]
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)
        
        # Execute in background thread (simulating remote execution)
        threading.Thread(target=execute_task, daemon=True).start()
        
        return future
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get overall cluster status."""
        available_nodes = sum(1 for status in self.node_status.values() if status == "available")
        total_nodes = len(self.node_addresses)
        
        total_cpu_cores = sum(
            capabilities.get("cpu_cores", 0) 
            for addr, capabilities in self.node_capabilities.items()
            if self.node_status[addr] == "available"
        )
        
        return {
            "total_nodes": total_nodes,
            "available_nodes": available_nodes,
            "failed_nodes": len(self.failed_nodes),
            "total_cpu_cores": total_cpu_cores,
            "cluster_health": available_nodes / total_nodes if total_nodes > 0 else 0
        }


def optimize_for_scale(cache_size: int = 1000, 
                      enable_parallelization: bool = True,
                      max_workers: int = None):
    """
    Decorator to optimize functions for scale.
    
    Args:
        cache_size: Size of result cache
        enable_parallelization: Whether to enable parallel execution
        max_workers: Maximum number of parallel workers
    """
    def decorator(func: Callable) -> Callable:
        # Function cache
        cache = {}
        cache_lock = threading.Lock()
        
        # Parallel execution pool
        if enable_parallelization:
            executor = ThreadPoolExecutor(max_workers=max_workers)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            cache_key = (args, tuple(sorted(kwargs.items())))
            
            # Check cache
            with cache_lock:
                if cache_key in cache:
                    return cache[cache_key]
            
            # Execute function
            if enable_parallelization and hasattr(args[0], '__iter__') and len(args) == 1:
                # Parallel execution for iterable inputs
                iterable = args[0]
                if len(iterable) > 10:  # Only parallelize for larger inputs
                    futures = [executor.submit(func, item) for item in iterable]
                    results = [future.result() for future in futures]
                else:
                    results = func(*args, **kwargs)
            else:
                results = func(*args, **kwargs)
            
            # Cache result
            with cache_lock:
                if len(cache) >= cache_size:
                    # Remove oldest entry (simple LRU)
                    oldest_key = next(iter(cache))
                    del cache[oldest_key]
                
                cache[cache_key] = results
            
            return results
        
        return wrapper
    return decorator


# Example usage and testing
if __name__ == "__main__":
    # Test high-performance processing pool
    pool = HighPerformanceProcessingPool(num_workers=4)
    
    def test_task(x):
        time.sleep(0.1)  # Simulate work
        return x * x
    
    # Submit tasks
    futures = [pool.submit(test_task, i) for i in range(20)]
    results = [future.result() for future in futures]
    
    print(f"Processed {len(results)} tasks")
    print(f"Pool metrics: {pool.get_metrics()}")
    
    # Test auto-scaler
    policy = ScalingPolicy(min_instances=2, max_instances=10)
    resource_manager = ResourceManager()
    autoscaler = AutoScaler(policy, resource_manager)
    
    print(f"Auto-scaler initialized with policy: {policy}")
    
    # Cleanup
    pool.shutdown()
    autoscaler.stop_monitoring()