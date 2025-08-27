"""
Distributed Photonic Computing System

Implements distributed processing, auto-scaling, and load balancing for 
large-scale photonic neural network deployments across multiple nodes.
"""

import asyncio
import time
import threading
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict, deque
import json
import uuid
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp

try:
    from .optimization import create_optimized_network, OptimizedPhotonicNeuralNetwork
    from .circuit_breaker import CircuitBreaker, create_circuit_breaker
    from .utils.monitoring import SystemMetrics
    from .robust_error_handling import RobustErrorHandler
except ImportError:
    from optimization import create_optimized_network, OptimizedPhotonicNeuralNetwork  
    from circuit_breaker import CircuitBreaker, create_circuit_breaker
    from utils.monitoring import SystemMetrics
    from robust_error_handling import RobustErrorHandler

logger = logging.getLogger(__name__)


class NodeStatus(Enum):
    """Distributed node status."""
    HEALTHY = "healthy"
    BUSY = "busy"
    OVERLOADED = "overloaded"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class NodeConfig:
    """Configuration for a distributed node."""
    node_id: str
    host: str = "localhost"
    port: int = 8080
    max_concurrent_tasks: int = 10
    cpu_cores: int = mp.cpu_count()
    memory_gb: float = 8.0
    gpu_available: bool = False
    specialized_hardware: List[str] = field(default_factory=list)


@dataclass
class InferenceTask:
    """Distributed inference task."""
    task_id: str
    input_data: np.ndarray
    model_name: str
    priority: TaskPriority = TaskPriority.NORMAL
    timeout: float = 30.0
    created_at: float = field(default_factory=time.time)
    client_id: Optional[str] = None
    callback_url: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NodeMetrics:
    """Metrics for a distributed node."""
    node_id: str
    status: NodeStatus
    cpu_usage_percent: float
    memory_usage_percent: float
    active_tasks: int
    completed_tasks: int
    failed_tasks: int
    avg_response_time_ms: float
    queue_length: int
    last_heartbeat: float
    uptime_seconds: float
    power_consumption_watts: Optional[float] = None


class DistributedNode:
    """Individual node in distributed photonic computing cluster."""
    
    def __init__(self, config: NodeConfig):
        """Initialize distributed node."""
        self.config = config
        self.status = NodeStatus.OFFLINE
        self.task_queue = asyncio.Queue()
        self.active_tasks: Dict[str, InferenceTask] = {}
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.start_time = time.time()
        
        # Task processing
        self.executor = ThreadPoolExecutor(max_workers=config.max_concurrent_tasks)
        self.photonic_networks: Dict[str, OptimizedPhotonicNeuralNetwork] = {}
        
        # Monitoring
        self.circuit_breaker = create_circuit_breaker(
            f"node_{config.node_id}",
            failure_threshold=5,
            recovery_timeout=60.0
        )
        self.error_handler = RobustErrorHandler(f"node_{config.node_id}")
        
        # Metrics tracking
        self.response_times = deque(maxlen=100)
        self.last_heartbeat = time.time()
        
        logger.info(f"Initialized distributed node: {config.node_id}")
    
    async def start(self):
        """Start the distributed node."""
        self.status = NodeStatus.HEALTHY
        self.last_heartbeat = time.time()
        
        # Start task processing loop
        asyncio.create_task(self._process_tasks())
        
        # Initialize photonic networks
        await self._initialize_models()
        
        logger.info(f"Node {self.config.node_id} started successfully")
    
    async def stop(self):
        """Stop the distributed node gracefully."""
        self.status = NodeStatus.OFFLINE
        
        # Wait for active tasks to complete (with timeout)
        timeout = 30.0
        start_time = time.time()
        
        while self.active_tasks and (time.time() - start_time) < timeout:
            await asyncio.sleep(1.0)
        
        # Force stop executor
        self.executor.shutdown(wait=True)
        
        logger.info(f"Node {self.config.node_id} stopped")
    
    async def submit_task(self, task: InferenceTask) -> bool:
        """
        Submit task for processing.
        
        Args:
            task: Inference task to process
            
        Returns:
            True if task was accepted
        """
        if self.status not in [NodeStatus.HEALTHY, NodeStatus.BUSY]:
            return False
        
        if len(self.active_tasks) >= self.config.max_concurrent_tasks:
            self.status = NodeStatus.OVERLOADED
            return False
        
        try:
            await self.task_queue.put(task)
            logger.debug(f"Task {task.task_id} queued on node {self.config.node_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to queue task {task.task_id}: {e}")
            return False
    
    async def _process_tasks(self):
        """Main task processing loop."""
        while self.status != NodeStatus.OFFLINE:
            try:
                # Get task from queue (with timeout)
                task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                
                # Submit task to executor
                future = self.executor.submit(self._execute_task, task)
                
                # Don't await here - let tasks run concurrently
                asyncio.create_task(self._handle_task_completion(task, future))
                
            except asyncio.TimeoutError:
                # No tasks available, continue
                continue
            except Exception as e:
                logger.error(f"Error in task processing loop: {e}")
                await asyncio.sleep(1.0)
    
    def _execute_task(self, task: InferenceTask) -> Dict[str, Any]:
        """Execute inference task."""
        start_time = time.time()
        self.active_tasks[task.task_id] = task
        
        try:
            # Update status if getting busy
            if len(self.active_tasks) > self.config.max_concurrent_tasks * 0.8:
                self.status = NodeStatus.BUSY
            
            # Get or load model
            if task.model_name not in self.photonic_networks:
                self.photonic_networks[task.model_name] = create_optimized_network(
                    task.model_name, "high"
                )
            
            network = self.photonic_networks[task.model_name]
            
            # Execute inference with circuit breaker protection
            @self.circuit_breaker
            def protected_inference():
                return network.optimized_forward(task.input_data, measure_latency=True)
            
            outputs, metrics = protected_inference()
            
            execution_time = time.time() - start_time
            self.response_times.append(execution_time * 1000)  # ms
            
            result = {
                "task_id": task.task_id,
                "outputs": outputs.tolist(),
                "execution_time_ms": execution_time * 1000,
                "metrics": metrics,
                "node_id": self.config.node_id,
                "success": True
            }
            
            self.completed_tasks += 1
            logger.debug(f"Task {task.task_id} completed in {execution_time*1000:.2f}ms")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.failed_tasks += 1
            
            error_result = {
                "task_id": task.task_id,
                "error": str(e),
                "execution_time_ms": execution_time * 1000,
                "node_id": self.config.node_id,
                "success": False
            }
            
            logger.error(f"Task {task.task_id} failed: {e}")
            return error_result
            
        finally:
            # Clean up
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]
            
            # Update status
            if len(self.active_tasks) <= self.config.max_concurrent_tasks * 0.5:
                self.status = NodeStatus.HEALTHY
    
    async def _handle_task_completion(self, task: InferenceTask, future):
        """Handle task completion and callbacks."""
        try:
            result = await asyncio.get_event_loop().run_in_executor(None, future.result)
            
            # Handle callback if specified
            if task.callback_url:
                await self._send_callback(task.callback_url, result)
                
        except Exception as e:
            logger.error(f"Error handling task completion: {e}")
    
    async def _send_callback(self, callback_url: str, result: Dict[str, Any]):
        """Send result to callback URL."""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(callback_url, json=result) as response:
                    if response.status == 200:
                        logger.debug(f"Callback sent successfully to {callback_url}")
                    else:
                        logger.warning(f"Callback failed with status {response.status}")
        except Exception as e:
            logger.error(f"Failed to send callback to {callback_url}: {e}")
    
    async def _initialize_models(self):
        """Initialize commonly used models."""
        common_models = ["mnist", "cifar10", "vowel_classification"]
        
        for model_name in common_models:
            try:
                self.photonic_networks[model_name] = create_optimized_network(
                    model_name, "medium"
                )
                logger.info(f"Initialized model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize model {model_name}: {e}")
    
    def get_metrics(self) -> NodeMetrics:
        """Get current node metrics."""
        import psutil
        
        return NodeMetrics(
            node_id=self.config.node_id,
            status=self.status,
            cpu_usage_percent=psutil.cpu_percent(),
            memory_usage_percent=psutil.virtual_memory().percent,
            active_tasks=len(self.active_tasks),
            completed_tasks=self.completed_tasks,
            failed_tasks=self.failed_tasks,
            avg_response_time_ms=np.mean(self.response_times) if self.response_times else 0,
            queue_length=self.task_queue.qsize(),
            last_heartbeat=self.last_heartbeat,
            uptime_seconds=time.time() - self.start_time
        )
    
    def update_heartbeat(self):
        """Update heartbeat timestamp."""
        self.last_heartbeat = time.time()


class LoadBalancer:
    """Intelligent load balancer for distributed photonic computing."""
    
    def __init__(self, strategy: str = "least_loaded"):
        """
        Initialize load balancer.
        
        Args:
            strategy: Load balancing strategy ("round_robin", "least_loaded", "weighted")
        """
        self.strategy = strategy
        self.nodes: Dict[str, DistributedNode] = {}
        self.node_weights: Dict[str, float] = {}
        self.round_robin_index = 0
        self.task_history: Dict[str, List[float]] = defaultdict(list)
        
    def register_node(self, node: DistributedNode, weight: float = 1.0):
        """Register a node with the load balancer."""
        self.nodes[node.config.node_id] = node
        self.node_weights[node.config.node_id] = weight
        logger.info(f"Registered node {node.config.node_id} with weight {weight}")
    
    def unregister_node(self, node_id: str):
        """Unregister a node from the load balancer."""
        if node_id in self.nodes:
            del self.nodes[node_id]
            del self.node_weights[node_id]
            logger.info(f"Unregistered node {node_id}")
    
    async def submit_task(self, task: InferenceTask) -> Optional[str]:
        """
        Submit task using load balancing strategy.
        
        Args:
            task: Task to submit
            
        Returns:
            Node ID that accepted the task, or None if no nodes available
        """
        available_nodes = [
            node for node in self.nodes.values()
            if node.status in [NodeStatus.HEALTHY, NodeStatus.BUSY]
        ]
        
        if not available_nodes:
            logger.warning("No healthy nodes available for task submission")
            return None
        
        # Select node based on strategy
        selected_node = self._select_node(available_nodes, task)
        
        if selected_node:
            success = await selected_node.submit_task(task)
            if success:
                # Record task assignment for learning
                self.task_history[selected_node.config.node_id].append(time.time())
                return selected_node.config.node_id
        
        return None
    
    def _select_node(self, available_nodes: List[DistributedNode], 
                    task: InferenceTask) -> Optional[DistributedNode]:
        """Select best node based on strategy."""
        if self.strategy == "round_robin":
            return self._round_robin_selection(available_nodes)
        elif self.strategy == "least_loaded":
            return self._least_loaded_selection(available_nodes)
        elif self.strategy == "weighted":
            return self._weighted_selection(available_nodes)
        elif self.strategy == "priority_aware":
            return self._priority_aware_selection(available_nodes, task)
        else:
            return available_nodes[0] if available_nodes else None
    
    def _round_robin_selection(self, nodes: List[DistributedNode]) -> DistributedNode:
        """Round-robin node selection."""
        selected = nodes[self.round_robin_index % len(nodes)]
        self.round_robin_index += 1
        return selected
    
    def _least_loaded_selection(self, nodes: List[DistributedNode]) -> DistributedNode:
        """Select node with least current load."""
        def load_score(node):
            metrics = node.get_metrics()
            # Combine active tasks, CPU usage, and queue length
            return (
                metrics.active_tasks * 0.4 +
                metrics.cpu_usage_percent * 0.01 +
                metrics.queue_length * 0.3
            )
        
        return min(nodes, key=load_score)
    
    def _weighted_selection(self, nodes: List[DistributedNode]) -> DistributedNode:
        """Select node based on weights and current performance."""
        scores = []
        for node in nodes:
            weight = self.node_weights.get(node.config.node_id, 1.0)
            metrics = node.get_metrics()
            
            # Higher weight and lower load = higher score
            load_factor = 1.0 - (metrics.active_tasks / node.config.max_concurrent_tasks)
            score = weight * load_factor
            scores.append((score, node))
        
        # Select node with highest score
        return max(scores, key=lambda x: x[0])[1]
    
    def _priority_aware_selection(self, nodes: List[DistributedNode], 
                                 task: InferenceTask) -> DistributedNode:
        """Select node considering task priority."""
        # For high priority tasks, prefer nodes with lower load
        if task.priority in [TaskPriority.HIGH, TaskPriority.CRITICAL]:
            return self._least_loaded_selection(nodes)
        else:
            return self._weighted_selection(nodes)
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get comprehensive cluster status."""
        node_statuses = {}
        total_capacity = 0
        total_active = 0
        
        for node_id, node in self.nodes.items():
            metrics = node.get_metrics()
            node_statuses[node_id] = {
                "status": metrics.status.value,
                "active_tasks": metrics.active_tasks,
                "capacity": node.config.max_concurrent_tasks,
                "cpu_usage": metrics.cpu_usage_percent,
                "avg_response_time": metrics.avg_response_time_ms
            }
            
            total_capacity += node.config.max_concurrent_tasks
            total_active += metrics.active_tasks
        
        return {
            "total_nodes": len(self.nodes),
            "healthy_nodes": len([n for n in self.nodes.values() 
                                 if n.status == NodeStatus.HEALTHY]),
            "total_capacity": total_capacity,
            "total_active_tasks": total_active,
            "utilization_percent": (total_active / total_capacity * 100) if total_capacity > 0 else 0,
            "load_balancing_strategy": self.strategy,
            "nodes": node_statuses
        }


class AutoScaler:
    """Automatic scaling manager for photonic computing clusters."""
    
    def __init__(self, 
                 target_utilization: float = 70.0,
                 scale_up_threshold: float = 80.0,
                 scale_down_threshold: float = 30.0,
                 min_nodes: int = 2,
                 max_nodes: int = 20):
        """
        Initialize auto-scaler.
        
        Args:
            target_utilization: Target cluster utilization percentage
            scale_up_threshold: Utilization threshold to trigger scale up
            scale_down_threshold: Utilization threshold to trigger scale down
            min_nodes: Minimum number of nodes
            max_nodes: Maximum number of nodes
        """
        self.target_utilization = target_utilization
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        
        self.load_balancer: Optional[LoadBalancer] = None
        self.scaling_history = deque(maxlen=100)
        self.is_active = False
        
    def set_load_balancer(self, load_balancer: LoadBalancer):
        """Set the load balancer to monitor and scale."""
        self.load_balancer = load_balancer
    
    async def start_autoscaling(self):
        """Start automatic scaling monitoring."""
        if not self.load_balancer:
            raise ValueError("Load balancer not set")
        
        self.is_active = True
        logger.info("Started automatic scaling")
        
        while self.is_active:
            try:
                await self._evaluate_scaling()
                await asyncio.sleep(30.0)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"Error in auto-scaling loop: {e}")
                await asyncio.sleep(10.0)
    
    def stop_autoscaling(self):
        """Stop automatic scaling."""
        self.is_active = False
        logger.info("Stopped automatic scaling")
    
    async def _evaluate_scaling(self):
        """Evaluate if scaling action is needed."""
        cluster_status = self.load_balancer.get_cluster_status()
        
        current_utilization = cluster_status["utilization_percent"]
        healthy_nodes = cluster_status["healthy_nodes"]
        
        # Decide scaling action
        scaling_decision = None
        
        if (current_utilization > self.scale_up_threshold and 
            healthy_nodes < self.max_nodes):
            scaling_decision = "scale_up"
            
        elif (current_utilization < self.scale_down_threshold and 
              healthy_nodes > self.min_nodes):
            scaling_decision = "scale_down"
        
        if scaling_decision:
            await self._execute_scaling(scaling_decision, cluster_status)
    
    async def _execute_scaling(self, action: str, cluster_status: Dict[str, Any]):
        """Execute scaling action."""
        current_time = time.time()
        
        # Check if we've scaled recently (cooldown period)
        if (self.scaling_history and 
            current_time - self.scaling_history[-1]["timestamp"] < 120.0):  # 2 min cooldown
            return
        
        if action == "scale_up":
            await self._scale_up()
        elif action == "scale_down":
            await self._scale_down()
        
        # Record scaling action
        self.scaling_history.append({
            "action": action,
            "timestamp": current_time,
            "utilization": cluster_status["utilization_percent"],
            "nodes_before": cluster_status["healthy_nodes"]
        })
    
    async def _scale_up(self):
        """Add new nodes to cluster."""
        new_node_id = f"auto_node_{uuid.uuid4().hex[:8]}"
        config = NodeConfig(
            node_id=new_node_id,
            max_concurrent_tasks=8,
            cpu_cores=mp.cpu_count() // 2
        )
        
        new_node = DistributedNode(config)
        await new_node.start()
        
        self.load_balancer.register_node(new_node, weight=1.0)
        
        logger.info(f"Scaled up: Added node {new_node_id}")
    
    async def _scale_down(self):
        """Remove nodes from cluster."""
        # Find the least utilized node
        nodes = list(self.load_balancer.nodes.values())
        if len(nodes) <= self.min_nodes:
            return
        
        least_utilized = min(nodes, key=lambda n: n.get_metrics().active_tasks)
        
        # Graceful shutdown
        await least_utilized.stop()
        self.load_balancer.unregister_node(least_utilized.config.node_id)
        
        logger.info(f"Scaled down: Removed node {least_utilized.config.node_id}")
    
    def get_scaling_metrics(self) -> Dict[str, Any]:
        """Get auto-scaling metrics."""
        recent_actions = list(self.scaling_history)[-10:]  # Last 10 actions
        
        return {
            "is_active": self.is_active,
            "target_utilization": self.target_utilization,
            "scale_up_threshold": self.scale_up_threshold,
            "scale_down_threshold": self.scale_down_threshold,
            "min_nodes": self.min_nodes,
            "max_nodes": self.max_nodes,
            "recent_scaling_actions": recent_actions,
            "total_scaling_actions": len(self.scaling_history)
        }


# Factory functions
def create_distributed_cluster(node_configs: List[NodeConfig], 
                             load_balancing_strategy: str = "least_loaded") -> Tuple[List[DistributedNode], LoadBalancer]:
    """
    Create a distributed photonic computing cluster.
    
    Args:
        node_configs: List of node configurations
        load_balancing_strategy: Load balancing strategy
        
    Returns:
        Tuple of (nodes, load_balancer)
    """
    nodes = []
    load_balancer = LoadBalancer(strategy=load_balancing_strategy)
    
    for config in node_configs:
        node = DistributedNode(config)
        nodes.append(node)
        load_balancer.register_node(node)
    
    return nodes, load_balancer


def create_autoscaling_cluster(initial_nodes: int = 3,
                             max_nodes: int = 10,
                             target_utilization: float = 70.0) -> Tuple[List[DistributedNode], LoadBalancer, AutoScaler]:
    """
    Create an auto-scaling photonic computing cluster.
    
    Args:
        initial_nodes: Number of initial nodes
        max_nodes: Maximum number of nodes
        target_utilization: Target cluster utilization
        
    Returns:
        Tuple of (nodes, load_balancer, auto_scaler)
    """
    # Create initial node configurations
    node_configs = []
    for i in range(initial_nodes):
        config = NodeConfig(
            node_id=f"node_{i}",
            max_concurrent_tasks=8,
            cpu_cores=mp.cpu_count() // initial_nodes,
            memory_gb=8.0
        )
        node_configs.append(config)
    
    # Create cluster
    nodes, load_balancer = create_distributed_cluster(node_configs, "least_loaded")
    
    # Create auto-scaler
    auto_scaler = AutoScaler(
        target_utilization=target_utilization,
        min_nodes=initial_nodes,
        max_nodes=max_nodes
    )
    auto_scaler.set_load_balancer(load_balancer)
    
    return nodes, load_balancer, auto_scaler