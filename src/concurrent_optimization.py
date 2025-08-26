"""
Concurrent Processing Optimization for Photonic AI Systems.

Implements advanced concurrency patterns including parallel wavelength processing,
asynchronous inference pipelines, and intelligent resource pooling for maximum
throughput and efficiency.
"""

import numpy as np
import asyncio
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple, Any, Union, Callable, Awaitable
import queue
import time
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
import logging
from pathlib import Path
import psutil
from enum import Enum

try:
    from .advanced_caching import matrix_cache, inference_cache, compute_input_hash
    from .core import PhotonicProcessor, WavelengthConfig
    from .models import PhotonicNeuralNetwork
except ImportError:
    from advanced_caching import matrix_cache, inference_cache, compute_input_hash
    from core import PhotonicProcessor, WavelengthConfig
    from models import PhotonicNeuralNetwork

logger = logging.getLogger(__name__)


class ProcessingMode(Enum):
    """Processing execution modes."""
    SEQUENTIAL = "sequential"
    THREADED = "threaded"
    MULTIPROCESS = "multiprocess"
    HYBRID = "hybrid"
    ASYNC = "async"


@dataclass
class ConcurrencyConfig:
    """Configuration for concurrent processing."""
    max_threads: Optional[int] = None
    max_processes: Optional[int] = None
    batch_size: int = 32
    queue_size: int = 1000
    enable_async: bool = True
    wavelength_parallelism: bool = True
    inference_parallelism: bool = True
    memory_limit_mb: int = 2048
    
    def __post_init__(self):
        if self.max_threads is None:
            self.max_threads = min(32, (psutil.cpu_count() or 1) * 2)
        if self.max_processes is None:
            self.max_processes = psutil.cpu_count() or 1


class WavelengthProcessor:
    """Optimized processor for parallel wavelength operations."""
    
    def __init__(self, wavelength_config: WavelengthConfig, concurrency_config: ConcurrencyConfig):
        self.wavelength_config = wavelength_config
        self.config = concurrency_config
        self.thread_pool = ThreadPoolExecutor(max_workers=self.config.max_threads)
        self.process_pool = ProcessPoolExecutor(max_workers=self.config.max_processes)
        
        # Pre-allocated arrays for efficiency
        self.temp_buffers = {}
        self._initialize_buffers()
        
        logger.info(f"Initialized WavelengthProcessor with {self.config.max_threads} threads")
    
    def _initialize_buffers(self):
        """Pre-allocate temporary buffers for each thread."""
        for i in range(self.config.max_threads):
            self.temp_buffers[i] = {
                'input_buffer': np.zeros((self.config.batch_size, 1024), dtype=complex),
                'output_buffer': np.zeros((self.config.batch_size, 1024), dtype=complex),
                'weight_buffer': np.zeros((1024, 1024), dtype=complex)
            }
    
    def process_wavelengths_parallel(self, 
                                   inputs: np.ndarray, 
                                   weights: np.ndarray,
                                   operation_func: Callable) -> np.ndarray:
        """Process multiple wavelengths in parallel."""
        batch_size, input_dim, num_wavelengths = inputs.shape
        output_dim = weights.shape[1]
        
        # Check cache first
        input_hash = compute_input_hash([inputs, weights])
        cached_result = inference_cache.get_cached_inference(input_hash)
        if cached_result is not None:
            return cached_result[0]
        
        # Prepare output array
        outputs = np.zeros((batch_size, output_dim, num_wavelengths), dtype=complex)
        
        # Create tasks for parallel execution
        futures = []
        
        if self.config.wavelength_parallelism and num_wavelengths > 1:
            # Submit wavelength processing tasks
            for w in range(num_wavelengths):
                future = self.thread_pool.submit(
                    self._process_single_wavelength,
                    inputs[:, :, w], weights[:, :, w], operation_func, w
                )
                futures.append((future, w))
            
            # Collect results
            for future, w in futures:
                outputs[:, :, w] = future.result()
        else:
            # Sequential processing
            for w in range(num_wavelengths):
                outputs[:, :, w] = self._process_single_wavelength(
                    inputs[:, :, w], weights[:, :, w], operation_func, w
                )
        
        # Cache result
        inference_cache.cache_inference(input_hash, outputs, {'wavelengths': num_wavelengths})
        
        return outputs
    
    def _process_single_wavelength(self, 
                                 inputs: np.ndarray, 
                                 weights: np.ndarray,
                                 operation_func: Callable,
                                 wavelength_idx: int) -> np.ndarray:
        """Process single wavelength channel."""
        thread_id = threading.get_ident() % self.config.max_threads
        buffers = self.temp_buffers.get(thread_id, {})
        
        # Use pre-allocated buffers if available
        if 'input_buffer' in buffers and inputs.shape[0] <= buffers['input_buffer'].shape[0]:
            input_buffer = buffers['input_buffer'][:inputs.shape[0], :inputs.shape[1]]
            input_buffer[:] = inputs
            inputs = input_buffer
        
        # Perform matrix operation
        result = operation_func(inputs, weights)
        
        return result
    
    async def process_wavelengths_async(self, 
                                      inputs: np.ndarray, 
                                      weights: np.ndarray,
                                      operation_func: Callable) -> np.ndarray:
        """Asynchronous wavelength processing."""
        batch_size, input_dim, num_wavelengths = inputs.shape
        output_dim = weights.shape[1]
        outputs = np.zeros((batch_size, output_dim, num_wavelengths), dtype=complex)
        
        # Create async tasks
        tasks = []
        loop = asyncio.get_event_loop()
        
        for w in range(num_wavelengths):
            task = loop.run_in_executor(
                self.thread_pool,
                self._process_single_wavelength,
                inputs[:, :, w], weights[:, :, w], operation_func, w
            )
            tasks.append((task, w))
        
        # Wait for all tasks to complete
        for task, w in tasks:
            outputs[:, :, w] = await task
        
        return outputs
    
    def shutdown(self):
        """Shutdown thread and process pools."""
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)


class InferencePipeline:
    """High-throughput inference pipeline with concurrency."""
    
    def __init__(self, models: List[PhotonicNeuralNetwork], config: ConcurrencyConfig):
        self.models = models
        self.config = config
        self.request_queue = asyncio.Queue(maxsize=config.queue_size)
        self.result_cache = {}
        self.active_tasks = set()
        
        # Worker pools
        self.thread_pool = ThreadPoolExecutor(max_workers=config.max_threads)
        self.wavelength_processors = [
            WavelengthProcessor(model.wavelength_config, config) 
            for model in models
        ]
        
        # Metrics
        self.processed_requests = 0
        self.total_latency = 0.0
        self.concurrent_requests = 0
        
        logger.info(f"Initialized InferencePipeline with {len(models)} models")
    
    async def submit_inference_request(self, inputs: np.ndarray, model_idx: int = 0) -> str:
        """Submit inference request asynchronously."""
        request_id = f"req_{int(time.time() * 1000000)}_{id(inputs)}"
        
        # Check if already processing
        input_hash = compute_input_hash(inputs)
        if input_hash in self.result_cache:
            return request_id  # Return immediately, result already cached
        
        # Add to queue
        await self.request_queue.put((request_id, inputs, model_idx, input_hash))
        
        return request_id
    
    async def process_inference_queue(self):
        """Process inference requests from queue."""
        while True:
            try:
                request_id, inputs, model_idx, input_hash = await self.request_queue.get()
                
                # Check cache again
                if input_hash in self.result_cache:
                    self.request_queue.task_done()
                    continue
                
                # Create processing task
                task = asyncio.create_task(
                    self._process_single_inference(request_id, inputs, model_idx, input_hash)
                )
                self.active_tasks.add(task)
                
                # Clean up completed tasks
                task.add_done_callback(self.active_tasks.discard)
                
                self.request_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing inference queue: {e}")
    
    async def _process_single_inference(self, 
                                      request_id: str, 
                                      inputs: np.ndarray, 
                                      model_idx: int,
                                      input_hash: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process single inference request."""
        start_time = time.perf_counter()
        self.concurrent_requests += 1
        
        try:
            model = self.models[model_idx]
            wavelength_processor = self.wavelength_processors[model_idx]
            
            # Perform inference using concurrent wavelength processing
            if self.config.enable_async:
                result, metrics = await self._async_forward_pass(
                    model, inputs, wavelength_processor
                )
            else:
                result, metrics = model.forward(inputs, measure_latency=True)
            
            # Cache result
            self.result_cache[input_hash] = (result, metrics)
            
            # Update metrics
            end_time = time.perf_counter()
            latency = (end_time - start_time) * 1000  # Convert to ms
            self.total_latency += latency
            self.processed_requests += 1
            
            return result, metrics
            
        except Exception as e:
            logger.error(f"Error processing inference {request_id}: {e}")
            raise
        finally:
            self.concurrent_requests -= 1
    
    async def _async_forward_pass(self, 
                                model: PhotonicNeuralNetwork, 
                                inputs: np.ndarray,
                                wavelength_processor: WavelengthProcessor) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Asynchronous forward pass through model."""
        current_input = inputs
        total_latency = 0.0
        layer_metrics = []
        
        # Process each layer asynchronously when possible
        for i, layer in enumerate(model.layers):
            start_time = time.perf_counter()
            
            # Use wavelength processor for parallel processing
            if hasattr(layer, 'processor') and self.config.wavelength_parallelism:
                # Ensure inputs are in correct format for wavelength processing
                if current_input.ndim == 2:
                    current_input = np.tile(current_input[:, :, np.newaxis], 
                                          (1, 1, layer.processor.wavelength_config.num_channels))
                
                # Process with parallel wavelength operations
                output = await wavelength_processor.process_wavelengths_async(
                    current_input, layer.weights, 
                    lambda x, w: layer.forward(x[:, :, 0] if x.ndim == 3 else x)
                )
            else:
                # Standard synchronous processing
                loop = asyncio.get_event_loop()
                output = await loop.run_in_executor(
                    self.thread_pool, layer.forward, current_input
                )
            
            end_time = time.perf_counter()
            latency = (end_time - start_time) * 1000
            total_latency += latency
            
            layer_metrics.append({
                "layer_id": i,
                "latency_ms": latency,
                "async_processed": True
            })
            
            current_input = output
        
        # Aggregate wavelength outputs
        final_output = model._aggregate_wavelength_outputs(current_input)
        
        metrics = {
            "total_latency_ms": total_latency,
            "layer_metrics": layer_metrics,
            "async_pipeline": True
        }
        
        return final_output, metrics
    
    async def get_inference_result(self, input_hash: str, timeout: float = 5.0) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """Get inference result by input hash."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if input_hash in self.result_cache:
                return self.result_cache[input_hash]
            await asyncio.sleep(0.01)  # Small delay
        
        return None
    
    def get_pipeline_metrics(self) -> Dict[str, Any]:
        """Get pipeline performance metrics."""
        avg_latency = (self.total_latency / self.processed_requests 
                      if self.processed_requests > 0 else 0.0)
        
        return {
            "processed_requests": self.processed_requests,
            "avg_latency_ms": avg_latency,
            "concurrent_requests": self.concurrent_requests,
            "queue_size": self.request_queue.qsize(),
            "cache_hits": len(self.result_cache),
            "active_tasks": len(self.active_tasks)
        }
    
    async def shutdown(self):
        """Gracefully shutdown the pipeline."""
        # Cancel active tasks
        for task in self.active_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.active_tasks:
            await asyncio.gather(*self.active_tasks, return_exceptions=True)
        
        # Shutdown worker pools
        self.thread_pool.shutdown(wait=True)
        for processor in self.wavelength_processors:
            processor.shutdown()


class ResourcePool:
    """Intelligent resource pooling for photonic computations."""
    
    def __init__(self, config: ConcurrencyConfig):
        self.config = config
        self.available_gpus = self._detect_gpus()
        self.memory_pools = {}
        self.computation_pools = {}
        
        # Initialize memory pools
        self._initialize_memory_pools()
        
        logger.info(f"Initialized ResourcePool with {len(self.available_gpus)} GPUs")
    
    def _detect_gpus(self) -> List[int]:
        """Detect available GPU devices."""
        try:
            import cupy
            return list(range(cupy.cuda.runtime.getDeviceCount()))
        except:
            return []  # No CUDA support
    
    def _initialize_memory_pools(self):
        """Initialize memory pools for efficient allocation."""
        # CPU memory pool
        self.memory_pools['cpu'] = {
            'small_arrays': queue.Queue(),  # For small temporary arrays
            'large_arrays': queue.Queue(),  # For large matrices
            'complex_arrays': queue.Queue() # For complex number arrays
        }
        
        # GPU memory pools (if available)
        for gpu_id in self.available_gpus:
            self.memory_pools[f'gpu_{gpu_id}'] = {
                'device_arrays': queue.Queue(),
                'pinned_memory': queue.Queue()
            }
    
    @asynccontextmanager
    async def acquire_resources(self, resource_type: str, size_hint: int = 0):
        """Acquire computational resources."""
        resource = None
        try:
            if resource_type == 'cpu_array':
                resource = self._get_cpu_array(size_hint)
            elif resource_type.startswith('gpu_'):
                gpu_id = int(resource_type.split('_')[1])
                resource = self._get_gpu_array(gpu_id, size_hint)
            
            yield resource
            
        finally:
            if resource is not None:
                self._return_resource(resource_type, resource)
    
    def _get_cpu_array(self, size_hint: int) -> np.ndarray:
        """Get pre-allocated CPU array from pool."""
        pool_key = 'small_arrays' if size_hint < 1024*1024 else 'large_arrays'
        
        try:
            return self.memory_pools['cpu'][pool_key].get_nowait()
        except queue.Empty:
            # Create new array if pool is empty
            if size_hint > 0:
                return np.empty(size_hint, dtype=complex)
            else:
                return np.empty(1024*1024, dtype=complex)  # Default size
    
    def _get_gpu_array(self, gpu_id: int, size_hint: int):
        """Get pre-allocated GPU array from pool."""
        try:
            import cupy
            with cupy.cuda.Device(gpu_id):
                pool = self.memory_pools[f'gpu_{gpu_id}']['device_arrays']
                try:
                    return pool.get_nowait()
                except queue.Empty:
                    if size_hint > 0:
                        return cupy.empty(size_hint, dtype=complex)
                    else:
                        return cupy.empty(1024*1024, dtype=complex)
        except ImportError:
            return None
    
    def _return_resource(self, resource_type: str, resource: Any):
        """Return resource to pool for reuse."""
        try:
            if resource_type == 'cpu_array':
                if hasattr(resource, 'size'):
                    pool_key = 'small_arrays' if resource.size < 1024*1024 else 'large_arrays'
                    # Clear array and return to pool
                    resource.fill(0)
                    self.memory_pools['cpu'][pool_key].put(resource)
            
            elif resource_type.startswith('gpu_'):
                gpu_id = int(resource_type.split('_')[1])
                # Clear GPU array and return to pool
                if hasattr(resource, 'fill'):
                    resource.fill(0)
                    self.memory_pools[f'gpu_{gpu_id}']['device_arrays'].put(resource)
        
        except Exception as e:
            logger.warning(f"Failed to return resource to pool: {e}")


# Global instances
default_config = ConcurrencyConfig()
resource_pool = ResourcePool(default_config)