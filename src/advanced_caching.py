"""
Advanced Caching System for Photonic AI Optimization.

Implements intelligent caching strategies for matrix operations, model weights,
inference results, and computation graphs to minimize redundant calculations.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import hashlib
import pickle
import time
from dataclasses import dataclass, field
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
import threading
import logging
from pathlib import Path
import sqlite3
import json
from enum import Enum

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Cache replacement strategies."""
    LRU = "least_recently_used"
    LFU = "least_frequently_used"
    TTL = "time_to_live"
    ADAPTIVE = "adaptive"


@dataclass
class CacheMetrics:
    """Cache performance metrics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    memory_usage_bytes: int = 0
    avg_access_time_ns: float = 0.0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


@dataclass
class CacheEntry:
    """Individual cache entry with metadata."""
    key: str
    value: Any
    timestamp: float
    access_count: int = 0
    last_access: float = 0.0
    size_bytes: int = 0
    computation_cost: float = 0.0  # Cost to recompute
    
    def __post_init__(self):
        self.last_access = self.timestamp
        if hasattr(self.value, 'nbytes'):
            self.size_bytes = self.value.nbytes
        else:
            try:
                self.size_bytes = len(pickle.dumps(self.value))
            except:
                self.size_bytes = 1024  # Estimate


class PhotonicMatrixCache:
    """Optimized cache for photonic matrix operations."""
    
    def __init__(self, max_size_mb: int = 512, strategy: CacheStrategy = CacheStrategy.ADAPTIVE):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.strategy = strategy
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.metrics = CacheMetrics()
        self.lock = threading.RLock()
        
        # Adaptive strategy parameters
        self.frequency_weights: Dict[str, float] = defaultdict(float)
        self.access_pattern_history: List[str] = []
        self.history_size = 1000
        
        logger.info(f"Initialized PhotonicMatrixCache with {max_size_mb}MB capacity")
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve cached value."""
        start_time = time.perf_counter_ns()
        
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                entry.access_count += 1
                entry.last_access = time.time()
                
                # Move to end for LRU
                if self.strategy == CacheStrategy.LRU:
                    self.cache.move_to_end(key)
                
                # Update frequency weights for adaptive strategy
                if self.strategy == CacheStrategy.ADAPTIVE:
                    self.frequency_weights[key] += 1.0
                    self._update_access_pattern(key)
                
                self.metrics.hits += 1
                end_time = time.perf_counter_ns()
                self._update_access_time(end_time - start_time)
                
                return entry.value
            else:
                self.metrics.misses += 1
                return None
    
    def put(self, key: str, value: Any, computation_cost: float = 1.0) -> bool:
        """Store value in cache."""
        with self.lock:
            current_time = time.time()
            entry = CacheEntry(
                key=key,
                value=value,
                timestamp=current_time,
                computation_cost=computation_cost
            )
            
            # Check if we need to evict entries
            while self._needs_eviction(entry.size_bytes):
                if not self._evict_entry():
                    logger.warning("Cache eviction failed, rejecting new entry")
                    return False
            
            # Store entry
            self.cache[key] = entry
            self.metrics.memory_usage_bytes += entry.size_bytes
            
            # Initialize frequency weight for adaptive strategy
            if self.strategy == CacheStrategy.ADAPTIVE:
                self.frequency_weights[key] = 1.0
                self._update_access_pattern(key)
            
            return True
    
    def _needs_eviction(self, new_entry_size: int) -> bool:
        """Check if eviction is needed for new entry."""
        return self.metrics.memory_usage_bytes + new_entry_size > self.max_size_bytes
    
    def _evict_entry(self) -> bool:
        """Evict entry based on strategy."""
        if not self.cache:
            return False
        
        victim_key = self._select_victim()
        if victim_key:
            victim_entry = self.cache[victim_key]
            del self.cache[victim_key]
            self.metrics.memory_usage_bytes -= victim_entry.size_bytes
            self.metrics.evictions += 1
            
            # Clean up frequency weights
            if victim_key in self.frequency_weights:
                del self.frequency_weights[victim_key]
            
            logger.debug(f"Evicted cache entry: {victim_key}")
            return True
        
        return False
    
    def _select_victim(self) -> Optional[str]:
        """Select victim for eviction based on strategy."""
        if not self.cache:
            return None
        
        current_time = time.time()
        
        if self.strategy == CacheStrategy.LRU:
            # First item in OrderedDict is least recently used
            return next(iter(self.cache))
        
        elif self.strategy == CacheStrategy.LFU:
            # Find entry with lowest access count
            return min(self.cache.keys(), 
                      key=lambda k: self.cache[k].access_count)
        
        elif self.strategy == CacheStrategy.TTL:
            # Find oldest entry (could be extended with actual TTL)
            return min(self.cache.keys(), 
                      key=lambda k: self.cache[k].timestamp)
        
        elif self.strategy == CacheStrategy.ADAPTIVE:
            # Complex scoring based on frequency, recency, and computation cost
            scores = {}
            for key, entry in self.cache.items():
                frequency_score = self.frequency_weights.get(key, 0.0)
                recency_score = 1.0 / max(current_time - entry.last_access, 1.0)
                cost_score = entry.computation_cost
                
                # Combined score (higher is better, so we want lowest)
                scores[key] = frequency_score * recency_score * cost_score
            
            return min(scores.keys(), key=lambda k: scores[k])
        
        return next(iter(self.cache))  # Fallback to first entry
    
    def _update_access_pattern(self, key: str):
        """Update access pattern history for adaptive strategy."""
        self.access_pattern_history.append(key)
        if len(self.access_pattern_history) > self.history_size:
            self.access_pattern_history.pop(0)
    
    def _update_access_time(self, access_time_ns: int):
        """Update average access time metrics."""
        total_accesses = self.metrics.hits + self.metrics.misses
        if total_accesses > 0:
            self.metrics.avg_access_time_ns = (
                (self.metrics.avg_access_time_ns * (total_accesses - 1) + access_time_ns) /
                total_accesses
            )
    
    def clear(self):
        """Clear all cached entries."""
        with self.lock:
            self.cache.clear()
            self.frequency_weights.clear()
            self.access_pattern_history.clear()
            self.metrics = CacheMetrics()
    
    def get_metrics(self) -> CacheMetrics:
        """Get current cache metrics."""
        return self.metrics


class InferenceCache:
    """Specialized cache for inference results."""
    
    def __init__(self, max_entries: int = 10000, enable_persistence: bool = True):
        self.max_entries = max_entries
        self.enable_persistence = enable_persistence
        self.cache: Dict[str, Tuple[np.ndarray, Dict[str, Any]]] = {}
        self.access_times: Dict[str, float] = {}
        self.lock = threading.RLock()
        
        # Persistence
        if enable_persistence:
            self.db_path = Path("inference_cache.db")
            self._init_persistence()
        
        logger.info(f"Initialized InferenceCache with {max_entries} max entries")
    
    def _init_persistence(self):
        """Initialize persistent storage."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS inference_cache (
                        key TEXT PRIMARY KEY,
                        result BLOB,
                        metadata TEXT,
                        timestamp REAL
                    )
                ''')
        except Exception as e:
            logger.warning(f"Failed to initialize persistence: {e}")
            self.enable_persistence = False
    
    def get_cached_inference(self, input_hash: str) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """Get cached inference result."""
        with self.lock:
            # Check memory cache first
            if input_hash in self.cache:
                self.access_times[input_hash] = time.time()
                return self.cache[input_hash]
            
            # Check persistent cache
            if self.enable_persistence:
                try:
                    with sqlite3.connect(self.db_path) as conn:
                        cursor = conn.execute(
                            'SELECT result, metadata FROM inference_cache WHERE key = ?',
                            (input_hash,)
                        )
                        row = cursor.fetchone()
                        if row:
                            result = pickle.loads(row[0])
                            metadata = json.loads(row[1])
                            
                            # Load into memory cache
                            self.cache[input_hash] = (result, metadata)
                            self.access_times[input_hash] = time.time()
                            
                            return (result, metadata)
                except Exception as e:
                    logger.warning(f"Failed to load from persistent cache: {e}")
            
            return None
    
    def cache_inference(self, input_hash: str, result: np.ndarray, metadata: Dict[str, Any]):
        """Cache inference result."""
        with self.lock:
            current_time = time.time()
            
            # Store in memory
            self.cache[input_hash] = (result, metadata)
            self.access_times[input_hash] = current_time
            
            # Evict if necessary
            if len(self.cache) > self.max_entries:
                self._evict_lru()
            
            # Store persistently
            if self.enable_persistence:
                try:
                    with sqlite3.connect(self.db_path) as conn:
                        conn.execute('''
                            INSERT OR REPLACE INTO inference_cache 
                            (key, result, metadata, timestamp) VALUES (?, ?, ?, ?)
                        ''', (
                            input_hash,
                            pickle.dumps(result),
                            json.dumps(metadata),
                            current_time
                        ))
                except Exception as e:
                    logger.warning(f"Failed to persist cache entry: {e}")
    
    def _evict_lru(self):
        """Evict least recently used entry."""
        if self.access_times:
            lru_key = min(self.access_times.keys(), 
                         key=lambda k: self.access_times[k])
            del self.cache[lru_key]
            del self.access_times[lru_key]


class ComputationGraphCache:
    """Cache for computation graphs and partial results."""
    
    def __init__(self):
        self.partial_results: Dict[str, Any] = {}
        self.graph_cache: Dict[str, Any] = {}
        self.lock = threading.RLock()
        
    def cache_partial_result(self, operation_id: str, inputs_hash: str, result: Any):
        """Cache partial computation result."""
        with self.lock:
            key = f"{operation_id}:{inputs_hash}"
            self.partial_results[key] = result
    
    def get_partial_result(self, operation_id: str, inputs_hash: str) -> Optional[Any]:
        """Get cached partial result."""
        with self.lock:
            key = f"{operation_id}:{inputs_hash}"
            return self.partial_results.get(key)
    
    def invalidate_operation(self, operation_id: str):
        """Invalidate all cached results for an operation."""
        with self.lock:
            keys_to_remove = [k for k in self.partial_results.keys() 
                             if k.startswith(f"{operation_id}:")]
            for key in keys_to_remove:
                del self.partial_results[key]


def compute_input_hash(inputs: Union[np.ndarray, List[np.ndarray]], 
                      model_params: Optional[Dict[str, Any]] = None) -> str:
    """Compute hash for caching input/output pairs."""
    hasher = hashlib.sha256()
    
    if isinstance(inputs, list):
        for inp in inputs:
            hasher.update(inp.tobytes())
    else:
        hasher.update(inputs.tobytes())
    
    if model_params:
        param_str = json.dumps(model_params, sort_keys=True)
        hasher.update(param_str.encode())
    
    return hasher.hexdigest()


# Global cache instances
matrix_cache = PhotonicMatrixCache()
inference_cache = InferenceCache()
graph_cache = ComputationGraphCache()