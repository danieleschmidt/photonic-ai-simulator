"""
Advanced Caching and Memory Optimization for Photonic AI

Implements intelligent caching, memory pooling, and optimization strategies
for high-performance photonic neural network deployments.
"""

import time
import threading
import hashlib
import pickle
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from enum import Enum
import logging
from collections import OrderedDict, defaultdict
import numpy as np
import gc
import psutil

logger = logging.getLogger(__name__)


class CachePolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"           # Least Recently Used
    LFU = "lfu"           # Least Frequently Used
    ADAPTIVE = "adaptive" # Adaptive based on access patterns


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    data: Any
    size_bytes: int
    created_at: float
    last_accessed: float
    access_count: int = 0
    ttl: Optional[float] = None
    
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    def update_access(self):
        """Update access metadata."""
        self.last_accessed = time.time()
        self.access_count += 1


@dataclass
class CacheConfig:
    """Configuration for intelligent cache."""
    max_size_bytes: int = 1024 * 1024 * 1024  # 1GB default
    max_entries: int = 10000
    policy: CachePolicy = CachePolicy.ADAPTIVE
    default_ttl: Optional[float] = None
    enable_compression: bool = True
    memory_threshold: float = 0.85  # 85% memory usage threshold


class IntelligentCache:
    """
    Intelligent multi-tier cache for photonic AI systems.
    
    Provides adaptive caching with automatic memory management
    and compression capabilities.
    """
    
    def __init__(self, config: CacheConfig = None):
        """Initialize intelligent cache."""
        self.config = config or CacheConfig()
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order = OrderedDict()  # For LRU
        self.frequency_counter = defaultdict(int)  # For LFU
        self.size_tracker = 0
        self._lock = threading.RLock()
        
        # Cache statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "compressions": 0,
            "decompressions": 0
        }
        
        # Adaptive policy state
        self.current_policy = self.config.policy
        
        logger.info(f"Initialized intelligent cache with {self.config.max_size_bytes // (1024*1024)}MB capacity")
    
    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve item from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        with self._lock:
            if key not in self.cache:
                self.stats["misses"] += 1
                return None
            
            entry = self.cache[key]
            
            # Check expiration
            if entry.is_expired():
                self._remove_entry(key)
                self.stats["misses"] += 1
                return None
            
            # Update access metadata
            entry.update_access()
            self._update_access_tracking(key)
            self.stats["hits"] += 1
            
            # Decompress if needed
            data = entry.data
            if self.config.enable_compression and isinstance(data, bytes):
                try:
                    data = self._decompress_data(data)
                    self.stats["decompressions"] += 1
                except Exception as e:
                    logger.error(f"Decompression failed for key {key}: {e}")
                    return None
            
            return data
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """
        Store item in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            
        Returns:
            True if successfully cached
        """
        try:
            with self._lock:
                # Serialize and compress data
                processed_data = value
                if self.config.enable_compression:
                    processed_data = self._compress_data(value)
                    self.stats["compressions"] += 1
                
                # Calculate size
                size_bytes = self._calculate_size(processed_data)
                
                # Check if we need to make room
                if not self._ensure_capacity(size_bytes):
                    logger.warning(f"Could not make room for cache entry {key}")
                    return False
                
                # Remove existing entry if present
                if key in self.cache:
                    self._remove_entry(key)
                
                # Create cache entry
                entry = CacheEntry(
                    key=key,
                    data=processed_data,
                    size_bytes=size_bytes,
                    created_at=time.time(),
                    last_accessed=time.time(),
                    ttl=ttl or self.config.default_ttl
                )
                
                # Store entry
                self.cache[key] = entry
                self.size_tracker += size_bytes
                self._update_access_tracking(key)
                
                logger.debug(f"Cached {key} ({size_bytes} bytes)")
                return True
                
        except Exception as e:
            logger.error(f"Failed to cache {key}: {e}")
            return False
    
    def invalidate(self, key: str) -> bool:
        """Remove item from cache."""
        with self._lock:
            if key in self.cache:
                self._remove_entry(key)
                return True
            return False
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self.cache.clear()
            self.access_order.clear()
            self.frequency_counter.clear()
            self.size_tracker = 0
            logger.info("Cache cleared")
    
    def _ensure_capacity(self, required_bytes: int) -> bool:
        """Ensure cache has capacity for new entry."""
        # Check memory pressure
        if self._is_memory_pressure():
            self._emergency_eviction()
        
        # Check size limits
        while (self.size_tracker + required_bytes > self.config.max_size_bytes or
               len(self.cache) >= self.config.max_entries):
            
            if not self.cache:
                return False
            
            # Evict based on current policy
            evicted = self._evict_entry()
            if not evicted:
                return False
        
        return True
    
    def _evict_entry(self) -> bool:
        """Evict entry based on current policy."""
        if not self.cache:
            return False
        
        eviction_key = None
        
        if self.current_policy == CachePolicy.LRU:
            eviction_key = self._lru_eviction()
        elif self.current_policy == CachePolicy.LFU:
            eviction_key = self._lfu_eviction()
        elif self.current_policy == CachePolicy.ADAPTIVE:
            eviction_key = self._adaptive_eviction()
        
        if eviction_key:
            self._remove_entry(eviction_key)
            self.stats["evictions"] += 1
            return True
        
        return False
    
    def _lru_eviction(self) -> Optional[str]:
        """Least Recently Used eviction."""
        if self.access_order:
            return next(iter(self.access_order))
        return None
    
    def _lfu_eviction(self) -> Optional[str]:
        """Least Frequently Used eviction."""
        if not self.cache:
            return None
        
        # Find entry with lowest access count
        min_access_count = min(entry.access_count for entry in self.cache.values())
        for key, entry in self.cache.items():
            if entry.access_count == min_access_count:
                return key
        return None
    
    def _adaptive_eviction(self) -> Optional[str]:
        """Adaptive eviction based on access patterns."""
        # Prefer expired entries
        for key, entry in self.cache.items():
            if entry.is_expired():
                return key
        
        # If memory pressure is high, use LRU
        if self._is_memory_pressure():
            return self._lru_eviction()
        
        # Otherwise, use LFU for better hit rate
        return self._lfu_eviction()
    
    def _remove_entry(self, key: str):
        """Remove entry and update tracking."""
        if key in self.cache:
            entry = self.cache[key]
            self.size_tracker -= entry.size_bytes
            del self.cache[key]
            
            # Update tracking structures
            if key in self.access_order:
                del self.access_order[key]
            if key in self.frequency_counter:
                del self.frequency_counter[key]
    
    def _update_access_tracking(self, key: str):
        """Update access tracking structures."""
        # Update LRU tracking
        if key in self.access_order:
            del self.access_order[key]
        self.access_order[key] = None
        
        # Update LFU tracking
        self.frequency_counter[key] += 1
    
    def _compress_data(self, data: Any) -> bytes:
        """Compress data for storage."""
        import zlib
        serialized = pickle.dumps(data)
        return zlib.compress(serialized)
    
    def _decompress_data(self, compressed_data: bytes) -> Any:
        """Decompress data for retrieval."""
        import zlib
        decompressed = zlib.decompress(compressed_data)
        return pickle.loads(decompressed)
    
    def _calculate_size(self, data: Any) -> int:
        """Calculate size of data in bytes."""
        if isinstance(data, bytes):
            return len(data)
        elif isinstance(data, str):
            return len(data.encode('utf-8'))
        elif isinstance(data, np.ndarray):
            return data.nbytes
        else:
            # Estimate using pickle
            return len(pickle.dumps(data))
    
    def _is_memory_pressure(self) -> bool:
        """Check if system is under memory pressure."""
        memory_percent = psutil.virtual_memory().percent / 100.0
        return memory_percent > self.config.memory_threshold
    
    def _emergency_eviction(self):
        """Perform emergency eviction to free memory."""
        logger.warning("Memory pressure detected, performing emergency eviction")
        
        # Evict 20% of cache entries
        eviction_count = max(1, len(self.cache) // 5)
        
        for _ in range(eviction_count):
            if not self._evict_entry():
                break
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            hit_rate = 0.0
            total_requests = self.stats["hits"] + self.stats["misses"]
            if total_requests > 0:
                hit_rate = self.stats["hits"] / total_requests
            
            return {
                "entries": len(self.cache),
                "size_bytes": self.size_tracker,
                "size_mb": self.size_tracker / (1024 * 1024),
                "capacity_mb": self.config.max_size_bytes / (1024 * 1024),
                "utilization_percent": (self.size_tracker / self.config.max_size_bytes) * 100,
                "hit_rate": hit_rate,
                "policy": self.current_policy.value,
                "stats": dict(self.stats)
            }


class MemoryOptimizer:
    """Memory usage optimizer for photonic AI systems."""
    
    def __init__(self):
        """Initialize memory optimizer."""
        self.peak_usage = 0
        self.gc_threshold = 0.85  # Trigger GC at 85% memory usage
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        memory = psutil.virtual_memory()
        process = psutil.Process()
        process_memory = process.memory_info()
        
        usage = {
            "system_total_gb": memory.total / (1024**3),
            "system_used_gb": memory.used / (1024**3),
            "system_available_gb": memory.available / (1024**3),
            "system_percent": memory.percent,
            "process_rss_mb": process_memory.rss / (1024**2),
            "process_vms_mb": process_memory.vms / (1024**2),
            "peak_usage_mb": self.peak_usage / (1024**2)
        }
        
        # Update peak usage
        if process_memory.rss > self.peak_usage:
            self.peak_usage = process_memory.rss
        
        return usage
    
    def should_trigger_gc(self) -> bool:
        """Check if garbage collection should be triggered."""
        memory_percent = psutil.virtual_memory().percent / 100.0
        return memory_percent > self.gc_threshold
    
    def optimize_memory(self):
        """Perform memory optimization."""
        if self.should_trigger_gc():
            logger.info("Triggering garbage collection due to high memory usage")
            gc.collect()


# Cache decorators for easy usage
def cached(ttl: Optional[float] = None, key_func: Optional[Callable] = None):
    """
    Decorator to cache function results.
    
    Args:
        ttl: Time to live for cached results
        key_func: Function to generate cache key from arguments
    """
    def decorator(func: Callable) -> Callable:
        cache = IntelligentCache()
        
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key generation
                key_parts = [func.__name__]
                for arg in args:
                    if isinstance(arg, np.ndarray):
                        key_parts.append(hashlib.md5(arg.tobytes()).hexdigest()[:8])
                    else:
                        key_parts.append(str(hash(arg))[:8])
                for k, v in sorted(kwargs.items()):
                    key_parts.append(f"{k}={hash(v)}")
                cache_key = "_".join(key_parts)
            
            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            # Compute result and cache it
            result = func(*args, **kwargs)
            cache.put(cache_key, result, ttl=ttl)
            
            return result
        
        return wrapper
    return decorator


# Factory functions
def create_intelligent_cache(size_mb: int = 1024, 
                           policy: CachePolicy = CachePolicy.ADAPTIVE,
                           enable_compression: bool = True) -> IntelligentCache:
    """Create an intelligent cache with specified configuration."""
    config = CacheConfig(
        max_size_bytes=size_mb * 1024 * 1024,
        policy=policy,
        enable_compression=enable_compression
    )
    return IntelligentCache(config)


def create_memory_optimizer() -> MemoryOptimizer:
    """Create a memory optimizer."""
    return MemoryOptimizer()