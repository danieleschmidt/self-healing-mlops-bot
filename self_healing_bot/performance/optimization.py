"""Performance optimization and auto-scaling for the self-healing bot."""

import asyncio
import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import threading
from functools import wraps
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics tracking."""
    execution_time: float
    memory_usage: float
    cpu_usage: float
    concurrent_operations: int
    cache_hit_rate: float
    throughput: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class AdvancedCache:
    """High-performance cache with TTL and LRU eviction."""
    
    def __init__(self, max_size: int = 10000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_times: Dict[str, float] = {}
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            
            entry = self._cache[key]
            
            # Check TTL
            if time.time() > entry['expires_at']:
                del self._cache[key]
                del self._access_times[key]
                self._misses += 1
                return None
            
            # Update access time for LRU
            self._access_times[key] = time.time()
            self._hits += 1
            
            return entry['value']
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        ttl = ttl or self.default_ttl
        expires_at = time.time() + ttl
        
        with self._lock:
            # Remove oldest entries if at capacity
            if len(self._cache) >= self.max_size:
                self._evict_lru()
            
            self._cache[key] = {
                'value': value,
                'expires_at': expires_at,
                'created_at': time.time()
            }
            self._access_times[key] = time.time()
    
    def _evict_lru(self) -> None:
        """Evict least recently used entries."""
        if not self._access_times:
            return
        
        # Remove 10% of entries (oldest access times)
        evict_count = max(1, len(self._access_times) // 10)
        oldest_keys = sorted(self._access_times.keys(), 
                           key=lambda k: self._access_times[k])[:evict_count]
        
        for key in oldest_keys:
            if key in self._cache:
                del self._cache[key]
            if key in self._access_times:
                del self._access_times[key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests) if total_requests > 0 else 0
            
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': hit_rate,
                'utilization': len(self._cache) / self.max_size
            }


class ConcurrencyManager:
    """Advanced concurrency management with rate limiting."""
    
    def __init__(self, max_concurrent: int = 100, max_workers: int = 10):
        self.max_concurrent = max_concurrent
        self.max_workers = max_workers
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._metrics = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'average_execution_time': 0.0
        }
        self._lock = threading.Lock()
    
    async def execute_async(self, coro: Callable, task_id: str = None) -> Any:
        """Execute async function with concurrency control."""
        async with self._semaphore:
            try:
                start_time = time.time()
                self._metrics['total_tasks'] += 1
                
                # Execute the coroutine
                if asyncio.iscoroutine(coro):
                    result = await coro
                else:
                    result = await coro()
                
                # Update metrics
                execution_time = time.time() - start_time
                self._update_metrics(execution_time, success=True)
                
                return result
                
            except Exception as e:
                self._update_metrics(0, success=False)
                logger.error("Task execution failed", task_id=task_id, error=str(e))
                raise
    
    def _update_metrics(self, execution_time: float, success: bool) -> None:
        """Update execution metrics."""
        with self._lock:
            if success:
                self._metrics['completed_tasks'] += 1
                # Update rolling average
                total_completed = self._metrics['completed_tasks']
                current_avg = self._metrics['average_execution_time']
                self._metrics['average_execution_time'] = (
                    (current_avg * (total_completed - 1) + execution_time) / total_completed
                )
            else:
                self._metrics['failed_tasks'] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get concurrency metrics."""
        with self._lock:
            return {
                **self._metrics,
                'available_permits': self._semaphore._value,
            }


class PerformanceOptimizer:
    """Main performance optimization controller."""
    
    def __init__(self):
        self.cache = AdvancedCache()
        self.concurrency_manager = ConcurrencyManager()
        self._optimization_enabled = True
    
    def cached(self, ttl: int = 3600, key_func: Callable = None):
        """Decorator for caching function results."""
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                if not self._optimization_enabled:
                    return await func(*args, **kwargs)
                
                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    cache_key = f"{func.__name__}:{hash((args, tuple(sorted(kwargs.items()))))}"
                
                # Try cache first
                cached_result = self.cache.get(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # Execute function and cache result
                result = await func(*args, **kwargs)
                self.cache.set(cache_key, result, ttl)
                return result
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                if not self._optimization_enabled:
                    return func(*args, **kwargs)
                
                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    cache_key = f"{func.__name__}:{hash((args, tuple(sorted(kwargs.items()))))}"
                
                # Try cache first
                cached_result = self.cache.get(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # Execute function and cache result
                result = func(*args, **kwargs)
                self.cache.set(cache_key, result, ttl)
                return result
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
        cache_stats = self.cache.get_stats()
        concurrency_stats = self.concurrency_manager.get_metrics()
        
        return {
            'optimization_enabled': self._optimization_enabled,
            'cache': cache_stats,
            'concurrency': concurrency_stats,
        }


# Global performance optimizer instance
performance_optimizer = PerformanceOptimizer()