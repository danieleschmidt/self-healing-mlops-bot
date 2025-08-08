"""Enhanced caching system with adaptive TTL and intelligent invalidation."""

import asyncio
import hashlib
import json
import logging
import time
from typing import Any, Dict, Optional, Callable, Union, List
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Cache strategy types."""
    LRU = "lru"              # Least Recently Used
    LFU = "lfu"              # Least Frequently Used
    TTL = "ttl"              # Time To Live
    ADAPTIVE = "adaptive"     # Adaptive TTL based on access patterns


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    value: Any
    created_at: float
    last_accessed: float
    access_count: int = 0
    hit_rate: float = 0.0
    ttl: float = 300.0  # Default 5 minutes
    
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        return time.time() - self.created_at > self.ttl
    
    def access(self):
        """Record an access to this entry."""
        current_time = time.time()
        self.last_accessed = current_time
        self.access_count += 1
        
        # Update hit rate (simple moving average)
        time_since_creation = current_time - self.created_at
        if time_since_creation > 0:
            self.hit_rate = self.access_count / time_since_creation * 60  # hits per minute


@dataclass 
class CacheStats:
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0
    max_size: int = 1000
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    @property
    def memory_pressure(self) -> float:
        """Calculate memory pressure (size / max_size)."""
        return self.size / self.max_size if self.max_size > 0 else 0.0


class AdaptiveCache:
    """High-performance cache with adaptive TTL and intelligent eviction."""
    
    def __init__(
        self, 
        max_size: int = 1000,
        default_ttl: float = 300.0,
        strategy: CacheStrategy = CacheStrategy.ADAPTIVE,
        cleanup_interval: float = 60.0
    ):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.strategy = strategy
        self.cleanup_interval = cleanup_interval
        
        # Storage
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: List[str] = []  # For LRU
        self._frequency: Dict[str, int] = {}  # For LFU
        
        # Statistics
        self.stats = CacheStats(max_size=max_size)
        
        # Background cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
    
    def _generate_key(self, key: Union[str, tuple, dict]) -> str:
        """Generate a consistent cache key."""
        if isinstance(key, str):
            return key
        elif isinstance(key, (tuple, list)):
            return hashlib.md5(str(key).encode()).hexdigest()
        elif isinstance(key, dict):
            # Sort dict keys for consistent hashing
            sorted_items = sorted(key.items())
            return hashlib.md5(str(sorted_items).encode()).hexdigest()
        else:
            return hashlib.md5(str(key).encode()).hexdigest()
    
    async def get(self, key: Union[str, tuple, dict]) -> Optional[Any]:
        """Get value from cache."""
        cache_key = self._generate_key(key)
        
        async with self._lock:
            if cache_key not in self._cache:
                self.stats.misses += 1
                return None
            
            entry = self._cache[cache_key]
            
            # Check if expired
            if entry.is_expired():
                await self._evict_key(cache_key)
                self.stats.misses += 1
                return None
            
            # Record access
            entry.access()
            self._record_access(cache_key)
            self.stats.hits += 1
            
            logger.debug(f"Cache hit for key: {cache_key[:16]}...")
            return entry.value
    
    async def set(
        self, 
        key: Union[str, tuple, dict], 
        value: Any, 
        ttl: Optional[float] = None
    ) -> None:
        """Set value in cache."""
        cache_key = self._generate_key(key)
        
        async with self._lock:
            # Calculate adaptive TTL if needed
            effective_ttl = ttl or self._calculate_adaptive_ttl(cache_key, value)
            
            # Check if we need to evict
            if len(self._cache) >= self.max_size and cache_key not in self._cache:
                await self._evict_least_valuable()
            
            # Create or update entry
            entry = CacheEntry(
                value=value,
                created_at=time.time(),
                last_accessed=time.time(),
                ttl=effective_ttl
            )
            
            self._cache[cache_key] = entry
            self._record_access(cache_key)
            
            if cache_key not in [k for k in self._cache.keys()]:
                self.stats.size += 1
            
            logger.debug(f"Cache set for key: {cache_key[:16]}... (TTL: {effective_ttl}s)")
    
    async def invalidate(self, key: Union[str, tuple, dict]) -> bool:
        """Invalidate a cache entry."""
        cache_key = self._generate_key(key)
        
        async with self._lock:
            if cache_key in self._cache:
                await self._evict_key(cache_key)
                return True
            return False
    
    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all keys matching a pattern."""
        async with self._lock:
            keys_to_remove = []
            
            for cache_key in self._cache.keys():
                if pattern in cache_key:
                    keys_to_remove.append(cache_key)
            
            for cache_key in keys_to_remove:
                await self._evict_key(cache_key)
            
            return len(keys_to_remove)
    
    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._frequency.clear()
            self.stats.size = 0
            logger.info("Cache cleared")
    
    def _calculate_adaptive_ttl(self, key: str, value: Any) -> float:
        """Calculate adaptive TTL based on access patterns and value characteristics."""
        if self.strategy != CacheStrategy.ADAPTIVE:
            return self.default_ttl
        
        # Base TTL
        base_ttl = self.default_ttl
        
        # Adjust based on existing entry history
        if key in self._cache:
            existing_entry = self._cache[key]
            
            # If frequently accessed, increase TTL
            if existing_entry.hit_rate > 1.0:  # More than 1 hit per minute
                base_ttl *= 2.0
            elif existing_entry.hit_rate < 0.1:  # Less than 1 hit per 10 minutes
                base_ttl *= 0.5
        
        # Adjust based on value size/complexity
        try:
            value_size = len(str(value))
            if value_size > 10000:  # Large values get longer TTL
                base_ttl *= 1.5
            elif value_size < 100:  # Small values get shorter TTL
                base_ttl *= 0.8
        except:
            pass  # Ignore size calculation errors
        
        # Adjust based on memory pressure
        if self.stats.memory_pressure > 0.8:
            base_ttl *= 0.7  # Reduce TTL under high memory pressure
        elif self.stats.memory_pressure < 0.5:
            base_ttl *= 1.2  # Increase TTL under low memory pressure
        
        return max(30.0, min(base_ttl, 3600.0))  # Clamp between 30s and 1 hour
    
    def _record_access(self, key: str) -> None:
        """Record access for eviction policies."""
        # Update LRU order
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
        
        # Update LFU frequency
        self._frequency[key] = self._frequency.get(key, 0) + 1
    
    async def _evict_least_valuable(self) -> None:
        """Evict the least valuable entry based on strategy."""
        if not self._cache:
            return
        
        key_to_evict = None
        
        if self.strategy == CacheStrategy.LRU:
            # Remove least recently used
            key_to_evict = self._access_order[0] if self._access_order else list(self._cache.keys())[0]
            
        elif self.strategy == CacheStrategy.LFU:
            # Remove least frequently used
            if self._frequency:
                key_to_evict = min(self._frequency.keys(), key=lambda k: self._frequency[k])
            else:
                key_to_evict = list(self._cache.keys())[0]
                
        elif self.strategy == CacheStrategy.TTL:
            # Remove oldest entry
            oldest_key = None
            oldest_time = float('inf')
            
            for key, entry in self._cache.items():
                if entry.created_at < oldest_time:
                    oldest_time = entry.created_at
                    oldest_key = key
            
            key_to_evict = oldest_key
            
        elif self.strategy == CacheStrategy.ADAPTIVE:
            # Remove entry with lowest value score
            lowest_score = float('inf')
            
            current_time = time.time()
            for key, entry in self._cache.items():
                # Calculate value score based on hit rate, recency, and TTL remaining
                age = current_time - entry.created_at
                recency = current_time - entry.last_accessed
                ttl_remaining = entry.ttl - age
                
                # Lower score = less valuable
                score = (
                    entry.hit_rate * 0.4 +           # 40% weight on hit rate
                    (1.0 / (1.0 + recency)) * 0.3 +  # 30% weight on recency
                    (ttl_remaining / entry.ttl) * 0.3 # 30% weight on TTL remaining
                )
                
                if score < lowest_score:
                    lowest_score = score
                    key_to_evict = key
        
        if key_to_evict:
            await self._evict_key(key_to_evict)
    
    async def _evict_key(self, key: str) -> None:
        """Remove a key from all cache structures."""
        if key in self._cache:
            del self._cache[key]
            self.stats.size -= 1
            self.stats.evictions += 1
        
        if key in self._access_order:
            self._access_order.remove(key)
        
        if key in self._frequency:
            del self._frequency[key]
    
    async def _cleanup_expired(self) -> None:
        """Background task to clean up expired entries."""
        async with self._lock:
            expired_keys = []
            current_time = time.time()
            
            for key, entry in self._cache.items():
                if entry.is_expired():
                    expired_keys.append(key)
            
            for key in expired_keys:
                await self._evict_key(key)
            
            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    async def start_background_cleanup(self) -> None:
        """Start background cleanup task."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info("Cache background cleanup started")
    
    async def stop_background_cleanup(self) -> None:
        """Stop background cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
            logger.info("Cache background cleanup stopped")
    
    async def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in cache cleanup loop: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "stats": {
                "hits": self.stats.hits,
                "misses": self.stats.misses,
                "evictions": self.stats.evictions,
                "hit_rate": self.stats.hit_rate,
                "size": self.stats.size,
                "max_size": self.stats.max_size,
                "memory_pressure": self.stats.memory_pressure
            },
            "config": {
                "strategy": self.strategy.value,
                "default_ttl": self.default_ttl,
                "cleanup_interval": self.cleanup_interval
            }
        }


# Specialized caches for different use cases
class DetectorResultCache(AdaptiveCache):
    """Cache for detector results with pattern-based invalidation."""
    
    def __init__(self):
        super().__init__(
            max_size=500,
            default_ttl=300.0,  # 5 minutes
            strategy=CacheStrategy.ADAPTIVE
        )
    
    async def cache_detector_result(
        self, 
        detector_name: str, 
        repo_full_name: str, 
        event_data: Dict[str, Any], 
        result: List[Dict[str, Any]]
    ) -> None:
        """Cache detector result with structured key."""
        key = {
            "detector": detector_name,
            "repo": repo_full_name,
            "event_hash": hashlib.md5(json.dumps(event_data, sort_keys=True).encode()).hexdigest()
        }
        
        await self.set(key, result, ttl=300.0)  # 5 minutes
    
    async def get_detector_result(
        self, 
        detector_name: str, 
        repo_full_name: str, 
        event_data: Dict[str, Any]
    ) -> Optional[List[Dict[str, Any]]]:
        """Get cached detector result."""
        key = {
            "detector": detector_name,
            "repo": repo_full_name,
            "event_hash": hashlib.md5(json.dumps(event_data, sort_keys=True).encode()).hexdigest()
        }
        
        return await self.get(key)
    
    async def invalidate_repo(self, repo_full_name: str) -> int:
        """Invalidate all results for a repository."""
        return await self.invalidate_pattern(repo_full_name)


class PlaybookResultCache(AdaptiveCache):
    """Cache for playbook execution results."""
    
    def __init__(self):
        super().__init__(
            max_size=200,
            default_ttl=600.0,  # 10 minutes
            strategy=CacheStrategy.LRU
        )
    
    async def cache_playbook_result(
        self, 
        playbook_name: str, 
        context_hash: str, 
        result: List[Any]
    ) -> None:
        """Cache playbook execution result."""
        key = f"{playbook_name}:{context_hash}"
        await self.set(key, result, ttl=600.0)
    
    async def get_playbook_result(
        self, 
        playbook_name: str, 
        context_hash: str
    ) -> Optional[List[Any]]:
        """Get cached playbook result."""
        key = f"{playbook_name}:{context_hash}"
        return await self.get(key)


# Global cache instances
detector_cache = DetectorResultCache()
playbook_cache = PlaybookResultCache()
general_cache = AdaptiveCache(max_size=1000, strategy=CacheStrategy.ADAPTIVE)


def cached(
    cache: AdaptiveCache = None, 
    ttl: Optional[float] = None,
    key_func: Optional[Callable] = None
):
    """Decorator for caching function results."""
    def decorator(func: Callable) -> Callable:
        cache_instance = cache or general_cache
        
        async def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = (func.__name__, args, tuple(sorted(kwargs.items())))
            
            # Try to get from cache
            cached_result = await cache_instance.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            await cache_instance.set(cache_key, result, ttl=ttl)
            
            logger.debug(f"Cache miss for {func.__name__}, result cached")
            return result
        
        return wrapper
    return decorator