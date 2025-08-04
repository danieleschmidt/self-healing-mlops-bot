"""Intelligent caching system with adaptive strategies."""

import asyncio
import time
import json
import hashlib
from typing import Any, Dict, Optional, Callable, Union, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import wraps
import redis.asyncio as redis
from collections import defaultdict

from ..core.config import config
from ..monitoring.logging import get_logger, performance_logger
from ..monitoring.metrics import metrics

logger = get_logger(__name__)


@dataclass
class CacheStats:
    """Cache statistics for monitoring."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    last_hit: Optional[datetime] = None
    last_miss: Optional[datetime] = None
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class AdaptiveCache:
    """Intelligent cache with adaptive TTL and eviction strategies."""
    
    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or config.redis_url
        self.redis_client: Optional[redis.Redis] = None
        self.local_cache: Dict[str, Dict[str, Any]] = {}
        self.stats: Dict[str, CacheStats] = defaultdict(CacheStats)
        self.access_patterns: Dict[str, List[float]] = defaultdict(list)
        self.max_local_size = 1000
        self.adaptive_ttl_enabled = True
        
    async def initialize(self):
        """Initialize Redis connection."""
        try:
            self.redis_client = redis.from_url(self.redis_url)
            await self.redis_client.ping()
            logger.info("Redis cache initialized successfully")
        except Exception as e:
            logger.warning(f"Redis unavailable, using local cache only: {e}")
            self.redis_client = None
    
    async def get(self, key: str, namespace: str = "default") -> Optional[Any]:
        """Get value from cache with intelligent access pattern tracking."""
        cache_key = self._build_key(key, namespace)
        start_time = time.time()
        
        try:
            # Try local cache first
            if cache_key in self.local_cache:
                entry = self.local_cache[cache_key]
                if not self._is_expired(entry):
                    self._record_hit(namespace, start_time)
                    self._update_access_pattern(cache_key)
                    return entry["value"]
                else:
                    # Expired, remove from local cache
                    del self.local_cache[cache_key]
            
            # Try Redis cache
            if self.redis_client:
                try:
                    cached_data = await self.redis_client.get(cache_key)
                    if cached_data:
                        entry = json.loads(cached_data)
                        if not self._is_expired(entry):
                            # Store in local cache for faster future access
                            self._store_local(cache_key, entry)
                            self._record_hit(namespace, start_time)
                            self._update_access_pattern(cache_key)
                            return entry["value"]
                except Exception as e:
                    logger.warning(f"Redis get error: {e}")
            
            # Cache miss
            self._record_miss(namespace, start_time)
            return None
            
        except Exception as e:
            logger.error(f"Cache get error for key {cache_key}: {e}")
            return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        namespace: str = "default"
    ):
        """Set value in cache with adaptive TTL."""
        cache_key = self._build_key(key, namespace)
        
        # Calculate adaptive TTL
        if ttl is None and self.adaptive_ttl_enabled:
            ttl = self._calculate_adaptive_ttl(cache_key)
        elif ttl is None:
            ttl = 3600  # Default 1 hour
        
        entry = {
            "value": value,
            "created_at": time.time(),
            "ttl": ttl,
            "access_count": 0
        }
        
        try:
            # Store in local cache
            self._store_local(cache_key, entry)
            
            # Store in Redis
            if self.redis_client:
                try:
                    await self.redis_client.setex(
                        cache_key,
                        ttl,
                        json.dumps(entry, default=str)
                    )
                except Exception as e:
                    logger.warning(f"Redis set error: {e}")
                    
        except Exception as e:
            logger.error(f"Cache set error for key {cache_key}: {e}")
    
    async def invalidate(self, key: str, namespace: str = "default"):
        """Invalidate cache entry."""
        cache_key = self._build_key(key, namespace)
        
        # Remove from local cache
        if cache_key in self.local_cache:
            del self.local_cache[cache_key]
        
        # Remove from Redis
        if self.redis_client:
            try:
                await self.redis_client.delete(cache_key)
            except Exception as e:
                logger.warning(f"Redis delete error: {e}")
    
    async def invalidate_pattern(self, pattern: str, namespace: str = "default"):
        """Invalidate all keys matching pattern."""
        full_pattern = self._build_key(pattern, namespace)
        
        # Local cache invalidation
        keys_to_remove = [
            key for key in self.local_cache.keys()
            if self._matches_pattern(key, full_pattern)
        ]
        for key in keys_to_remove:
            del self.local_cache[key]
        
        # Redis invalidation
        if self.redis_client:
            try:
                keys = await self.redis_client.keys(full_pattern)
                if keys:
                    await self.redis_client.delete(*keys)
            except Exception as e:
                logger.warning(f"Redis pattern delete error: {e}")
    
    def _build_key(self, key: str, namespace: str) -> str:
        """Build cache key with namespace."""
        return f"bot:{namespace}:{key}"
    
    def _is_expired(self, entry: Dict[str, Any]) -> bool:
        """Check if cache entry is expired."""
        return time.time() > entry["created_at"] + entry["ttl"]
    
    def _store_local(self, key: str, entry: Dict[str, Any]):
        """Store entry in local cache with LRU eviction."""
        if len(self.local_cache) >= self.max_local_size:
            # Simple LRU: remove oldest entry
            oldest_key = min(
                self.local_cache.keys(),
                key=lambda k: self.local_cache[k]["created_at"]
            )
            del self.local_cache[oldest_key]
            self.stats["local"].evictions += 1
        
        self.local_cache[key] = entry
    
    def _calculate_adaptive_ttl(self, key: str) -> int:
        """Calculate adaptive TTL based on access patterns."""
        access_times = self.access_patterns.get(key, [])
        
        if len(access_times) < 2:
            return 3600  # Default 1 hour
        
        # Calculate access frequency
        recent_accesses = [t for t in access_times if time.time() - t < 86400]  # Last 24h
        
        if len(recent_accesses) > 10:  # High frequency
            return 7200  # 2 hours
        elif len(recent_accesses) > 5:  # Medium frequency
            return 3600  # 1 hour
        else:  # Low frequency
            return 1800  # 30 minutes
    
    def _update_access_pattern(self, key: str):
        """Update access pattern for adaptive TTL."""
        current_time = time.time()
        access_times = self.access_patterns[key]
        
        # Keep only recent access times (last 7 days)
        cutoff = current_time - 604800
        access_times[:] = [t for t in access_times if t > cutoff]
        
        # Add current access
        access_times.append(current_time)
        
        # Limit list size
        if len(access_times) > 100:
            access_times[:] = access_times[-50:]
    
    def _matches_pattern(self, key: str, pattern: str) -> bool:
        """Simple pattern matching for cache invalidation."""
        return pattern.replace("*", "") in key
    
    def _record_hit(self, namespace: str, start_time: float):
        """Record cache hit."""
        self.stats[namespace].hits += 1
        self.stats[namespace].last_hit = datetime.utcnow()
        
        duration = time.time() - start_time
        performance_logger.log_execution_time(
            "cache_get", duration, True, namespace=namespace, result="hit"
        )
    
    def _record_miss(self, namespace: str, start_time: float):
        """Record cache miss."""
        self.stats[namespace].misses += 1
        self.stats[namespace].last_miss = datetime.utcnow()
        
        duration = time.time() - start_time
        performance_logger.log_execution_time(
            "cache_get", duration, True, namespace=namespace, result="miss"
        )
    
    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get cache statistics."""
        return {
            namespace: {
                "hits": stats.hits,
                "misses": stats.misses,
                "hit_rate": stats.hit_rate,
                "evictions": stats.evictions,
                "last_hit": stats.last_hit.isoformat() if stats.last_hit else None,
                "last_miss": stats.last_miss.isoformat() if stats.last_miss else None
            }
            for namespace, stats in self.stats.items()
        }


def cache_result(
    ttl: Optional[int] = None,
    namespace: str = "default",
    key_func: Optional[Callable] = None
):
    """Decorator for caching function results."""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Generate key from function name and arguments
                key_parts = [func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = hashlib.md5(":".join(key_parts).encode()).hexdigest()
            
            # Try to get from cache
            cached_result = await cache.get(cache_key, namespace)
            if cached_result is not None:
                return cached_result
            
            # Execute function
            start_time = time.time()
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Cache the result
                await cache.set(cache_key, result, ttl, namespace)
                
                # Log performance
                duration = time.time() - start_time
                performance_logger.log_execution_time(
                    f"cached_function_{func.__name__}", duration, True
                )
                
                return result
                
            except Exception as e:
                # Log performance even on failure
                duration = time.time() - start_time
                performance_logger.log_execution_time(
                    f"cached_function_{func.__name__}", duration, False
                )
                raise
        
        return wrapper
    return decorator


class CacheWarmer:
    """Proactive cache warming based on usage patterns."""
    
    def __init__(self, cache_instance: AdaptiveCache):
        self.cache = cache_instance
        self.warming_tasks: Dict[str, asyncio.Task] = {}
    
    async def start_warming(self):
        """Start cache warming based on access patterns."""
        while True:
            try:
                await self._warm_popular_keys()
                await asyncio.sleep(300)  # Check every 5 minutes
            except Exception as e:
                logger.exception(f"Cache warming error: {e}")
                await asyncio.sleep(60)  # Retry after 1 minute on error
    
    async def _warm_popular_keys(self):
        """Warm cache for popular keys that are about to expire."""
        current_time = time.time()
        
        for key, access_times in self.cache.access_patterns.items():
            # Skip if not frequently accessed
            if len(access_times) < 5:
                continue
            
            # Check if key is in cache and about to expire
            if key in self.cache.local_cache:
                entry = self.cache.local_cache[key]
                time_to_expire = (entry["created_at"] + entry["ttl"]) - current_time
                
                # Warm if expiring within 10 minutes and frequently accessed
                if 0 < time_to_expire < 600 and len(access_times) > 10:
                    await self._warm_key(key)
    
    async def _warm_key(self, key: str):
        """Warm a specific cache key."""
        # This would implement key-specific warming logic
        # For now, just log that we would warm the key
        logger.debug(f"Would warm cache key: {key}")


# Global cache instance
cache = AdaptiveCache()


# Cache utilities for specific bot operations
class BotCacheUtils:
    """Utilities for bot-specific caching."""
    
    @staticmethod
    @cache_result(ttl=1800, namespace="github_api")  # 30 minutes
    async def cache_github_file_content(repo: str, file_path: str, ref: str) -> Optional[str]:
        """Cache GitHub file content with high TTL."""
        # This would be implemented by the actual GitHub integration
        return None
    
    @staticmethod
    @cache_result(ttl=300, namespace="detectors")  # 5 minutes
    async def cache_detector_results(repo: str, event_type: str, event_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Cache detector results for rapid re-processing."""
        # This would be implemented by the detector system
        return []
    
    @staticmethod
    @cache_result(ttl=3600, namespace="playbooks")  # 1 hour
    async def cache_playbook_metadata(playbook_name: str) -> Dict[str, Any]:
        """Cache playbook metadata."""
        # This would be implemented by the playbook system
        return {}
    
    @staticmethod
    async def invalidate_repo_cache(repo: str):
        """Invalidate all cache entries for a repository."""
        await cache.invalidate_pattern(f"*{repo}*", "github_api")
        await cache.invalidate_pattern(f"*{repo}*", "detectors")
        logger.info(f"Invalidated cache for repository: {repo}")
    
    @staticmethod
    async def warm_repo_cache(repo: str, common_files: List[str]):
        """Proactively warm cache for common repository files."""
        for file_path in common_files:
            try:
                await BotCacheUtils.cache_github_file_content(repo, file_path, "main")
            except Exception as e:
                logger.debug(f"Failed to warm cache for {repo}/{file_path}: {e}")