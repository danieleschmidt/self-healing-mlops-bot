"""Enterprise-grade multi-level caching system with intelligent strategies."""

import asyncio
import time
import json
import hashlib
import pickle
import sqlite3
import gzip
import os
import threading
from typing import Any, Dict, Optional, Callable, Union, List, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import wraps
from enum import Enum
from pathlib import Path
import redis.asyncio as redis
from collections import defaultdict, OrderedDict, deque
import weakref
import psutil
import struct
import zlib
import logging

logger = logging.getLogger(__name__)


class CacheLevel(Enum):
    """Cache levels for multi-tier caching."""
    L1_MEMORY = "l1_memory"
    L2_REDIS = "l2_redis"
    L3_PERSISTENT = "l3_persistent"


class CacheStrategy(Enum):
    """Cache invalidation strategies."""
    TTL = "ttl"
    LRU = "lru"
    LFU = "lfu"
    WRITE_THROUGH = "write_through"
    WRITE_BEHIND = "write_behind"
    READ_ASIDE = "read_aside"


class InvalidationStrategy(Enum):
    """Cache invalidation patterns."""
    IMMEDIATE = "immediate"
    LAZY = "lazy"
    TIME_BASED = "time_based"
    VERSION_BASED = "version_based"
    DEPENDENCY_BASED = "dependency_based"


@dataclass
class CacheStats:
    """Comprehensive cache statistics for monitoring."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    writes: int = 0
    deletes: int = 0
    errors: int = 0
    last_hit: Optional[datetime] = None
    last_miss: Optional[datetime] = None
    total_size_bytes: int = 0
    avg_key_size: float = 0.0
    avg_value_size: float = 0.0
    memory_usage: int = 0
    network_calls: int = 0
    disk_reads: int = 0
    disk_writes: int = 0
    compression_ratio: float = 0.0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    @property
    def error_rate(self) -> float:
        """Calculate error rate."""
        total = self.hits + self.misses + self.errors
        return self.errors / total if total > 0 else 0.0
    
    @property
    def efficiency_score(self) -> float:
        """Calculate overall cache efficiency score."""
        if self.hits + self.misses == 0:
            return 0.0
        base_score = self.hit_rate * 100
        error_penalty = self.error_rate * 50
        memory_efficiency = min(1.0, self.memory_usage / (1024 * 1024 * 100)) * 10  # Penalty for >100MB
        return max(0, base_score - error_penalty - memory_efficiency)
    
    def reset(self):
        """Reset all statistics."""
        self.__dict__.update(CacheStats().__dict__)


@dataclass
class CacheEntry:
    """Enhanced cache entry with comprehensive metadata."""
    value: Any
    created_at: float
    last_accessed: float
    access_count: int = 0
    ttl: Optional[float] = None
    size_bytes: int = 0
    version: int = 1
    dependency_keys: Set[str] = field(default_factory=set)
    tags: Set[str] = field(default_factory=set)
    priority: int = 0  # Higher priority = less likely to evict
    compressed: bool = False
    checksum: str = ""
    access_pattern_score: float = 0.0
    
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.ttl is None:
            return False
        return time.time() > (self.created_at + self.ttl)
    
    def touch(self):
        """Update access information."""
        self.last_accessed = time.time()
        self.access_count += 1
        # Update access pattern score (exponential decay with recent access boost)
        current_time = time.time()
        time_since_creation = current_time - self.created_at
        if time_since_creation > 0:
            frequency = self.access_count / time_since_creation
            recency = 1.0 / (1.0 + (current_time - self.last_accessed))
            self.access_pattern_score = frequency * 0.7 + recency * 0.3
    
    def calculate_value_score(self) -> float:
        """Calculate overall value score for eviction decisions."""
        age_penalty = (time.time() - self.created_at) / 3600  # Hours since creation
        size_penalty = self.size_bytes / (1024 * 1024)  # MB penalty
        
        value_score = (
            self.access_pattern_score * 0.4 +
            self.priority * 0.3 +
            (1.0 / (1.0 + age_penalty)) * 0.2 +
            (1.0 / (1.0 + size_penalty)) * 0.1
        )
        
        return max(0.0, value_score)


class BloomFilter:
    """Simple Bloom filter for negative cache optimization."""
    
    def __init__(self, capacity: int = 100000, error_rate: float = 0.01):
        self.capacity = capacity
        self.error_rate = error_rate
        self.bit_array = bytearray(self._calculate_bit_array_size())
        self.hash_count = self._calculate_hash_count()
        
    def _calculate_bit_array_size(self) -> int:
        """Calculate optimal bit array size."""
        return int(-self.capacity * math.log(self.error_rate) / (math.log(2) ** 2))
    
    def _calculate_hash_count(self) -> int:
        """Calculate optimal number of hash functions."""
        return int((len(self.bit_array) / self.capacity) * math.log(2))
    
    def _hash(self, key: str, seed: int) -> int:
        """Generate hash for given key and seed."""
        return hash(key + str(seed)) % len(self.bit_array)
    
    def add(self, key: str):
        """Add key to bloom filter."""
        for i in range(self.hash_count):
            bit_index = self._hash(key, i)
            self.bit_array[bit_index] = 1
    
    def might_contain(self, key: str) -> bool:
        """Check if key might be in the set."""
        for i in range(self.hash_count):
            bit_index = self._hash(key, i)
            if self.bit_array[bit_index] == 0:
                return False
        return True


class CacheWarmer:
    """Intelligent cache warming system."""
    
    def __init__(self, cache_instance):
        self.cache = cache_instance
        self.warming_schedule: Dict[str, Dict] = {}
        self.warming_patterns: Dict[str, Callable] = {}
        self.active_warming_tasks: Set[asyncio.Task] = set()
        
    async def schedule_warming(
        self,
        pattern: str,
        warm_func: Callable,
        schedule: str = "*/15 * * * *",  # Every 15 minutes
        priority: int = 0
    ):
        """Schedule cache warming for a pattern."""
        self.warming_schedule[pattern] = {
            "func": warm_func,
            "schedule": schedule,
            "priority": priority,
            "last_run": 0
        }
        logger.info(f"Scheduled cache warming for pattern: {pattern}")
    
    async def warm_by_access_patterns(self):
        """Warm cache based on historical access patterns."""
        current_time = time.time()
        
        # Analyze access patterns
        popular_keys = []
        for key, access_times in self.cache.access_patterns.items():
            if len(access_times) < 3:
                continue
                
            # Calculate access frequency
            recent_accesses = [t for t in access_times if current_time - t < 3600]  # Last hour
            if len(recent_accesses) >= 5:  # Popular key
                popular_keys.append((key, len(recent_accesses)))
        
        # Sort by popularity and warm top keys
        popular_keys.sort(key=lambda x: x[1], reverse=True)
        
        for key, _ in popular_keys[:50]:  # Top 50 keys
            try:
                # Check if key is about to expire or already expired
                entry = await self.cache._get_from_any_level(key)
                if entry and entry.is_expired():
                    # Re-warm the key if we have a warming function
                    await self._execute_warming(key)
            except Exception as e:
                logger.error(f"Error warming key {key}: {e}")
    
    async def _execute_warming(self, key: str):
        """Execute warming for a specific key."""
        # This would be implemented based on specific warming strategies
        logger.debug(f"Would warm cache key: {key}")


class DistributedCache:
    """Distributed cache coordination for multiple instances."""
    
    def __init__(self, cache_instance, node_id: str = None):
        self.cache = cache_instance
        self.node_id = node_id or f"node_{int(time.time())}"
        self.peer_nodes: Set[str] = set()
        self.invalidation_channel = "cache_invalidation"
        self.sync_channel = "cache_sync"
        
    async def initialize_distributed(self):
        """Initialize distributed cache features."""
        if self.cache.redis_client:
            # Subscribe to invalidation messages
            pubsub = self.cache.redis_client.pubsub()
            await pubsub.subscribe(self.invalidation_channel)
            
            # Start listening for distributed invalidations
            asyncio.create_task(self._listen_for_invalidations(pubsub))
    
    async def broadcast_invalidation(self, keys: List[str], source_node: str = None):
        """Broadcast invalidation to other cache instances."""
        if not self.cache.redis_client:
            return
            
        message = {
            "type": "invalidate",
            "keys": keys,
            "source_node": source_node or self.node_id,
            "timestamp": time.time()
        }
        
        await self.cache.redis_client.publish(
            self.invalidation_channel, 
            json.dumps(message)
        )
    
    async def _listen_for_invalidations(self, pubsub):
        """Listen for distributed invalidation messages."""
        async for message in pubsub.listen():
            if message["type"] == "message":
                try:
                    data = json.loads(message["data"])
                    if data["source_node"] != self.node_id:
                        # Invalidate keys from other nodes
                        for key in data["keys"]:
                            await self.cache.invalidate(
                                key, 
                                levels={CacheLevel.L1_MEMORY},  # Only invalidate local
                                cascade=False
                            )
                except Exception as e:
                    logger.error(f"Error processing distributed invalidation: {e}")


class MultiLevelCache:
    """Enterprise-grade multi-level cache with intelligent distribution."""
    
    def __init__(
        self, 
        redis_url: str = None,
        max_l1_size: int = 10000,
        max_l1_memory_mb: int = 512,
        persistent_cache_dir: str = None,
        enable_compression: bool = True,
        enable_encryption: bool = False,
        default_ttl: float = 3600.0,
        compression_threshold: int = 1024,
        enable_distributed: bool = False
    ):
        # Configuration
        self.redis_url = redis_url or "redis://localhost:6379"
        self.max_l1_size = max_l1_size
        self.max_l1_memory_mb = max_l1_memory_mb * 1024 * 1024  # Convert to bytes
        self.persistent_cache_dir = Path(persistent_cache_dir or "/tmp/bot_cache")
        self.enable_compression = enable_compression
        self.enable_encryption = enable_encryption
        self.default_ttl = default_ttl
        self.compression_threshold = compression_threshold
        self.enable_distributed = enable_distributed
        
        # L1 Cache (In-Memory)
        self.l1_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.l1_lock = asyncio.Lock()
        
        # L2 Cache (Redis)
        self.redis_client: Optional[redis.Redis] = None
        self.redis_pool: Optional[redis.ConnectionPool] = None
        
        # L3 Cache (Persistent)
        self.l3_db_path = self.persistent_cache_dir / "cache.db"
        self.l3_lock = asyncio.Lock()
        
        # Statistics and monitoring
        self.stats: Dict[str, CacheStats] = {
            CacheLevel.L1_MEMORY.value: CacheStats(),
            CacheLevel.L2_REDIS.value: CacheStats(),
            CacheLevel.L3_PERSISTENT.value: CacheStats(),
            "global": CacheStats()
        }
        
        # Advanced features
        self.access_patterns: Dict[str, List[float]] = defaultdict(list)
        self.dependency_graph: Dict[str, Set[str]] = defaultdict(set)  # key -> dependent keys
        self.tag_index: Dict[str, Set[str]] = defaultdict(set)  # tag -> keys
        self.warming_tasks: Dict[str, asyncio.Task] = {}
        self.background_tasks: Set[asyncio.Task] = set()
        
        # Cache strategy configuration
        self.default_strategy = CacheStrategy.READ_ASIDE
        self.invalidation_strategy = InvalidationStrategy.IMMEDIATE
        self.adaptive_ttl_enabled = True
        self.write_behind_buffer: Dict[str, CacheEntry] = {}
        self.write_behind_interval = 5.0  # seconds
        
        # Performance optimization
        self.bloom_filter = BloomFilter() if max_l1_size > 1000 else None
        self.circuit_breaker_failures = defaultdict(int)
        self.circuit_breaker_threshold = 5
        self.circuit_breaker_timeout = 60
        self.circuit_breaker_last_failure = defaultdict(float)
        
        # Distributed caching
        self.distributed_cache = None
        if enable_distributed:
            self.distributed_cache = DistributedCache(self)
        
        # Cache warming
        self.cache_warmer = CacheWarmer(self)
        
        # Connection pooling and resource management
        self.connection_semaphore = asyncio.Semaphore(50)  # Limit concurrent connections
    
    async def initialize(self):
        """Initialize all cache levels."""
        try:
            # Initialize Redis connection pool
            if self.redis_url:
                self.redis_pool = redis.ConnectionPool.from_url(
                    self.redis_url,
                    max_connections=20,
                    retry_on_timeout=True,
                    socket_keepalive=True,
                    socket_keepalive_options={},
                    health_check_interval=30
                )
                self.redis_client = redis.Redis(connection_pool=self.redis_pool)
                await self.redis_client.ping()
                logger.info("Redis L2 cache initialized successfully")
        except Exception as e:
            logger.warning(f"Redis L2 cache unavailable: {e}")
            self.redis_client = None
        
        # Initialize persistent cache directory
        try:
            self.persistent_cache_dir.mkdir(parents=True, exist_ok=True)
            await self._init_persistent_cache()
            logger.info(f"Persistent L3 cache initialized at {self.persistent_cache_dir}")
        except Exception as e:
            logger.error(f"Failed to initialize persistent cache: {e}")
        
        # Initialize distributed cache
        if self.distributed_cache:
            await self.distributed_cache.initialize_distributed()
        
        # Start background tasks
        await self._start_background_tasks()
        
        logger.info("Multi-level cache system initialized successfully")
    
    async def _init_persistent_cache(self):
        """Initialize persistent cache database."""
        def init_db():
            conn = sqlite3.connect(str(self.l3_db_path))
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    value BLOB,
                    created_at REAL,
                    last_accessed REAL,
                    access_count INTEGER DEFAULT 0,
                    ttl REAL,
                    size_bytes INTEGER,
                    version INTEGER DEFAULT 1,
                    tags TEXT,
                    priority INTEGER DEFAULT 0,
                    checksum TEXT,
                    compressed BOOLEAN DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_last_accessed ON cache_entries(last_accessed)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at ON cache_entries(created_at)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_tags ON cache_entries(tags)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_ttl ON cache_entries(created_at, ttl)
            """)
            conn.commit()
            conn.close()
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, init_db)
    
    async def _start_background_tasks(self):
        """Start background maintenance tasks."""
        tasks = [
            ("L1 cleanup", self._l1_cleanup_loop),
            ("Write-behind flush", self._write_behind_loop),
            ("Statistics collection", self._stats_collection_loop),
            ("Access pattern analysis", self._access_pattern_analysis_loop),
            ("Cache warming", self._cache_warming_loop),
            ("Circuit breaker reset", self._circuit_breaker_reset_loop)
        ]
        
        for name, coro in tasks:
            try:
                task = asyncio.create_task(coro())
                self.background_tasks.add(task)
                logger.debug(f"Started background task: {name}")
            except Exception as e:
                logger.error(f"Failed to start background task {name}: {e}")
    
    async def get(
        self, 
        key: str, 
        namespace: str = "default",
        strategy: CacheStrategy = None,
        promote_on_hit: bool = True,
        warm_on_miss: bool = False
    ) -> Optional[Any]:
        """Get value from multi-level cache with intelligent promotion."""
        cache_key = self._build_key(key, namespace)
        start_time = time.time()
        strategy = strategy or self.default_strategy
        found_level = None
        value = None
        
        # Check bloom filter for early miss detection
        if self.bloom_filter and not self.bloom_filter.might_contain(cache_key):
            self._record_miss("global", start_time)
            return None
        
        try:
            # L1 Cache (Memory) - Fastest
            async with self.l1_lock:
                if cache_key in self.l1_cache:
                    entry = self.l1_cache[cache_key]
                    if not entry.is_expired():
                        entry.touch()
                        # Move to end for LRU
                        self.l1_cache.move_to_end(cache_key)
                        value = entry.value
                        found_level = CacheLevel.L1_MEMORY
                        self._record_hit(CacheLevel.L1_MEMORY.value, start_time)
                        self._update_access_pattern(cache_key)
                        return self._decompress_if_needed(value, entry.compressed)
                    else:
                        # Expired, remove from L1
                        del self.l1_cache[cache_key]
                        self.stats[CacheLevel.L1_MEMORY.value].evictions += 1
            
            # L2 Cache (Redis) - Network
            if (self.redis_client and 
                not self._is_circuit_breaker_open(CacheLevel.L2_REDIS.value)):
                
                async with self.connection_semaphore:
                    try:
                        cached_data = await self.redis_client.get(cache_key)
                        if cached_data:
                            entry_data = pickle.loads(cached_data)
                            entry = CacheEntry(**entry_data)
                            
                            if not entry.is_expired():
                                value = entry.value
                                found_level = CacheLevel.L2_REDIS
                                self._record_hit(CacheLevel.L2_REDIS.value, start_time)
                                self._update_access_pattern(cache_key)
                                self.stats[CacheLevel.L2_REDIS.value].network_calls += 1
                                
                                # Promote to L1 if configured
                                if promote_on_hit:
                                    await self._promote_to_l1(cache_key, entry)
                                
                                return self._decompress_if_needed(value, entry.compressed)
                            else:
                                # Expired, remove from Redis
                                await self.redis_client.delete(cache_key)
                                self.stats[CacheLevel.L2_REDIS.value].evictions += 1
                                
                    except Exception as e:
                        logger.warning(f"Redis L2 cache error: {e}")
                        self._record_circuit_breaker_failure(CacheLevel.L2_REDIS.value)
                        self.stats[CacheLevel.L2_REDIS.value].errors += 1
            
            # L3 Cache (Persistent) - Slowest but most durable
            async with self.l3_lock:
                try:
                    def get_from_persistent():
                        conn = sqlite3.connect(str(self.l3_db_path))
                        cursor = conn.cursor()
                        cursor.execute("""
                            SELECT value, created_at, ttl, access_count, size_bytes, 
                                   version, tags, priority, compressed, checksum
                            FROM cache_entries WHERE key = ?
                        """, (cache_key,))
                        row = cursor.fetchone()
                        conn.close()
                        return row
                    
                    loop = asyncio.get_event_loop()
                    row = await loop.run_in_executor(None, get_from_persistent)
                    
                    if row:
                        (value_blob, created_at, ttl, access_count, size_bytes, 
                         version, tags_str, priority, compressed, checksum) = row
                        
                        # Check expiration
                        if ttl and time.time() > (created_at + ttl):
                            # Expired, remove from L3
                            await self._remove_from_l3(cache_key)
                            self.stats[CacheLevel.L3_PERSISTENT.value].evictions += 1
                        else:
                            value = pickle.loads(value_blob)
                            found_level = CacheLevel.L3_PERSISTENT
                            self._record_hit(CacheLevel.L3_PERSISTENT.value, start_time)
                            self._update_access_pattern(cache_key)
                            self.stats[CacheLevel.L3_PERSISTENT.value].disk_reads += 1
                            
                            # Update access count
                            await self._update_l3_access(cache_key)
                            
                            # Promote to higher levels if configured
                            if promote_on_hit:
                                entry = CacheEntry(
                                    value=value,
                                    created_at=created_at,
                                    last_accessed=time.time(),
                                    access_count=access_count + 1,
                                    ttl=ttl,
                                    size_bytes=size_bytes,
                                    version=version,
                                    tags=set(tags_str.split(",")) if tags_str else set(),
                                    priority=priority,
                                    compressed=bool(compressed),
                                    checksum=checksum
                                )
                                await self._promote_to_higher_levels(cache_key, entry)
                            
                            return self._decompress_if_needed(value, bool(compressed))
                            
                except Exception as e:
                    logger.warning(f"Persistent L3 cache error: {e}")
                    self.stats[CacheLevel.L3_PERSISTENT.value].errors += 1
            
            # Complete cache miss
            self._record_miss("global", start_time)
            
            # Optional: trigger cache warming on miss
            if warm_on_miss:
                asyncio.create_task(self._warm_on_miss(cache_key))
            
            return None
            
        except Exception as e:
            logger.error(f"Multi-level cache get error for key {cache_key}: {e}")
            self.stats["global"].errors += 1
            return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
        namespace: str = "default",
        strategy: CacheStrategy = None,
        tags: Set[str] = None,
        dependencies: Set[str] = None,
        priority: int = 0,
        force_level: CacheLevel = None
    ):
        """Set value in multi-level cache with intelligent distribution."""
        cache_key = self._build_key(key, namespace)
        strategy = strategy or self.default_strategy
        tags = tags or set()
        dependencies = dependencies or set()
        
        # Calculate adaptive TTL
        if ttl is None and self.adaptive_ttl_enabled:
            ttl = self._calculate_adaptive_ttl(cache_key, value)
        elif ttl is None:
            ttl = self.default_ttl
        
        # Compress large values if enabled
        compressed_value, is_compressed = self._compress_if_needed(value)
        value_size = len(pickle.dumps(compressed_value))
        
        # Generate checksum for integrity
        checksum = hashlib.sha256(pickle.dumps(value)).hexdigest()[:16]
        
        # Create cache entry
        entry = CacheEntry(
            value=compressed_value,
            created_at=time.time(),
            last_accessed=time.time(),
            access_count=0,
            ttl=ttl,
            size_bytes=value_size,
            version=1,
            dependency_keys=dependencies,
            tags=tags,
            priority=priority,
            compressed=is_compressed,
            checksum=checksum
        )
        
        try:
            # Update dependency graph and tag index
            self._update_dependency_graph(cache_key, dependencies)
            self._update_tag_index(cache_key, tags)
            
            # Add to bloom filter
            if self.bloom_filter:
                self.bloom_filter.add(cache_key)
            
            if strategy == CacheStrategy.WRITE_THROUGH:
                # Write to all levels synchronously
                await self._write_to_all_levels(cache_key, entry, force_level)
                
                # Broadcast invalidation for distributed cache
                if self.distributed_cache:
                    await self.distributed_cache.broadcast_invalidation([cache_key])
                    
            elif strategy == CacheStrategy.WRITE_BEHIND:
                # Add to write-behind buffer
                self.write_behind_buffer[cache_key] = entry
                # Always write to L1 immediately for read performance
                await self._write_to_l1(cache_key, entry)
            else:  # READ_ASIDE or default
                # Intelligent level selection based on access patterns and size
                await self._intelligent_write(cache_key, entry, force_level)
            
            self.stats["global"].writes += 1
            
        except Exception as e:
            logger.error(f"Multi-level cache set error for key {cache_key}: {e}")
            self.stats["global"].errors += 1
    
    async def invalidate(
        self, 
        key: str, 
        namespace: str = "default",
        cascade: bool = True,
        levels: Set[CacheLevel] = None,
        strategy: InvalidationStrategy = None
    ):
        """Invalidate cache entry across all or specified levels."""
        cache_key = self._build_key(key, namespace)
        levels = levels or {CacheLevel.L1_MEMORY, CacheLevel.L2_REDIS, CacheLevel.L3_PERSISTENT}
        strategy = strategy or self.invalidation_strategy
        
        try:
            if strategy == InvalidationStrategy.IMMEDIATE:
                await self._immediate_invalidate(cache_key, levels, cascade)
            elif strategy == InvalidationStrategy.LAZY:
                await self._lazy_invalidate(cache_key, levels)
            elif strategy == InvalidationStrategy.VERSION_BASED:
                await self._version_based_invalidate(cache_key, levels)
            else:
                await self._immediate_invalidate(cache_key, levels, cascade)
            
            # Broadcast invalidation for distributed cache
            if self.distributed_cache:
                await self.distributed_cache.broadcast_invalidation([cache_key])
            
            self.stats["global"].deletes += 1
            
        except Exception as e:
            logger.error(f"Cache invalidate error for key {cache_key}: {e}")
            self.stats["global"].errors += 1
    
    async def _immediate_invalidate(self, cache_key: str, levels: Set[CacheLevel], cascade: bool):
        """Immediately invalidate from specified levels."""
        # Remove from L1 cache
        if CacheLevel.L1_MEMORY in levels:
            async with self.l1_lock:
                if cache_key in self.l1_cache:
                    del self.l1_cache[cache_key]
                    self.stats[CacheLevel.L1_MEMORY.value].deletes += 1
        
        # Remove from Redis
        if CacheLevel.L2_REDIS in levels and self.redis_client:
            try:
                async with self.connection_semaphore:
                    deleted = await self.redis_client.delete(cache_key)
                    if deleted:
                        self.stats[CacheLevel.L2_REDIS.value].deletes += 1
                    self.stats[CacheLevel.L2_REDIS.value].network_calls += 1
            except Exception as e:
                logger.warning(f"Redis delete error: {e}")
                self.stats[CacheLevel.L2_REDIS.value].errors += 1
        
        # Remove from persistent cache
        if CacheLevel.L3_PERSISTENT in levels:
            await self._remove_from_l3(cache_key)
        
        # Cascade invalidation to dependent keys
        if cascade:
            await self._cascade_invalidate(cache_key)
        
        # Clean up metadata
        self._cleanup_key_metadata(cache_key)
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics across all levels."""
        l1_memory_usage = sum(entry.size_bytes for entry in self.l1_cache.values())
        
        return {
            "levels": {
                level_name: {
                    "hits": stats.hits,
                    "misses": stats.misses,
                    "writes": stats.writes,
                    "deletes": stats.deletes,
                    "evictions": stats.evictions,
                    "errors": stats.errors,
                    "hit_rate": stats.hit_rate,
                    "error_rate": stats.error_rate,
                    "efficiency_score": stats.efficiency_score,
                    "memory_usage": stats.memory_usage,
                    "network_calls": stats.network_calls,
                    "disk_reads": stats.disk_reads,
                    "disk_writes": stats.disk_writes,
                    "compression_ratio": stats.compression_ratio,
                    "last_hit": stats.last_hit.isoformat() if stats.last_hit else None,
                    "last_miss": stats.last_miss.isoformat() if stats.last_miss else None
                }
                for level_name, stats in self.stats.items()
            },
            "l1_cache_info": {
                "size": len(self.l1_cache),
                "max_size": self.max_l1_size,
                "memory_usage_bytes": l1_memory_usage,
                "memory_limit_bytes": self.max_l1_memory_mb,
                "memory_utilization": l1_memory_usage / self.max_l1_memory_mb if self.max_l1_memory_mb > 0 else 0,
                "avg_entry_size": l1_memory_usage / len(self.l1_cache) if self.l1_cache else 0,
                "top_accessed_keys": self._get_top_accessed_keys()
            },
            "background_tasks": {
                "write_behind_buffer_size": len(self.write_behind_buffer),
                "warming_tasks_active": len(self.warming_tasks),
                "background_tasks_count": len(self.background_tasks),
                "active_tasks": [task.get_name() for task in self.background_tasks if not task.done()]
            },
            "circuit_breakers": {
                level: {
                    "failures": self.circuit_breaker_failures[level],
                    "is_open": self._is_circuit_breaker_open(level),
                    "last_failure": self.circuit_breaker_last_failure.get(level, 0)
                }
                for level in [CacheLevel.L2_REDIS.value, CacheLevel.L3_PERSISTENT.value]
            },
            "access_patterns": {
                "total_keys_tracked": len(self.access_patterns),
                "dependency_relationships": sum(len(deps) for deps in self.dependency_graph.values()),
                "tag_index_size": len(self.tag_index),
                "hot_keys": self._get_hot_keys(),
                "cold_keys": self._get_cold_keys()
            },
            "performance_metrics": {
                "bloom_filter_enabled": self.bloom_filter is not None,
                "distributed_cache_enabled": self.distributed_cache is not None,
                "compression_enabled": self.enable_compression,
                "avg_response_time_ms": self._calculate_avg_response_time()
            }
        }
    
    async def warm_cache(
        self, 
        keys: List[str], 
        warm_func: Optional[Callable] = None,
        priority: int = 0
    ):
        """Warm cache with specified keys."""
        if not warm_func:
            logger.warning("No warming function provided")
            return
        
        warming_tasks = []
        for key in keys:
            task = asyncio.create_task(
                self._warm_single_key(key, warm_func, priority)
            )
            warming_tasks.append(task)
        
        # Execute warming tasks concurrently
        results = await asyncio.gather(*warming_tasks, return_exceptions=True)
        
        successful_warms = sum(1 for r in results if not isinstance(r, Exception))
        logger.info(f"Cache warming completed: {successful_warms}/{len(keys)} keys warmed successfully")
    
    async def _warm_single_key(self, key: str, warm_func: Callable, priority: int):
        """Warm a single cache key."""
        try:
            # Check if key is already cached and not expired
            existing_value = await self.get(key, promote_on_hit=False)
            if existing_value is not None:
                return  # Already cached
            
            # Generate value using warming function
            value = await warm_func(key) if asyncio.iscoroutinefunction(warm_func) else warm_func(key)
            
            if value is not None:
                await self.set(key, value, priority=priority)
                
        except Exception as e:
            logger.error(f"Error warming key {key}: {e}")
    
    # Helper methods for internal operations
    def _build_key(self, key: str, namespace: str) -> str:
        """Build cache key with namespace."""
        return f"bot:{namespace}:{key}"
    
    def _compress_if_needed(self, value: Any) -> tuple[Any, bool]:
        """Compress value if it exceeds threshold."""
        if not self.enable_compression:
            return value, False
        
        serialized = pickle.dumps(value)
        if len(serialized) > self.compression_threshold:
            compressed = gzip.compress(serialized)
            if len(compressed) < len(serialized) * 0.8:  # Only if compression saves at least 20%
                return compressed, True
        
        return value, False
    
    def _decompress_if_needed(self, value: Any, is_compressed: bool) -> Any:
        """Decompress value if needed."""
        if is_compressed and isinstance(value, bytes):
            try:
                decompressed = gzip.decompress(value)
                return pickle.loads(decompressed)
            except Exception as e:
                logger.error(f"Decompression error: {e}")
                return value
        return value
    
    def _calculate_adaptive_ttl(self, key: str, value: Any) -> float:
        """Calculate adaptive TTL based on access patterns and value characteristics."""
        base_ttl = self.default_ttl
        
        # Adjust based on access patterns
        access_times = self.access_patterns.get(key, [])
        if len(access_times) > 1:
            avg_interval = sum(access_times[i] - access_times[i-1] 
                              for i in range(1, len(access_times))) / (len(access_times) - 1)
            
            if avg_interval < 3600:  # Accessed frequently (< 1 hour intervals)
                base_ttl *= 2.0
            elif avg_interval > 86400:  # Accessed infrequently (> 1 day intervals)
                base_ttl *= 0.5
        
        # Adjust based on value size (larger values get longer TTL to amortize cost)
        try:
            value_size = len(pickle.dumps(value))
            if value_size > 100000:  # > 100KB
                base_ttl *= 1.5
            elif value_size < 1000:  # < 1KB
                base_ttl *= 0.8
        except:
            pass
        
        return max(300.0, min(base_ttl, 86400.0))  # Clamp between 5 minutes and 1 day
    
    def _update_access_pattern(self, key: str):
        """Update access pattern for key."""
        current_time = time.time()
        access_times = self.access_patterns[key]
        
        # Keep only recent access times (last 7 days)
        cutoff = current_time - 604800
        access_times[:] = [t for t in access_times if t > cutoff]
        
        # Add current access
        access_times.append(current_time)
        
        # Limit list size to prevent memory bloat
        if len(access_times) > 100:
            access_times[:] = access_times[-50:]
    
    def _record_hit(self, level: str, start_time: float):
        """Record cache hit for level."""
        self.stats[level].hits += 1
        self.stats[level].last_hit = datetime.utcnow()
        
        duration = time.time() - start_time
        # performance_logger.log_execution_time(
        #     f"cache_get_{level}", duration, True, result="hit"
        # )
    
    def _record_miss(self, level: str, start_time: float):
        """Record cache miss for level."""
        self.stats[level].misses += 1
        self.stats[level].last_miss = datetime.utcnow()
        
        duration = time.time() - start_time
        # performance_logger.log_execution_time(
        #     f"cache_get_{level}", duration, True, result="miss"
        # )
    
    def _is_circuit_breaker_open(self, level: str) -> bool:
        """Check if circuit breaker is open for level."""
        failures = self.circuit_breaker_failures[level]
        if failures < self.circuit_breaker_threshold:
            return False
        
        last_failure = self.circuit_breaker_last_failure.get(level, 0)
        return time.time() - last_failure < self.circuit_breaker_timeout
    
    def _record_circuit_breaker_failure(self, level: str):
        """Record circuit breaker failure."""
        self.circuit_breaker_failures[level] += 1
        self.circuit_breaker_last_failure[level] = time.time()
    
    # Background task implementations
    async def _l1_cleanup_loop(self):
        """Background cleanup for L1 cache."""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                await self._cleanup_expired_l1()
                await self._enforce_l1_memory_limit()
            except Exception as e:
                logger.error(f"L1 cleanup error: {e}")
    
    async def _write_behind_loop(self):
        """Background write-behind buffer flush."""
        while True:
            try:
                await asyncio.sleep(self.write_behind_interval)
                await self._flush_write_behind_buffer()
            except Exception as e:
                logger.error(f"Write-behind flush error: {e}")
    
    async def _stats_collection_loop(self):
        """Background statistics collection."""
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                await self._collect_memory_stats()
                await self._log_performance_metrics()
            except Exception as e:
                logger.error(f"Stats collection error: {e}")
    
    async def _access_pattern_analysis_loop(self):
        """Background access pattern analysis for optimization."""
        while True:
            try:
                await asyncio.sleep(900)  # Every 15 minutes
                await self._analyze_access_patterns()
                await self._optimize_cache_placement()
            except Exception as e:
                logger.error(f"Access pattern analysis error: {e}")
    
    async def _cache_warming_loop(self):
        """Background cache warming based on patterns."""
        while True:
            try:
                await asyncio.sleep(600)  # Every 10 minutes
                await self.cache_warmer.warm_by_access_patterns()
            except Exception as e:
                logger.error(f"Cache warming error: {e}")
    
    async def _circuit_breaker_reset_loop(self):
        """Background circuit breaker reset."""
        while True:
            try:
                await asyncio.sleep(self.circuit_breaker_timeout)
                current_time = time.time()
                
                for level in list(self.circuit_breaker_failures.keys()):
                    last_failure = self.circuit_breaker_last_failure.get(level, 0)
                    if current_time - last_failure > self.circuit_breaker_timeout:
                        # Reset circuit breaker
                        self.circuit_breaker_failures[level] = 0
                        logger.info(f"Circuit breaker reset for level: {level}")
                        
            except Exception as e:
                logger.error(f"Circuit breaker reset error: {e}")
    
    # Additional helper methods would be implemented here...
    async def _cleanup_expired_l1(self):
        """Clean up expired entries from L1 cache."""
        async with self.l1_lock:
            expired_keys = []
            for key, entry in self.l1_cache.items():
                if entry.is_expired():
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.l1_cache[key]
                self.stats[CacheLevel.L1_MEMORY.value].evictions += 1
            
            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired L1 entries")
    
    async def _enforce_l1_memory_limit(self):
        """Enforce L1 cache memory limits."""
        async with self.l1_lock:
            current_size = len(self.l1_cache)
            current_memory = sum(entry.size_bytes for entry in self.l1_cache.values())
            
            # Size-based eviction
            while current_size > self.max_l1_size:
                # Remove least valuable entry
                least_valuable_key = min(
                    self.l1_cache.keys(),
                    key=lambda k: self.l1_cache[k].calculate_value_score()
                )
                del self.l1_cache[least_valuable_key]
                current_size -= 1
                self.stats[CacheLevel.L1_MEMORY.value].evictions += 1
            
            # Memory-based eviction
            while current_memory > self.max_l1_memory_mb:
                # Remove largest entry with lowest value score
                largest_low_value_key = min(
                    self.l1_cache.keys(),
                    key=lambda k: (
                        self.l1_cache[k].calculate_value_score(),
                        -self.l1_cache[k].size_bytes
                    )
                )
                entry = self.l1_cache[largest_low_value_key]
                del self.l1_cache[largest_low_value_key]
                current_memory -= entry.size_bytes
                self.stats[CacheLevel.L1_MEMORY.value].evictions += 1
    
    def _get_top_accessed_keys(self, limit: int = 10) -> List[Dict]:
        """Get top accessed keys from L1 cache."""
        if not self.l1_cache:
            return []
        
        top_keys = sorted(
            self.l1_cache.items(),
            key=lambda x: x[1].access_count,
            reverse=True
        )[:limit]
        
        return [
            {
                "key": key,
                "access_count": entry.access_count,
                "size_bytes": entry.size_bytes,
                "value_score": entry.calculate_value_score()
            }
            for key, entry in top_keys
        ]
    
    def _get_hot_keys(self, limit: int = 20) -> List[str]:
        """Get hot keys based on recent access patterns."""
        current_time = time.time()
        hot_keys = []
        
        for key, access_times in self.access_patterns.items():
            recent_accesses = [t for t in access_times if current_time - t < 3600]  # Last hour
            if len(recent_accesses) >= 3:
                hot_keys.append((key, len(recent_accesses)))
        
        hot_keys.sort(key=lambda x: x[1], reverse=True)
        return [key for key, _ in hot_keys[:limit]]
    
    def _get_cold_keys(self, limit: int = 20) -> List[str]:
        """Get cold keys that haven't been accessed recently."""
        current_time = time.time()
        cold_keys = []
        
        for key, access_times in self.access_patterns.items():
            if access_times:
                last_access = max(access_times)
                if current_time - last_access > 86400:  # Not accessed in 24 hours
                    cold_keys.append((key, current_time - last_access))
        
        cold_keys.sort(key=lambda x: x[1], reverse=True)
        return [key for key, _ in cold_keys[:limit]]
    
    def _calculate_avg_response_time(self) -> float:
        """Calculate average response time across all levels."""
        # This would be implemented with proper timing measurements
        return 0.0  # Placeholder
    
    async def shutdown(self):
        """Gracefully shutdown the cache system."""
        logger.info("Shutting down multi-level cache system...")
        
        # Cancel all background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        # Flush write-behind buffer
        await self._flush_write_behind_buffer()
        
        # Close Redis connection
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("Multi-level cache system shutdown complete")


# Global cache instance
cache = MultiLevelCache()


# Decorator for caching function results
def cached(
    ttl: Optional[float] = None,
    namespace: str = "default",
    key_func: Optional[Callable] = None,
    strategy: CacheStrategy = CacheStrategy.READ_ASIDE,
    tags: Set[str] = None,
    priority: int = 0
):
    """Decorator for caching function results."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                key_parts = [func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = hashlib.md5(":".join(key_parts).encode()).hexdigest()
            
            # Try to get from cache
            cached_result = await cache.get(cache_key, namespace)
            if cached_result is not None:
                return cached_result
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache the result
            await cache.set(
                cache_key, result, ttl=ttl, namespace=namespace,
                strategy=strategy, tags=tags, priority=priority
            )
            
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For sync functions, we need to handle caching differently
            # This is a simplified version
            return func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Specialized cache utilities
class BotCacheUtils:
    """Utilities for bot-specific caching operations."""
    
    @staticmethod
    async def cache_github_file(repo: str, file_path: str, ref: str = "main") -> Optional[str]:
        """Cache GitHub file content with repository-specific TTL."""
        cache_key = f"github:{repo}:{file_path}:{ref}"
        
        # Try to get from cache first
        content = await cache.get(cache_key, namespace="github_api")
        if content is not None:
            return content
        
        # This would integrate with actual GitHub API
        # For now, return None to indicate miss
        return None
    
    @staticmethod
    async def invalidate_repo_cache(repo: str):
        """Invalidate all cache entries for a repository."""
        pattern = f"*{repo}*"
        invalidated = await cache.invalidate_pattern(pattern, "github_api")
        invalidated += await cache.invalidate_pattern(pattern, "detectors")
        logger.info(f"Invalidated {invalidated} cache entries for repository: {repo}")
        return invalidated
    
    @staticmethod
    async def warm_popular_repositories(repos: List[str]):
        """Warm cache for popular repositories."""
        common_files = [
            ".github/workflows/ci.yml",
            "requirements.txt",
            "package.json",
            "Dockerfile",
            "README.md"
        ]
        
        for repo in repos:
            for file_path in common_files:
                try:
                    await BotCacheUtils.cache_github_file(repo, file_path)
                except Exception as e:
                    logger.debug(f"Failed to warm cache for {repo}/{file_path}: {e}")