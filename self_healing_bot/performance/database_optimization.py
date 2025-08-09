"""Database query optimization and connection pooling system."""

import asyncio
import time
import logging
import hashlib
import pickle
import json
import statistics
from typing import Dict, List, Any, Optional, Callable, Union, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from enum import Enum
import sqlite3
import threading
from pathlib import Path

# Database drivers (import based on availability)
try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False

try:
    import aiomysql
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False

try:
    import aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import motor.motor_asyncio
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False

logger = logging.getLogger(__name__)


class DatabaseType(Enum):
    """Supported database types."""
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SQLITE = "sqlite"
    REDIS = "redis"
    MONGODB = "mongodb"


class QueryType(Enum):
    """Types of database queries."""
    SELECT = "select"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    TRANSACTION = "transaction"
    BULK_INSERT = "bulk_insert"
    AGGREGATE = "aggregate"


@dataclass
class QueryStats:
    """Statistics for a specific query."""
    query_hash: str
    query_text: str
    query_type: QueryType
    execution_count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    avg_time: float = 0.0
    error_count: int = 0
    last_executed: Optional[datetime] = None
    rows_affected: List[int] = field(default_factory=list)
    cache_hits: int = 0
    cache_misses: int = 0
    
    def update(self, execution_time: float, rows_affected: int = 0, cached: bool = False, error: bool = False):
        """Update query statistics."""
        self.execution_count += 1
        self.last_executed = datetime.now()
        
        if error:
            self.error_count += 1
        else:
            self.total_time += execution_time
            self.min_time = min(self.min_time, execution_time)
            self.max_time = max(self.max_time, execution_time)
            self.avg_time = self.total_time / (self.execution_count - self.error_count)
            self.rows_affected.append(rows_affected)
            
            if cached:
                self.cache_hits += 1
            else:
                self.cache_misses += 1
    
    @property
    def success_rate(self) -> float:
        """Calculate query success rate."""
        if self.execution_count == 0:
            return 0.0
        return (self.execution_count - self.error_count) / self.execution_count
    
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total_cacheable = self.cache_hits + self.cache_misses
        if total_cacheable == 0:
            return 0.0
        return self.cache_hits / total_cacheable
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_hash": self.query_hash,
            "query_text": self.query_text[:200] + "..." if len(self.query_text) > 200 else self.query_text,
            "query_type": self.query_type.value,
            "execution_count": self.execution_count,
            "avg_time": self.avg_time,
            "min_time": self.min_time if self.min_time != float('inf') else 0,
            "max_time": self.max_time,
            "error_count": self.error_count,
            "success_rate": self.success_rate,
            "cache_hit_rate": self.cache_hit_rate,
            "avg_rows_affected": statistics.mean(self.rows_affected) if self.rows_affected else 0,
            "last_executed": self.last_executed.isoformat() if self.last_executed else None
        }


@dataclass
class ConnectionPoolStats:
    """Statistics for database connection pool."""
    pool_name: str
    database_type: DatabaseType
    min_size: int
    max_size: int
    current_size: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    total_connections_created: int = 0
    total_connections_closed: int = 0
    connection_errors: int = 0
    avg_wait_time: float = 0.0
    max_wait_time: float = 0.0
    pool_exhaustion_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pool_name": self.pool_name,
            "database_type": self.database_type.value,
            "min_size": self.min_size,
            "max_size": self.max_size,
            "current_size": self.current_size,
            "active_connections": self.active_connections,
            "idle_connections": self.idle_connections,
            "utilization": self.active_connections / self.max_size if self.max_size > 0 else 0,
            "total_connections_created": self.total_connections_created,
            "total_connections_closed": self.total_connections_closed,
            "connection_errors": self.connection_errors,
            "avg_wait_time": self.avg_wait_time,
            "max_wait_time": self.max_wait_time,
            "pool_exhaustion_count": self.pool_exhaustion_count
        }


class QueryCache:
    """Intelligent query result caching system."""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, float] = {}
        self.hit_count = 0
        self.miss_count = 0
        
    def _generate_cache_key(self, query: str, params: tuple = None) -> str:
        """Generate cache key for query and parameters."""
        key_data = query
        if params:
            key_data += str(params)
        return hashlib.sha256(key_data.encode()).hexdigest()
    
    async def get(self, query: str, params: tuple = None) -> Optional[Any]:
        """Get cached query result."""
        cache_key = self._generate_cache_key(query, params)
        
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            
            # Check if expired
            if time.time() > entry["expires_at"]:
                del self.cache[cache_key]
                self.access_times.pop(cache_key, None)
                self.miss_count += 1
                return None
            
            # Update access time and return result
            self.access_times[cache_key] = time.time()
            self.hit_count += 1
            return entry["result"]
        
        self.miss_count += 1
        return None
    
    async def set(self, query: str, result: Any, params: tuple = None, ttl: int = None):
        """Cache query result."""
        cache_key = self._generate_cache_key(query, params)
        ttl = ttl or self.default_ttl
        
        # Evict oldest entries if cache is full
        if len(self.cache) >= self.max_size:
            # Remove least recently used entries
            sorted_keys = sorted(self.access_times.keys(), key=lambda k: self.access_times[k])
            for key in sorted_keys[:len(sorted_keys) // 4]:  # Remove 25% of entries
                self.cache.pop(key, None)
                self.access_times.pop(key, None)
        
        self.cache[cache_key] = {
            "result": result,
            "cached_at": time.time(),
            "expires_at": time.time() + ttl,
            "query": query[:100] + "..." if len(query) > 100 else query  # Store truncated query for debugging
        }
        self.access_times[cache_key] = time.time()
    
    async def invalidate_pattern(self, pattern: str):
        """Invalidate cache entries matching pattern."""
        keys_to_remove = []
        for key, entry in self.cache.items():
            if pattern.lower() in entry["query"].lower():
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            self.cache.pop(key, None)
            self.access_times.pop(key, None)
        
        return len(keys_to_remove)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0
        
        return {
            "cache_size": len(self.cache),
            "max_size": self.max_size,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate,
            "utilization": len(self.cache) / self.max_size
        }


class DatabaseConnectionPool:
    """Advanced database connection pool with monitoring and optimization."""
    
    def __init__(
        self,
        database_type: DatabaseType,
        connection_string: str,
        min_size: int = 5,
        max_size: int = 20,
        max_idle_time: int = 300,
        health_check_interval: int = 60,
        pool_name: str = "default"
    ):
        self.database_type = database_type
        self.connection_string = connection_string
        self.min_size = min_size
        self.max_size = max_size
        self.max_idle_time = max_idle_time
        self.health_check_interval = health_check_interval
        self.pool_name = pool_name
        
        # Pool management
        self.pool = None
        self.connection_semaphore = asyncio.Semaphore(max_size)
        self._lock = asyncio.Lock()
        
        # Statistics
        self.stats = ConnectionPoolStats(
            pool_name=pool_name,
            database_type=database_type,
            min_size=min_size,
            max_size=max_size
        )
        
        # Monitoring
        self.connection_wait_times = deque(maxlen=100)
        self.health_check_task: Optional[asyncio.Task] = None
        
    async def initialize(self):
        """Initialize the database connection pool."""
        try:
            if self.database_type == DatabaseType.POSTGRESQL and ASYNCPG_AVAILABLE:
                self.pool = await asyncpg.create_pool(
                    self.connection_string,
                    min_size=self.min_size,
                    max_size=self.max_size,
                    command_timeout=60
                )
            elif self.database_type == DatabaseType.MYSQL and MYSQL_AVAILABLE:
                self.pool = await aiomysql.create_pool(
                    **self._parse_mysql_connection_string(),
                    minsize=self.min_size,
                    maxsize=self.max_size
                )
            elif self.database_type == DatabaseType.REDIS and REDIS_AVAILABLE:
                self.pool = aioredis.ConnectionPool.from_url(
                    self.connection_string,
                    max_connections=self.max_size
                )
            elif self.database_type == DatabaseType.SQLITE:
                # SQLite doesn't use connection pooling in the traditional sense
                self.pool = self.connection_string
            else:\n                raise ValueError(f\"Database type {self.database_type} not supported or driver not available\")\n            \n            self.stats.current_size = self.min_size\n            self.stats.idle_connections = self.min_size\n            \n            # Start health check task\n            self.health_check_task = asyncio.create_task(self._health_check_loop())\n            \n            logger.info(f\"Database pool '{self.pool_name}' initialized: {self.database_type.value}\")\n            \n        except Exception as e:\n            logger.error(f\"Failed to initialize database pool '{self.pool_name}': {e}\")\n            raise\n    \n    def _parse_mysql_connection_string(self) -> Dict[str, Any]:\n        \"\"\"Parse MySQL connection string into parameters.\"\"\"\n        # Simple parser for MySQL connection strings\n        # Format: mysql://user:password@host:port/database\n        import urllib.parse\n        parsed = urllib.parse.urlparse(self.connection_string)\n        \n        return {\n            \"host\": parsed.hostname,\n            \"port\": parsed.port or 3306,\n            \"user\": parsed.username,\n            \"password\": parsed.password,\n            \"db\": parsed.path.lstrip('/') if parsed.path else None\n        }\n    \n    @asynccontextmanager\n    async def acquire_connection(self, timeout: float = 30.0):\n        \"\"\"Acquire a connection from the pool.\"\"\"\n        start_time = time.time()\n        \n        async with self.connection_semaphore:\n            try:\n                if self.database_type == DatabaseType.POSTGRESQL:\n                    async with self.pool.acquire() as conn:\n                        wait_time = time.time() - start_time\n                        self.connection_wait_times.append(wait_time)\n                        self._update_wait_time_stats(wait_time)\n                        \n                        self.stats.active_connections += 1\n                        self.stats.idle_connections -= 1\n                        \n                        yield conn\n                        \n                elif self.database_type == DatabaseType.MYSQL:\n                    async with self.pool.acquire() as conn:\n                        wait_time = time.time() - start_time\n                        self.connection_wait_times.append(wait_time)\n                        self._update_wait_time_stats(wait_time)\n                        \n                        self.stats.active_connections += 1\n                        self.stats.idle_connections -= 1\n                        \n                        yield conn\n                        \n                elif self.database_type == DatabaseType.SQLITE:\n                    # SQLite connection\n                    conn = sqlite3.connect(self.pool)\n                    conn.row_factory = sqlite3.Row  # Enable dict-like access\n                    try:\n                        yield conn\n                    finally:\n                        conn.close()\n                        \n                elif self.database_type == DatabaseType.REDIS:\n                    conn = aioredis.Redis(connection_pool=self.pool)\n                    yield conn\n                    \n            except Exception as e:\n                self.stats.connection_errors += 1\n                logger.error(f\"Connection acquisition error in pool '{self.pool_name}': {e}\")\n                raise\n            finally:\n                if self.database_type in [DatabaseType.POSTGRESQL, DatabaseType.MYSQL]:\n                    self.stats.active_connections -= 1\n                    self.stats.idle_connections += 1\n    \n    def _update_wait_time_stats(self, wait_time: float):\n        \"\"\"Update connection wait time statistics.\"\"\"\n        if self.connection_wait_times:\n            self.stats.avg_wait_time = statistics.mean(self.connection_wait_times)\n            self.stats.max_wait_time = max(self.connection_wait_times)\n    \n    async def _health_check_loop(self):\n        \"\"\"Background health check for connections.\"\"\"\n        while True:\n            try:\n                await asyncio.sleep(self.health_check_interval)\n                await self._perform_health_check()\n            except asyncio.CancelledError:\n                break\n            except Exception as e:\n                logger.error(f\"Health check error for pool '{self.pool_name}': {e}\")\n    \n    async def _perform_health_check(self):\n        \"\"\"Perform health check on the connection pool.\"\"\"\n        try:\n            async with self.acquire_connection(timeout=5.0) as conn:\n                if self.database_type == DatabaseType.POSTGRESQL:\n                    await conn.execute(\"SELECT 1\")\n                elif self.database_type == DatabaseType.MYSQL:\n                    async with conn.cursor() as cursor:\n                        await cursor.execute(\"SELECT 1\")\n                elif self.database_type == DatabaseType.SQLITE:\n                    conn.execute(\"SELECT 1\")\n                elif self.database_type == DatabaseType.REDIS:\n                    await conn.ping()\n                    \n                logger.debug(f\"Health check passed for pool '{self.pool_name}'\")\n                \n        except Exception as e:\n            logger.warning(f\"Health check failed for pool '{self.pool_name}': {e}\")\n            self.stats.connection_errors += 1\n    \n    async def shutdown(self):\n        \"\"\"Shutdown the connection pool.\"\"\"\n        if self.health_check_task:\n            self.health_check_task.cancel()\n            try:\n                await self.health_check_task\n            except asyncio.CancelledError:\n                pass\n        \n        if self.pool:\n            if hasattr(self.pool, 'close'):\n                if self.database_type == DatabaseType.POSTGRESQL:\n                    await self.pool.close()\n                elif self.database_type == DatabaseType.MYSQL:\n                    self.pool.close()\n                    await self.pool.wait_closed()\n        \n        logger.info(f\"Database pool '{self.pool_name}' shutdown complete\")\n    \n    def get_stats(self) -> Dict[str, Any]:\n        \"\"\"Get connection pool statistics.\"\"\"\n        return self.stats.to_dict()\n\n\nclass QueryOptimizer:\n    \"\"\"Intelligent query optimization and analysis system.\"\"\"\n    \n    def __init__(self):\n        self.query_stats: Dict[str, QueryStats] = {}\n        self.slow_query_threshold = 1.0  # seconds\n        self.optimization_suggestions: Dict[str, List[str]] = {}\n        \n    def _classify_query_type(self, query: str) -> QueryType:\n        \"\"\"Classify query type based on SQL content.\"\"\"\n        query_lower = query.strip().lower()\n        \n        if query_lower.startswith('select'):\n            if 'group by' in query_lower or 'count(' in query_lower or 'sum(' in query_lower:\n                return QueryType.AGGREGATE\n            return QueryType.SELECT\n        elif query_lower.startswith('insert'):\n            if 'values' in query_lower and query_lower.count('(') > 2:\n                return QueryType.BULK_INSERT\n            return QueryType.INSERT\n        elif query_lower.startswith('update'):\n            return QueryType.UPDATE\n        elif query_lower.startswith('delete'):\n            return QueryType.DELETE\n        elif query_lower.startswith(('begin', 'start', 'commit', 'rollback')):\n            return QueryType.TRANSACTION\n        else:\n            return QueryType.SELECT  # Default\n    \n    def _generate_query_hash(self, query: str) -> str:\n        \"\"\"Generate hash for query normalization.\"\"\"\n        # Normalize query by removing parameter values and extra whitespace\n        normalized = ' '.join(query.strip().split())\n        # Simple parameter normalization (replace numbers and strings)\n        import re\n        normalized = re.sub(r'\\b\\d+\\b', '?', normalized)\n        normalized = re.sub(r\"'[^']*'\", \"'?'\", normalized)\n        return hashlib.sha256(normalized.encode()).hexdigest()[:16]\n    \n    def record_query_execution(\n        self,\n        query: str,\n        execution_time: float,\n        rows_affected: int = 0,\n        cached: bool = False,\n        error: bool = False\n    ):\n        \"\"\"Record query execution statistics.\"\"\"\n        query_hash = self._generate_query_hash(query)\n        query_type = self._classify_query_type(query)\n        \n        if query_hash not in self.query_stats:\n            self.query_stats[query_hash] = QueryStats(\n                query_hash=query_hash,\n                query_text=query,\n                query_type=query_type\n            )\n        \n        self.query_stats[query_hash].update(execution_time, rows_affected, cached, error)\n        \n        # Generate optimization suggestions for slow queries\n        if execution_time > self.slow_query_threshold and not error:\n            self._analyze_slow_query(query_hash, query, execution_time)\n    \n    def _analyze_slow_query(self, query_hash: str, query: str, execution_time: float):\n        \"\"\"Analyze slow queries and generate optimization suggestions.\"\"\"\n        suggestions = []\n        query_lower = query.lower()\n        \n        # Check for missing WHERE clauses in SELECT/UPDATE/DELETE\n        if ('select' in query_lower or 'update' in query_lower or 'delete' in query_lower):\n            if 'where' not in query_lower:\n                suggestions.append(\"Consider adding WHERE clause to limit result set\")\n        \n        # Check for missing LIMIT in SELECT queries\n        if query_lower.startswith('select') and 'limit' not in query_lower:\n            suggestions.append(\"Consider adding LIMIT clause for large result sets\")\n        \n        # Check for SELECT *\n        if 'select *' in query_lower:\n            suggestions.append(\"Avoid SELECT * - specify only needed columns\")\n        \n        # Check for subqueries that could be JOINs\n        if query_lower.count('select') > 1:\n            suggestions.append(\"Consider using JOINs instead of subqueries for better performance\")\n        \n        # Check for functions in WHERE clauses\n        where_part = query_lower.split('where')[1] if 'where' in query_lower else \"\"\n        if any(func in where_part for func in ['upper(', 'lower(', 'substr(', 'date(']):\n            suggestions.append(\"Avoid functions in WHERE clauses - consider functional indexes\")\n        \n        # Check for OR conditions\n        if ' or ' in query_lower:\n            suggestions.append(\"Consider rewriting OR conditions as UNION for better index usage\")\n        \n        if suggestions:\n            self.optimization_suggestions[query_hash] = suggestions\n            logger.info(f\"Generated {len(suggestions)} optimization suggestions for slow query (${execution_time:.3f}s)\")\n    \n    def get_slow_queries(self, threshold: float = None) -> List[Dict[str, Any]]:\n        \"\"\"Get list of slow queries above threshold.\"\"\"\n        threshold = threshold or self.slow_query_threshold\n        \n        slow_queries = []\n        for stats in self.query_stats.values():\n            if stats.avg_time > threshold and stats.execution_count > 0:\n                query_dict = stats.to_dict()\n                query_dict[\"suggestions\"] = self.optimization_suggestions.get(stats.query_hash, [])\n                slow_queries.append(query_dict)\n        \n        # Sort by average execution time\n        return sorted(slow_queries, key=lambda q: q[\"avg_time\"], reverse=True)\n    \n    def get_query_analysis(self) -> Dict[str, Any]:\n        \"\"\"Get comprehensive query analysis.\"\"\"\n        if not self.query_stats:\n            return {\"error\": \"No query statistics available\"}\n        \n        # Aggregate statistics by query type\n        type_stats = defaultdict(lambda: {\n            \"count\": 0, \"total_time\": 0, \"avg_time\": 0, \"error_count\": 0\n        })\n        \n        total_queries = 0\n        total_errors = 0\n        total_time = 0\n        \n        for stats in self.query_stats.values():\n            total_queries += stats.execution_count\n            total_errors += stats.error_count\n            total_time += stats.total_time\n            \n            type_stats[stats.query_type.value][\"count\"] += stats.execution_count\n            type_stats[stats.query_type.value][\"total_time\"] += stats.total_time\n            type_stats[stats.query_type.value][\"error_count\"] += stats.error_count\n        \n        # Calculate averages\n        for type_name, stats in type_stats.items():\n            if stats[\"count\"] > 0:\n                stats[\"avg_time\"] = stats[\"total_time\"] / stats[\"count\"]\n                stats[\"success_rate\"] = (stats[\"count\"] - stats[\"error_count\"]) / stats[\"count\"]\n        \n        return {\n            \"overall_stats\": {\n                \"total_queries\": total_queries,\n                \"total_errors\": total_errors,\n                \"overall_success_rate\": (total_queries - total_errors) / total_queries if total_queries > 0 else 0,\n                \"avg_query_time\": total_time / total_queries if total_queries > 0 else 0,\n                \"unique_queries\": len(self.query_stats)\n            },\n            \"query_types\": dict(type_stats),\n            \"slow_queries_count\": len(self.get_slow_queries()),\n            \"optimization_suggestions_count\": len(self.optimization_suggestions),\n            \"top_slow_queries\": self.get_slow_queries()[:10]\n        }\n\n\nclass DatabaseManager:\n    \"\"\"Main database management system with optimization and monitoring.\"\"\"\n    \n    def __init__(self):\n        self.connection_pools: Dict[str, DatabaseConnectionPool] = {}\n        self.query_cache = QueryCache()\n        self.query_optimizer = QueryOptimizer()\n        self.default_pool_name = \"default\"\n        \n        # Configuration\n        self.enable_query_caching = True\n        self.enable_query_logging = True\n        self.enable_optimization_analysis = True\n        \n    async def add_database_pool(\n        self,\n        pool_name: str,\n        database_type: DatabaseType,\n        connection_string: str,\n        min_size: int = 5,\n        max_size: int = 20,\n        **kwargs\n    ):\n        \"\"\"Add a new database connection pool.\"\"\"\n        pool = DatabaseConnectionPool(\n            database_type=database_type,\n            connection_string=connection_string,\n            min_size=min_size,\n            max_size=max_size,\n            pool_name=pool_name,\n            **kwargs\n        )\n        \n        await pool.initialize()\n        self.connection_pools[pool_name] = pool\n        \n        logger.info(f\"Added database pool '{pool_name}' of type {database_type.value}\")\n        \n        # Set as default if it's the first pool\n        if len(self.connection_pools) == 1:\n            self.default_pool_name = pool_name\n    \n    @asynccontextmanager\n    async def get_connection(self, pool_name: str = None):\n        \"\"\"Get a database connection from the specified pool.\"\"\"\n        pool_name = pool_name or self.default_pool_name\n        \n        if pool_name not in self.connection_pools:\n            raise ValueError(f\"Database pool '{pool_name}' not found\")\n        \n        pool = self.connection_pools[pool_name]\n        async with pool.acquire_connection() as conn:\n            yield conn\n    \n    async def execute_query(\n        self,\n        query: str,\n        params: tuple = None,\n        pool_name: str = None,\n        cache_result: bool = None,\n        cache_ttl: int = None\n    ) -> Any:\n        \"\"\"Execute a database query with optimization and caching.\"\"\"\n        pool_name = pool_name or self.default_pool_name\n        cache_result = cache_result if cache_result is not None else self.enable_query_caching\n        \n        start_time = time.time()\n        result = None\n        cached = False\n        error = False\n        rows_affected = 0\n        \n        try:\n            # Try to get from cache first (for SELECT queries)\n            if cache_result and query.strip().lower().startswith('select'):\n                cached_result = await self.query_cache.get(query, params)\n                if cached_result is not None:\n                    cached = True\n                    result = cached_result\n                    logger.debug(f\"Query served from cache: {query[:50]}...\")\n            \n            if result is None:\n                # Execute query\n                pool = self.connection_pools[pool_name]\n                async with pool.acquire_connection() as conn:\n                    if pool.database_type == DatabaseType.POSTGRESQL:\n                        if params:\n                            result = await conn.fetch(query, *params)\n                        else:\n                            result = await conn.fetch(query)\n                        rows_affected = len(result) if result else 0\n                        \n                    elif pool.database_type == DatabaseType.MYSQL:\n                        async with conn.cursor() as cursor:\n                            if params:\n                                await cursor.execute(query, params)\n                            else:\n                                await cursor.execute(query)\n                            \n                            if query.strip().lower().startswith('select'):\n                                result = await cursor.fetchall()\n                                rows_affected = len(result) if result else 0\n                            else:\n                                rows_affected = cursor.rowcount\n                                result = rows_affected\n                    \n                    elif pool.database_type == DatabaseType.SQLITE:\n                        cursor = conn.cursor()\n                        if params:\n                            cursor.execute(query, params)\n                        else:\n                            cursor.execute(query)\n                        \n                        if query.strip().lower().startswith('select'):\n                            result = cursor.fetchall()\n                            rows_affected = len(result) if result else 0\n                        else:\n                            rows_affected = cursor.rowcount\n                            result = rows_affected\n                            conn.commit()\n                \n                # Cache the result if applicable\n                if (cache_result and result is not None and \n                    query.strip().lower().startswith('select')):\n                    await self.query_cache.set(query, result, params, cache_ttl)\n        \n        except Exception as e:\n            error = True\n            logger.error(f\"Query execution error: {e}\")\n            raise\n        \n        finally:\n            execution_time = time.time() - start_time\n            \n            # Record query statistics\n            if self.enable_optimization_analysis:\n                self.query_optimizer.record_query_execution(\n                    query, execution_time, rows_affected, cached, error\n                )\n            \n            # Log slow queries\n            if self.enable_query_logging and execution_time > 1.0:  # Log queries > 1 second\n                logger.warning(\n                    f\"Slow query detected ({execution_time:.3f}s): {query[:100]}...\"\n                )\n        \n        return result\n    \n    async def execute_transaction(\n        self,\n        queries: List[Tuple[str, tuple]],\n        pool_name: str = None\n    ) -> List[Any]:\n        \"\"\"Execute multiple queries in a transaction.\"\"\"\n        pool_name = pool_name or self.default_pool_name\n        pool = self.connection_pools[pool_name]\n        \n        results = []\n        start_time = time.time()\n        \n        try:\n            async with pool.acquire_connection() as conn:\n                if pool.database_type == DatabaseType.POSTGRESQL:\n                    async with conn.transaction():\n                        for query, params in queries:\n                            if params:\n                                result = await conn.fetch(query, *params)\n                            else:\n                                result = await conn.fetch(query)\n                            results.append(result)\n                \n                elif pool.database_type == DatabaseType.MYSQL:\n                    await conn.begin()\n                    try:\n                        for query, params in queries:\n                            async with conn.cursor() as cursor:\n                                if params:\n                                    await cursor.execute(query, params)\n                                else:\n                                    await cursor.execute(query)\n                                \n                                if query.strip().lower().startswith('select'):\n                                    result = await cursor.fetchall()\n                                else:\n                                    result = cursor.rowcount\n                                results.append(result)\n                        \n                        await conn.commit()\n                    except Exception:\n                        await conn.rollback()\n                        raise\n                \n                elif pool.database_type == DatabaseType.SQLITE:\n                    try:\n                        for query, params in queries:\n                            cursor = conn.cursor()\n                            if params:\n                                cursor.execute(query, params)\n                            else:\n                                cursor.execute(query)\n                            \n                            if query.strip().lower().startswith('select'):\n                                result = cursor.fetchall()\n                            else:\n                                result = cursor.rowcount\n                            results.append(result)\n                        \n                        conn.commit()\n                    except Exception:\n                        conn.rollback()\n                        raise\n        \n        except Exception as e:\n            logger.error(f\"Transaction execution error: {e}\")\n            raise\n        \n        finally:\n            execution_time = time.time() - start_time\n            \n            # Record transaction statistics\n            if self.enable_optimization_analysis:\n                transaction_query = f\"TRANSACTION({len(queries)} queries)\"\n                self.query_optimizer.record_query_execution(\n                    transaction_query, execution_time, len(results), False, False\n                )\n        \n        return results\n    \n    async def invalidate_cache(self, pattern: str = None):\n        \"\"\"Invalidate query cache entries.\"\"\"\n        if pattern:\n            return await self.query_cache.invalidate_pattern(pattern)\n        else:\n            # Clear entire cache\n            self.query_cache.cache.clear()\n            self.query_cache.access_times.clear()\n            return len(self.query_cache.cache)\n    \n    def get_comprehensive_stats(self) -> Dict[str, Any]:\n        \"\"\"Get comprehensive database performance statistics.\"\"\"\n        return {\n            \"connection_pools\": {\n                name: pool.get_stats() for name, pool in self.connection_pools.items()\n            },\n            \"query_cache\": self.query_cache.get_stats(),\n            \"query_analysis\": self.query_optimizer.get_query_analysis(),\n            \"configuration\": {\n                \"enable_query_caching\": self.enable_query_caching,\n                \"enable_query_logging\": self.enable_query_logging,\n                \"enable_optimization_analysis\": self.enable_optimization_analysis,\n                \"slow_query_threshold\": self.query_optimizer.slow_query_threshold\n            }\n        }\n    \n    async def get_optimization_report(self) -> Dict[str, Any]:\n        \"\"\"Get detailed optimization recommendations.\"\"\"\n        slow_queries = self.query_optimizer.get_slow_queries()\n        \n        recommendations = []\n        \n        # General recommendations based on statistics\n        cache_stats = self.query_cache.get_stats()\n        if cache_stats[\"hit_rate\"] < 0.3:\n            recommendations.append({\n                \"type\": \"caching\",\n                \"priority\": \"high\",\n                \"title\": \"Low cache hit rate\",\n                \"description\": f\"Query cache hit rate is {cache_stats['hit_rate']:.1%}. Consider increasing cache size or TTL.\",\n                \"action\": \"Review caching strategy and cache configuration\"\n            })\n        \n        # Connection pool recommendations\n        for name, pool in self.connection_pools.items():\n            stats = pool.get_stats()\n            if stats[\"utilization\"] > 0.8:\n                recommendations.append({\n                    \"type\": \"connection_pool\",\n                    \"priority\": \"medium\",\n                    \"title\": f\"High connection pool utilization in {name}\",\n                    \"description\": f\"Pool utilization is {stats['utilization']:.1%}. Consider increasing pool size.\",\n                    \"action\": \"Increase max_size for connection pool\"\n                })\n            \n            if stats[\"avg_wait_time\"] > 0.1:\n                recommendations.append({\n                    \"type\": \"connection_pool\",\n                    \"priority\": \"medium\",\n                    \"title\": f\"High connection wait time in {name}\",\n                    \"description\": f\"Average wait time is {stats['avg_wait_time']:.3f}s.\",\n                    \"action\": \"Consider increasing connection pool size or optimizing query performance\"\n                })\n        \n        return {\n            \"slow_queries\": slow_queries[:20],  # Top 20 slow queries\n            \"recommendations\": recommendations,\n            \"summary\": {\n                \"total_slow_queries\": len(slow_queries),\n                \"total_recommendations\": len(recommendations),\n                \"critical_issues\": len([r for r in recommendations if r[\"priority\"] == \"high\"])\n            }\n        }\n    \n    async def shutdown(self):\n        \"\"\"Shutdown all database connections and cleanup.\"\"\"\n        logger.info(\"Shutting down database manager...\")\n        \n        for pool in self.connection_pools.values():\n            await pool.shutdown()\n        \n        self.connection_pools.clear()\n        logger.info(\"Database manager shutdown complete\")\n\n\n# Global database manager instance\ndatabase_manager = DatabaseManager()\n\n\n# Convenience decorators and utilities\ndef db_query_cache(ttl: int = 300):\n    \"\"\"Decorator to cache database query results.\"\"\"\n    def decorator(func: Callable) -> Callable:\n        @wraps(func)\n        async def wrapper(*args, **kwargs):\n            # Generate cache key from function name and arguments\n            cache_key = f\"{func.__name__}:{hash(str(args) + str(kwargs))}\"\n            \n            # Try to get from cache\n            cached_result = await database_manager.query_cache.get(cache_key)\n            if cached_result is not None:\n                return cached_result\n            \n            # Execute function\n            result = await func(*args, **kwargs)\n            \n            # Cache result\n            await database_manager.query_cache.set(cache_key, result, ttl=ttl)\n            \n            return result\n        \n        return wrapper\n    return decorator"