"""Tests for performance optimization functionality."""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch
from collections import deque

from self_healing_bot.performance.caching import AdaptiveCache, cache_result, BotCacheUtils, CacheWarmer
from self_healing_bot.performance.concurrency import (
    AdaptiveExecutor, ResourceMonitor, ConcurrentBatchProcessor, CircuitBreaker
)
from self_healing_bot.performance.auto_scaling import AutoScaler, ScalingPolicy, ScalingEvent


class TestAdaptiveCache:
    """Test cases for AdaptiveCache."""
    
    @pytest.mark.asyncio
    async def test_cache_initialization(self):
        """Test cache initialization."""
        cache = AdaptiveCache()
        await cache.initialize()
        
        # Should initialize without errors
        assert cache is not None
        assert cache.local_cache == {}
        assert cache.max_local_size == 1000
    
    @pytest.mark.asyncio
    async def test_cache_set_get(self):
        """Test basic cache set and get operations."""
        cache = AdaptiveCache()
        await cache.initialize()
        
        # Set a value
        await cache.set("test_key", "test_value", ttl=3600)
        
        # Get the value
        result = await cache.get("test_key")
        assert result == "test_value"
    
    @pytest.mark.asyncio
    async def test_cache_expiration(self):
        """Test cache expiration."""
        cache = AdaptiveCache()
        await cache.initialize()
        
        # Set a value with very short TTL
        await cache.set("short_ttl_key", "test_value", ttl=1)
        
        # Should be available immediately
        result = await cache.get("short_ttl_key")
        assert result == "test_value"
        
        # Wait for expiration
        await asyncio.sleep(1.1)
        
        # Should be expired
        result = await cache.get("short_ttl_key")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_cache_miss(self):
        """Test cache miss."""
        cache = AdaptiveCache()
        await cache.initialize()
        
        result = await cache.get("nonexistent_key")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_cache_invalidation(self):
        """Test cache invalidation."""
        cache = AdaptiveCache()
        await cache.initialize()
        
        # Set a value
        await cache.set("test_key", "test_value")
        
        # Verify it exists
        result = await cache.get("test_key")
        assert result == "test_value"
        
        # Invalidate
        await cache.invalidate("test_key")
        
        # Should be gone
        result = await cache.get("test_key")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_cache_pattern_invalidation(self):
        """Test pattern-based cache invalidation."""
        cache = AdaptiveCache()
        await cache.initialize()
        
        # Set multiple values with similar keys
        await cache.set("user:123:profile", "profile_data")
        await cache.set("user:123:settings", "settings_data")
        await cache.set("user:456:profile", "other_profile")
        
        # Invalidate pattern
        await cache.invalidate_pattern("*user:123*")
        
        # User 123 data should be gone
        assert await cache.get("user:123:profile") is None
        assert await cache.get("user:123:settings") is None
        
        # User 456 data should remain
        assert await cache.get("user:456:profile") == "other_profile"
    
    def test_adaptive_ttl_calculation(self):
        """Test adaptive TTL calculation."""
        cache = AdaptiveCache()
        
        # Test with no access history
        ttl = cache._calculate_adaptive_ttl("new_key")
        assert ttl == 3600  # Default
        
        # Test with high frequency access
        key = "high_freq_key"
        current_time = time.time()
        cache.access_patterns[key] = [current_time - i * 60 for i in range(15)]  # 15 accesses
        
        ttl = cache._calculate_adaptive_ttl(key)
        assert ttl == 7200  # High frequency = longer TTL
        
        # Test with low frequency access
        key = "low_freq_key"
        cache.access_patterns[key] = [current_time - i * 3600 for i in range(3)]  # 3 accesses over hours
        
        ttl = cache._calculate_adaptive_ttl(key)
        assert ttl == 1800  # Low frequency = shorter TTL
    
    def test_cache_stats(self):
        """Test cache statistics tracking."""
        cache = AdaptiveCache()
        
        # Simulate hits and misses
        cache._record_hit("test_namespace", time.time())
        cache._record_hit("test_namespace", time.time())
        cache._record_miss("test_namespace", time.time())
        
        stats = cache.get_stats()
        
        assert "test_namespace" in stats
        namespace_stats = stats["test_namespace"]
        assert namespace_stats["hits"] == 2
        assert namespace_stats["misses"] == 1
        assert abs(namespace_stats["hit_rate"] - 0.666) < 0.01  # 2/3
    
    def test_local_cache_lru_eviction(self):
        """Test LRU eviction in local cache."""
        cache = AdaptiveCache()
        cache.max_local_size = 2  # Very small for testing
        
        # Add entries beyond capacity
        cache._store_local("key1", {"value": "value1", "created_at": 1, "ttl": 3600})
        cache._store_local("key2", {"value": "value2", "created_at": 2, "ttl": 3600})
        cache._store_local("key3", {"value": "value3", "created_at": 3, "ttl": 3600})
        
        # Should have evicted oldest entry
        assert len(cache.local_cache) == 2
        assert "key1" not in cache.local_cache  # Oldest, should be evicted
        assert "key2" in cache.local_cache
        assert "key3" in cache.local_cache


class TestCacheDecorator:
    """Test cases for cache_result decorator."""
    
    @pytest.mark.asyncio
    async def test_cache_decorator_async_function(self):
        """Test cache decorator with async function."""
        call_count = 0
        
        @cache_result(ttl=3600, namespace="test")
        async def expensive_async_function(x, y):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)  # Simulate work
            return x + y
        
        # First call should execute function
        result1 = await expensive_async_function(1, 2)
        assert result1 == 3
        assert call_count == 1
        
        # Second call should use cache
        result2 = await expensive_async_function(1, 2)
        assert result2 == 3
        assert call_count == 1  # Function not called again
        
        # Different arguments should execute function
        result3 = await expensive_async_function(2, 3)
        assert result3 == 5
        assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_cache_decorator_sync_function(self):
        """Test cache decorator with sync function."""
        call_count = 0
        
        @cache_result(ttl=3600, namespace="test")
        def expensive_sync_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2
        
        # First call should execute function
        result1 = await expensive_sync_function(5)
        assert result1 == 10
        assert call_count == 1
        
        # Second call should use cache
        result2 = await expensive_sync_function(5)
        assert result2 == 10
        assert call_count == 1  # Function not called again


class TestCacheWarmer:
    """Test cases for CacheWarmer."""
    
    @pytest.mark.asyncio
    async def test_cache_warmer_initialization(self):
        """Test cache warmer initialization."""
        cache = AdaptiveCache()
        warmer = CacheWarmer(cache)
        
        assert warmer.cache is cache
        assert warmer.warming_tasks == {}
    
    @pytest.mark.asyncio
    async def test_warm_popular_keys(self):
        """Test warming of popular keys."""
        cache = AdaptiveCache()
        warmer = CacheWarmer(cache)
        
        # Setup cache with popular key about to expire
        key = "bot:test:popular_key"
        current_time = time.time()
        
        # Add to local cache with short remaining TTL
        cache.local_cache[key] = {
            "value": "test_value",
            "created_at": current_time - 3000,  # Created 50 minutes ago
            "ttl": 3600,  # 1 hour TTL, so 10 minutes remaining
            "access_count": 0
        }
        
        # Add access pattern showing high frequency
        cache.access_patterns[key] = [current_time - i * 60 for i in range(15)]
        
        # Run warming (should log that it would warm the key)
        await warmer._warm_popular_keys()
        
        # Verify no errors occurred
        assert True  # Test passes if no exceptions


class TestAdaptiveExecutor:
    """Test cases for AdaptiveExecutor."""
    
    @pytest.mark.asyncio
    async def test_executor_initialization(self):
        """Test executor initialization."""
        executor = AdaptiveExecutor(max_workers=8)
        
        assert executor.max_workers == 8
        assert executor.current_workers <= executor.max_workers
        assert executor.thread_executor is not None
        assert executor.process_executor is not None
    
    @pytest.mark.asyncio
    async def test_submit_io_task_success(self):
        """Test successful I/O task submission."""
        executor = AdaptiveExecutor()
        
        def simple_io_task(x):
            return x * 2
        
        result = await executor.submit_io_task(simple_io_task, 5, task_type="test_io")
        
        assert result == 10
        assert "test_io" in executor.task_metrics
        assert executor.task_metrics["test_io"].completed_tasks == 1
        assert executor.task_metrics["test_io"].failed_tasks == 0
    
    @pytest.mark.asyncio
    async def test_submit_io_task_failure(self):
        """Test I/O task submission with failure."""
        executor = AdaptiveExecutor()
        
        def failing_task():
            raise ValueError("Task failed")
        
        with pytest.raises(ValueError):
            await executor.submit_io_task(failing_task, task_type="test_fail")
        
        assert executor.task_metrics["test_fail"].completed_tasks == 0
        assert executor.task_metrics["test_fail"].failed_tasks == 1
    
    @pytest.mark.asyncio
    async def test_submit_cpu_task_success(self):
        """Test successful CPU task submission."""
        executor = AdaptiveExecutor()
        
        def cpu_intensive_task(n):
            return sum(i * i for i in range(n))
        
        result = await executor.submit_cpu_task(cpu_intensive_task, 100, task_type="test_cpu")
        
        assert result == sum(i * i for i in range(100))
        assert executor.task_metrics["test_cpu"].completed_tasks == 1
    
    def test_get_executor_metrics(self):
        """Test executor metrics retrieval."""
        executor = AdaptiveExecutor()
        
        # Add some mock metrics
        executor.task_metrics["test"].completed_tasks = 5
        executor.task_metrics["test"].failed_tasks = 1
        executor.task_metrics["test"].total_tasks = 6
        executor.task_metrics["test"].total_execution_time = 10.0
        executor.task_metrics["test"].avg_execution_time = 10.0 / 6
        
        metrics = executor.get_metrics()
        
        assert metrics["current_workers"] == executor.current_workers
        assert metrics["max_workers"] == executor.max_workers
        assert "task_metrics" in metrics
        assert "test" in metrics["task_metrics"]
        
        test_metrics = metrics["task_metrics"]["test"]
        assert test_metrics["completed_tasks"] == 5
        assert test_metrics["failed_tasks"] == 1
        assert test_metrics["success_rate"] == 5/6


class TestResourceMonitor:
    """Test cases for ResourceMonitor."""
    
    @pytest.mark.asyncio
    async def test_resource_monitor_initialization(self):
        """Test resource monitor initialization."""
        monitor = ResourceMonitor()
        
        assert monitor.history_size == 60
        assert len(monitor.cpu_history) == 0
        assert len(monitor.memory_history) == 0
        assert not monitor.running
    
    @pytest.mark.asyncio
    async def test_get_current_usage(self):
        """Test current resource usage retrieval."""
        monitor = ResourceMonitor()
        
        usage = await monitor.get_current_usage()
        
        assert "cpu_percent" in usage
        assert "memory_percent" in usage
        assert "memory_available_mb" in usage
        assert "memory_used_mb" in usage
        
        # Values should be reasonable
        assert 0 <= usage["cpu_percent"] <= 100
        assert 0 <= usage["memory_percent"] <= 100
        assert usage["memory_available_mb"] >= 0
        assert usage["memory_used_mb"] >= 0
    
    def test_get_average_usage_empty_history(self):
        """Test average usage calculation with empty history."""
        monitor = ResourceMonitor()
        
        average = monitor.get_average_usage()
        
        assert average["cpu_percent"] == 0.0
        assert average["memory_percent"] == 0.0
    
    def test_get_average_usage_with_history(self):
        """Test average usage calculation with history."""
        monitor = ResourceMonitor()
        
        # Add some mock history
        monitor.cpu_history.extend([10.0, 20.0, 30.0, 40.0])
        monitor.memory_history.extend([15.0, 25.0, 35.0, 45.0])
        
        average = monitor.get_average_usage(window_minutes=1)
        
        assert average["cpu_percent"] == 25.0  # Average of 10, 20, 30, 40
        assert average["memory_percent"] == 30.0  # Average of 15, 25, 35, 45


class TestConcurrentBatchProcessor:
    """Test cases for ConcurrentBatchProcessor."""
    
    @pytest.mark.asyncio
    async def test_batch_processor_initialization(self):
        """Test batch processor initialization."""
        executor = AdaptiveExecutor()
        processor = ConcurrentBatchProcessor(executor, batch_size=5)
        
        assert processor.executor is executor
        assert processor.batch_size == 5
        assert processor.pending_batches == {}
    
    @pytest.mark.asyncio
    async def test_add_task_to_batch(self):
        """Test adding tasks to batch."""
        executor = AdaptiveExecutor()
        processor = ConcurrentBatchProcessor(executor, batch_size=3)
        
        def simple_process(data):
            return data["value"] * 2
        
        # Add tasks to batch (should not process yet)
        await processor.add_task("test_batch", {"value": 1}, simple_process)
        await processor.add_task("test_batch", {"value": 2}, simple_process)
        
        # Should have pending tasks
        assert len(processor.pending_batches["test_batch"]) == 2
    
    @pytest.mark.asyncio
    async def test_batch_processing_when_full(self):
        """Test batch processing when batch size is reached."""
        executor = AdaptiveExecutor()
        processor = ConcurrentBatchProcessor(executor, batch_size=2)
        
        results = []
        
        def simple_process(data):
            result = data["value"] * 2
            results.append(result)
            return result
        
        # Add tasks to fill batch
        await processor.add_task("test_batch", {"value": 1}, simple_process)
        await processor.add_task("test_batch", {"value": 2}, simple_process)
        
        # Give time for processing
        await asyncio.sleep(0.1)
        
        # Batch should have been processed
        assert len(processor.pending_batches["test_batch"]) == 0
        assert len(results) == 2
        assert 2 in results  # 1 * 2
        assert 4 in results  # 2 * 2
    
    @pytest.mark.asyncio
    async def test_flush_all_batches(self):
        """Test flushing all pending batches."""
        executor = AdaptiveExecutor()
        processor = ConcurrentBatchProcessor(executor, batch_size=10)  # Large batch size
        
        results = []
        
        def simple_process(data):
            result = data["value"] * 2
            results.append(result)
            return result
        
        # Add tasks that won't trigger automatic processing
        await processor.add_task("batch1", {"value": 1}, simple_process)
        await processor.add_task("batch1", {"value": 2}, simple_process)
        await processor.add_task("batch2", {"value": 3}, simple_process)
        
        # Flush all batches
        await processor.flush_all_batches()
        
        # All tasks should have been processed
        assert len(results) == 3
        assert 2 in results
        assert 4 in results
        assert 6 in results


class TestCircuitBreaker:
    """Test cases for CircuitBreaker."""
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_initialization(self):
        """Test circuit breaker initialization."""
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30)
        
        assert breaker.failure_threshold == 3
        assert breaker.recovery_timeout == 30
        assert breaker.failure_count == 0
        assert breaker.state == "closed"
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_success_calls(self):
        """Test successful calls through circuit breaker."""
        breaker = CircuitBreaker()
        
        def successful_function():
            return "success"
        
        result = await breaker.call(successful_function)
        
        assert result == "success"
        assert breaker.failure_count == 0
        assert breaker.state == "closed"
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_opening(self):
        """Test circuit breaker opening after failures."""
        breaker = CircuitBreaker(failure_threshold=2)
        
        def failing_function():
            raise ValueError("Function failed")
        
        # First failure
        with pytest.raises(ValueError):
            await breaker.call(failing_function)
        assert breaker.failure_count == 1
        assert breaker.state == "closed"
        
        # Second failure - should open circuit
        with pytest.raises(ValueError):
            await breaker.call(failing_function)
        assert breaker.failure_count == 2
        assert breaker.state == "open"
        
        # Next call should be rejected
        with pytest.raises(Exception) as exc_info:
            await breaker.call(failing_function)
        assert "Circuit breaker is open" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery after timeout."""
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)
        
        def failing_then_succeeding():
            if breaker.failure_count == 0:
                return "success"
            raise ValueError("Function failed")
        
        # Cause circuit to open
        with pytest.raises(ValueError):
            await breaker.call(lambda: exec('raise ValueError("Function failed")'))
        assert breaker.state == "open"
        
        # Wait for recovery timeout
        await asyncio.sleep(0.2)
        
        # Should attempt half-open
        result = await breaker.call(failing_then_succeeding)
        assert result == "success"
        assert breaker.state == "closed"
        assert breaker.failure_count == 0
    
    def test_circuit_breaker_state(self):
        """Test circuit breaker state reporting."""
        breaker = CircuitBreaker()
        breaker.failure_count = 3
        breaker.last_failure_time = time.time()
        
        state = breaker.get_state()
        
        assert state["state"] == "closed"  # Not opened yet
        assert state["failure_count"] == 3
        assert state["last_failure_time"] is not None
        assert state["time_since_last_failure"] is not None


class TestAutoScaler:
    """Test cases for AutoScaler."""
    
    def test_auto_scaler_initialization(self):
        """Test auto scaler initialization."""
        scaler = AutoScaler()
        
        assert scaler.scaling_enabled is True
        assert len(scaler.policies) > 0  # Should have default policies
        assert "worker_utilization" in scaler.policies
        assert "queue_length" in scaler.policies
    
    def test_add_scaling_policy(self):
        """Test adding scaling policy."""
        scaler = AutoScaler()
        
        policy = ScalingPolicy(
            metric_name="custom_metric",
            scale_up_threshold=80.0,
            scale_down_threshold=20.0,
            min_capacity=1,
            max_capacity=10
        )
        
        scaler.add_policy(policy)
        
        assert "custom_metric" in scaler.policies
        assert scaler.current_capacity["custom_metric"] == 1  # min_capacity
        assert "custom_metric" in scaler.scaling_history
    
    @pytest.mark.asyncio
    async def test_record_metric_below_threshold(self):
        """Test recording metric below scaling threshold."""
        scaler = AutoScaler()
        scaler.scaling_enabled = True
        
        # Record low utilization (should not trigger scaling)
        await scaler.record_metric("worker_utilization", 50.0)
        
        # Should have recorded the metric
        assert len(scaler.metrics_history["worker_utilization"]) == 1
    
    @pytest.mark.asyncio
    async def test_scaling_disabled(self):
        """Test behavior when scaling is disabled."""
        scaler = AutoScaler()
        scaler.disable_scaling()
        
        await scaler.record_metric("worker_utilization", 95.0)  # High utilization
        
        # Should record metric but not scale
        assert len(scaler.metrics_history["worker_utilization"]) == 1
        # Capacity should remain at minimum
        assert scaler.current_capacity["worker_utilization"] == 2  # min_capacity
    
    @pytest.mark.asyncio
    async def test_force_scaling(self):
        """Test forced scaling to specific capacity."""
        scaler = AutoScaler()
        
        success = await scaler.force_scale("worker_utilization", 5)
        
        assert success is True
        assert scaler.current_capacity["worker_utilization"] == 5
        
        # Should have created scaling event
        events = list(scaler.scaling_history["worker_utilization"])
        assert len(events) == 1
        assert events[0].action.startswith("manual_")
    
    @pytest.mark.asyncio
    async def test_force_scaling_invalid_capacity(self):
        """Test forced scaling with invalid capacity."""
        scaler = AutoScaler()
        
        policy = scaler.policies["worker_utilization"]
        
        # Try to scale beyond max capacity
        success = await scaler.force_scale("worker_utilization", policy.max_capacity + 10)
        assert success is False
        
        # Try to scale below min capacity
        success = await scaler.force_scale("worker_utilization", policy.min_capacity - 1)
        assert success is False
    
    def test_get_scaling_status(self):
        """Test scaling status retrieval."""
        scaler = AutoScaler()
        
        status = scaler.get_scaling_status()
        
        assert "scaling_enabled" in status
        assert "policies" in status
        assert "current_capacity" in status
        assert "recent_events" in status
        
        # Should have default policies
        assert "worker_utilization" in status["policies"]
        assert "queue_length" in status["policies"]
    
    def test_enable_disable_scaling(self):
        """Test enabling and disabling scaling."""
        scaler = AutoScaler()
        
        # Test disabling
        scaler.disable_scaling()
        assert scaler.scaling_enabled is False
        
        # Test enabling
        scaler.enable_scaling()
        assert scaler.scaling_enabled is True
    
    def test_get_recent_metrics(self):
        """Test recent metrics retrieval."""
        scaler = AutoScaler()
        
        # Add some metrics
        current_time = time.time()
        scaler.metrics_history["test_metric"] = deque([
            (current_time - 60, 10.0),
            (current_time - 30, 20.0),
            (current_time, 30.0)
        ])
        
        recent = scaler._get_recent_metrics("test_metric", 2)
        
        assert len(recent) == 2
        assert recent == [20.0, 30.0]  # Last 2 values