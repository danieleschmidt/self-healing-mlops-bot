"""Enhanced bulkhead isolation manager with resource pool management and dynamic sizing."""

import asyncio
import time
import logging
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
from statistics import mean
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class PoolPriority(Enum):
    """Priority levels for resource pools."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class BulkheadConfig:
    """Configuration for bulkhead isolation."""
    name: str
    max_concurrent: int = 10
    queue_size: int = 50
    timeout: float = 30.0
    
    # Dynamic sizing
    enable_auto_scaling: bool = False
    min_size: int = 1
    max_size: int = 100
    scale_up_threshold: float = 0.8  # Scale up when 80% utilized
    scale_down_threshold: float = 0.2  # Scale down when 20% utilized
    
    # Priority queuing
    enable_priority_queue: bool = False
    priority_queue_sizes: Dict[PoolPriority, int] = None
    
    # Monitoring and metrics
    metrics_enabled: bool = True
    slow_request_threshold: float = 5.0  # Log slow requests
    
    def __post_init__(self):
        if self.priority_queue_sizes is None:
            self.priority_queue_sizes = {
                PoolPriority.CRITICAL: max(2, self.queue_size // 4),
                PoolPriority.HIGH: max(5, self.queue_size // 3),
                PoolPriority.NORMAL: self.queue_size // 2,
                PoolPriority.LOW: self.queue_size // 4
            }


@dataclass
class PoolMetrics:
    """Metrics for a resource pool."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    timeout_requests: int = 0
    rejected_requests: int = 0
    
    # Timing metrics
    total_execution_time: float = 0.0
    avg_execution_time: float = 0.0
    max_execution_time: float = 0.0
    min_execution_time: float = float('inf')
    
    # Queue metrics
    max_queue_size: int = 0
    total_queue_time: float = 0.0
    avg_queue_time: float = 0.0
    
    # Resource utilization
    peak_concurrent: int = 0
    avg_utilization: float = 0.0
    
    last_updated: float = 0.0


class PriorityQueue:
    """Priority queue for bulkhead requests."""
    
    def __init__(self, config: BulkheadConfig):
        self.config = config
        self.queues = {
            priority: asyncio.Queue(maxsize=size)
            for priority, size in config.priority_queue_sizes.items()
        }
        self.total_size = 0
        self.max_size = sum(config.priority_queue_sizes.values())
    
    async def put(self, item: Any, priority: PoolPriority = PoolPriority.NORMAL):
        """Put item in priority queue."""
        if self.total_size >= self.max_size:
            raise BulkheadFullError("Priority queue is full")
        
        queue = self.queues.get(priority, self.queues[PoolPriority.NORMAL])
        
        try:
            queue.put_nowait(item)
            self.total_size += 1
        except asyncio.QueueFull:
            # Try lower priority queues if current is full
            for lower_priority in [PoolPriority.LOW, PoolPriority.NORMAL, PoolPriority.HIGH]:
                if lower_priority != priority:
                    try:
                        self.queues[lower_priority].put_nowait(item)
                        self.total_size += 1
                        return
                    except asyncio.QueueFull:
                        continue
            raise BulkheadFullError("All priority queues are full")
    
    async def get(self) -> Any:
        """Get item from highest priority queue."""
        # Check queues in priority order
        for priority in [PoolPriority.CRITICAL, PoolPriority.HIGH, PoolPriority.NORMAL, PoolPriority.LOW]:
            queue = self.queues[priority]
            if not queue.empty():
                item = await queue.get()
                self.total_size -= 1
                return item
        
        # If no items in any queue, wait for the first available
        tasks = [
            asyncio.create_task(queue.get())
            for queue in self.queues.values()
            if not queue.empty()
        ]
        
        if not tasks:
            # Wait for any queue to have an item
            tasks = [
                asyncio.create_task(queue.get())
                for queue in self.queues.values()
            ]
        
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        
        # Cancel pending tasks
        for task in pending:
            task.cancel()
        
        # Get result from completed task
        completed_task = done.pop()
        self.total_size -= 1
        return await completed_task
    
    def qsize(self) -> int:
        """Get total queue size."""
        return self.total_size
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        return {
            "total_size": self.total_size,
            "max_size": self.max_size,
            "queue_sizes": {
                priority.name: queue.qsize()
                for priority, queue in self.queues.items()
            }
        }


class ResourcePool:
    """Resource pool with bulkhead isolation."""
    
    def __init__(self, config: BulkheadConfig):
        self.config = config
        self.semaphore = asyncio.Semaphore(config.max_concurrent)
        self.metrics = PoolMetrics()
        
        # Queue management
        if config.enable_priority_queue:
            self.queue = PriorityQueue(config)
        else:
            self.queue = asyncio.Queue(maxsize=config.queue_size)
        
        # Dynamic sizing
        self.current_size = config.max_concurrent
        self.utilization_history = deque(maxlen=20)
        self.last_scale_time = time.time()
        
        # Active requests tracking
        self.active_requests = 0
        self.active_request_times = {}
        
        # Workers
        self.workers = []
        self.worker_shutdown = False
        
        # Start worker tasks
        self._start_workers()
    
    def _start_workers(self):
        """Start worker tasks to process queued requests."""
        for i in range(min(5, self.config.max_concurrent)):
            worker = asyncio.create_task(self._worker(f"worker_{i}"))
            self.workers.append(worker)
    
    async def _worker(self, worker_id: str):
        """Worker task to process queued requests."""
        while not self.worker_shutdown:
            try:
                # Get next request from queue
                if isinstance(self.queue, PriorityQueue):
                    request_item = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                else:
                    request_item = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                
                # Process the request
                await self._process_request(request_item)
                
            except asyncio.TimeoutError:
                # No requests in queue, continue
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                await asyncio.sleep(0.1)
    
    async def _process_request(self, request_item: Dict[str, Any]):
        """Process a queued request."""
        func = request_item['func']
        args = request_item.get('args', ())
        kwargs = request_item.get('kwargs', {})
        future = request_item['future']
        priority = request_item.get('priority', PoolPriority.NORMAL)
        enqueue_time = request_item.get('enqueue_time', time.time())
        
        queue_time = time.time() - enqueue_time
        
        async with self.semaphore:
            start_time = time.time()
            self.active_requests += 1
            request_id = f"req_{start_time}_{id(future)}"
            self.active_request_times[request_id] = start_time
            
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                future.set_result(result)
                
                # Record successful execution
                execution_time = time.time() - start_time
                self._update_metrics(execution_time, queue_time, success=True)
                
            except Exception as e:
                future.set_exception(e)
                
                # Record failed execution
                execution_time = time.time() - start_time
                self._update_metrics(execution_time, queue_time, success=False)
                
            finally:
                self.active_requests -= 1
                if request_id in self.active_request_times:
                    del self.active_request_times[request_id]
    
    def _update_metrics(self, execution_time: float, queue_time: float, success: bool):
        """Update pool metrics."""
        self.metrics.total_requests += 1
        self.metrics.total_execution_time += execution_time
        self.metrics.total_queue_time += queue_time
        
        if success:
            self.metrics.successful_requests += 1
        else:
            self.metrics.failed_requests += 1
        
        # Update timing metrics
        self.metrics.max_execution_time = max(self.metrics.max_execution_time, execution_time)
        self.metrics.min_execution_time = min(self.metrics.min_execution_time, execution_time)
        
        if self.metrics.total_requests > 0:
            self.metrics.avg_execution_time = (
                self.metrics.total_execution_time / self.metrics.total_requests
            )
            self.metrics.avg_queue_time = (
                self.metrics.total_queue_time / self.metrics.total_requests
            )
        
        # Update utilization metrics
        self.metrics.peak_concurrent = max(self.metrics.peak_concurrent, self.active_requests)
        current_utilization = self.active_requests / self.current_size
        self.utilization_history.append(current_utilization)
        
        if self.utilization_history:
            self.metrics.avg_utilization = mean(self.utilization_history)
        
        self.metrics.last_updated = time.time()
        
        # Log slow requests
        if (self.config.metrics_enabled and 
            execution_time > self.config.slow_request_threshold):
            logger.warning(
                f"Slow request in pool {self.config.name}: "
                f"{execution_time:.2f}s (threshold: {self.config.slow_request_threshold}s)"
            )
    
    async def execute(
        self, 
        func: Callable, 
        *args, 
        priority: PoolPriority = PoolPriority.NORMAL,
        timeout: Optional[float] = None,
        **kwargs
    ) -> Any:
        """Execute function in the resource pool."""
        if timeout is None:
            timeout = self.config.timeout
        
        # Check if we can execute immediately (for fast path)
        if self.queue.qsize() == 0 and self.semaphore._value > 0:
            # Fast path - execute immediately
            async with self.semaphore:
                start_time = time.time()
                self.active_requests += 1
                request_id = f"req_{start_time}_{id(func)}"
                self.active_request_times[request_id] = start_time
                
                try:
                    if asyncio.iscoroutinefunction(func):
                        result = await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
                    else:
                        result = func(*args, **kwargs)
                    
                    execution_time = time.time() - start_time
                    self._update_metrics(execution_time, 0.0, success=True)
                    
                    return result
                    
                except asyncio.TimeoutError:
                    self.metrics.timeout_requests += 1
                    raise BulkheadTimeoutError(f"Request timed out after {timeout}s")
                    
                except Exception as e:
                    execution_time = time.time() - start_time
                    self._update_metrics(execution_time, 0.0, success=False)
                    raise
                    
                finally:
                    self.active_requests -= 1
                    if request_id in self.active_request_times:
                        del self.active_request_times[request_id]
        
        # Slow path - queue the request
        future = asyncio.Future()
        request_item = {
            'func': func,
            'args': args,
            'kwargs': kwargs,
            'future': future,
            'priority': priority,
            'enqueue_time': time.time()
        }
        
        try:
            if isinstance(self.queue, PriorityQueue):
                await self.queue.put(request_item, priority)
            else:
                self.queue.put_nowait(request_item)
            
            # Wait for result with timeout
            result = await asyncio.wait_for(future, timeout=timeout)
            return result
            
        except asyncio.QueueFull:
            self.metrics.rejected_requests += 1
            raise BulkheadFullError(f"Resource pool {self.config.name} is full")
            
        except asyncio.TimeoutError:
            self.metrics.timeout_requests += 1
            # Cancel the future to prevent resource leaks
            future.cancel()
            raise BulkheadTimeoutError(f"Request timed out after {timeout}s")
    
    @asynccontextmanager
    async def acquire(self, timeout: Optional[float] = None):
        """Context manager for acquiring pool resources."""
        if timeout is None:
            timeout = self.config.timeout
        
        try:
            await asyncio.wait_for(self.semaphore.acquire(), timeout=timeout)
            start_time = time.time()
            self.active_requests += 1
            
            yield
            
        except asyncio.TimeoutError:
            self.metrics.timeout_requests += 1
            raise BulkheadTimeoutError(f"Failed to acquire resource within {timeout}s")
            
        finally:
            self.active_requests -= 1
            self.semaphore.release()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive pool statistics."""
        current_time = time.time()
        
        stats = {
            "name": self.config.name,
            "current_size": self.current_size,
            "max_size": self.config.max_size,
            "active_requests": self.active_requests,
            "queue_size": self.queue.qsize() if hasattr(self.queue, 'qsize') else 0,
            "available_slots": self.semaphore._value,
            "utilization": self.active_requests / self.current_size if self.current_size > 0 else 0,
            "metrics": {
                "total_requests": self.metrics.total_requests,
                "successful_requests": self.metrics.successful_requests,
                "failed_requests": self.metrics.failed_requests,
                "timeout_requests": self.metrics.timeout_requests,
                "rejected_requests": self.metrics.rejected_requests,
                "success_rate": (
                    (self.metrics.successful_requests / self.metrics.total_requests * 100)
                    if self.metrics.total_requests > 0 else 100.0
                ),
                "avg_execution_time": self.metrics.avg_execution_time,
                "max_execution_time": self.metrics.max_execution_time,
                "avg_queue_time": self.metrics.avg_queue_time,
                "peak_concurrent": self.metrics.peak_concurrent,
                "avg_utilization": self.metrics.avg_utilization
            }
        }
        
        # Add queue stats if priority queue
        if isinstance(self.queue, PriorityQueue):
            stats["priority_queue"] = self.queue.get_stats()
        
        return stats
    
    async def shutdown(self):
        """Shutdown the resource pool."""
        logger.info(f"Shutting down resource pool: {self.config.name}")
        
        self.worker_shutdown = True
        
        # Cancel all worker tasks
        for worker in self.workers:
            worker.cancel()
        
        # Wait for workers to complete
        await asyncio.gather(*self.workers, return_exceptions=True)
        
        logger.info(f"Resource pool {self.config.name} shutdown complete")


class BulkheadManager:
    """Enhanced bulkhead manager with dynamic pool management."""
    
    def __init__(self):
        self.pools: Dict[str, ResourcePool] = {}
        self.global_metrics = defaultdict(int)
    
    def create_pool(self, config: BulkheadConfig) -> str:
        """Create a new resource pool."""
        if config.name in self.pools:
            logger.warning(f"Pool {config.name} already exists, updating configuration")
            # TODO: Handle pool reconfiguration
        
        pool = ResourcePool(config)
        self.pools[config.name] = pool
        
        logger.info(
            f"Created resource pool '{config.name}' with {config.max_concurrent} slots"
        )
        
        return config.name
    
    def get_pool(self, name: str) -> Optional[ResourcePool]:
        """Get resource pool by name."""
        return self.pools.get(name)
    
    async def execute_in_pool(
        self, 
        pool_name: str, 
        func: Callable, 
        *args, 
        priority: PoolPriority = PoolPriority.NORMAL,
        timeout: Optional[float] = None,
        **kwargs
    ) -> Any:
        """Execute function in specified resource pool."""
        pool = self.get_pool(pool_name)
        if not pool:
            # Create default pool if it doesn't exist
            config = BulkheadConfig(name=pool_name)
            self.create_pool(config)
            pool = self.pools[pool_name]
        
        self.global_metrics["total_requests"] += 1
        
        try:
            result = await pool.execute(func, *args, priority=priority, timeout=timeout, **kwargs)
            self.global_metrics["successful_requests"] += 1
            return result
            
        except Exception as e:
            self.global_metrics["failed_requests"] += 1
            raise
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all pools."""
        return {
            "global_metrics": dict(self.global_metrics),
            "total_pools": len(self.pools),
            "pools": {
                name: pool.get_stats()
                for name, pool in self.pools.items()
            }
        }
    
    async def shutdown_all(self):
        """Shutdown all resource pools."""
        logger.info("Shutting down all resource pools")
        
        shutdown_tasks = [
            pool.shutdown() for pool in self.pools.values()
        ]
        
        await asyncio.gather(*shutdown_tasks, return_exceptions=True)
        
        self.pools.clear()
        logger.info("All resource pools shutdown complete")


# Custom exceptions
class BulkheadFullError(Exception):
    """Exception raised when bulkhead capacity is exceeded."""
    pass


class BulkheadTimeoutError(Exception):
    """Exception raised when bulkhead operation times out."""
    pass


# Global bulkhead manager instance
bulkhead_manager = BulkheadManager()


# Predefined pool configurations
GITHUB_API_POOL_CONFIG = BulkheadConfig(
    name="github_api",
    max_concurrent=5,
    queue_size=20,
    timeout=30.0,
    enable_priority_queue=True,
    slow_request_threshold=2.0
)

DATABASE_POOL_CONFIG = BulkheadConfig(
    name="database",
    max_concurrent=20,
    queue_size=50,
    timeout=10.0,
    enable_auto_scaling=True,
    min_size=5,
    max_size=50
)

ML_INFERENCE_POOL_CONFIG = BulkheadConfig(
    name="ml_inference",
    max_concurrent=3,
    queue_size=10,
    timeout=120.0,
    enable_priority_queue=True,
    slow_request_threshold=30.0
)


def create_bulkhead_decorator(config: BulkheadConfig, priority: PoolPriority = PoolPriority.NORMAL):
    """Create a decorator for bulkhead-protected functions."""
    bulkhead_manager.create_pool(config)
    
    def decorator(func: Callable) -> Callable:
        async def async_wrapper(*args, **kwargs):
            return await bulkhead_manager.execute_in_pool(
                config.name, func, *args, priority=priority, **kwargs
            )
        
        def sync_wrapper(*args, **kwargs):
            if asyncio.iscoroutinefunction(func):
                return asyncio.run(async_wrapper(*args, **kwargs))
            else:
                return asyncio.run(
                    bulkhead_manager.execute_in_pool(
                        config.name, func, *args, priority=priority, **kwargs
                    )
                )
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator