"""Performance optimization and resource management."""

import asyncio
import logging
import time
import resource
import gc
import psutil
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics tracking."""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    memory_available: int = 0
    active_tasks: int = 0
    queue_depth: int = 0
    average_response_time: float = 0.0
    throughput: float = 0.0  # operations per second
    error_rate: float = 0.0


class PerformanceMonitor:
    """Monitor system performance and resource usage."""
    
    def __init__(self):
        self.metrics = PerformanceMetrics()
        self._start_time = time.time()
        self._request_count = 0
        self._error_count = 0
        self._response_times = []
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
    
    async def start_monitoring(self, interval: float = 30.0):
        """Start performance monitoring."""
        if not self._monitoring:
            self._monitoring = True
            self._monitor_task = asyncio.create_task(self._monitor_loop(interval))
            logger.info("Performance monitoring started")
    
    async def stop_monitoring(self):
        """Stop performance monitoring."""
        if self._monitoring:
            self._monitoring = False
            if self._monitor_task:
                self._monitor_task.cancel()
                try:
                    await self._monitor_task
                except asyncio.CancelledError:
                    pass
            logger.info("Performance monitoring stopped")
    
    async def _monitor_loop(self, interval: float):
        """Background monitoring loop."""
        while self._monitoring:
            try:
                await self._update_metrics()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in performance monitoring: {e}")
    
    async def _update_metrics(self):
        """Update performance metrics."""
        try:
            # System metrics
            self.metrics.cpu_usage = psutil.cpu_percent(interval=None)
            
            memory = psutil.virtual_memory()
            self.metrics.memory_usage = memory.percent
            self.metrics.memory_available = memory.available
            
            # Task metrics
            current_task = asyncio.current_task()
            all_tasks = asyncio.all_tasks()
            self.metrics.active_tasks = len([t for t in all_tasks if not t.done()])
            
            # Calculate throughput and response times
            current_time = time.time()
            elapsed = current_time - self._start_time
            
            if elapsed > 0:
                self.metrics.throughput = self._request_count / elapsed
                
            if self._response_times:
                self.metrics.average_response_time = sum(self._response_times) / len(self._response_times)
                # Keep only recent response times
                if len(self._response_times) > 100:
                    self._response_times = self._response_times[-50:]
            
            # Error rate
            if self._request_count > 0:
                self.metrics.error_rate = self._error_count / self._request_count
            
        except ImportError:
            # psutil not available, use basic metrics
            self.metrics.cpu_usage = 0.0
            self.metrics.memory_usage = 0.0
    
    def record_request(self, response_time: float, success: bool = True):
        """Record a request for metrics."""
        self._request_count += 1
        self._response_times.append(response_time)
        
        if not success:
            self._error_count += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return {
            "cpu_usage_percent": self.metrics.cpu_usage,
            "memory_usage_percent": self.metrics.memory_usage,
            "memory_available_mb": self.metrics.memory_available // (1024 * 1024),
            "active_tasks": self.metrics.active_tasks,
            "average_response_time_ms": self.metrics.average_response_time * 1000,
            "throughput_ops_per_second": self.metrics.throughput,
            "error_rate_percent": self.metrics.error_rate * 100,
            "total_requests": self._request_count,
            "total_errors": self._error_count,
            "uptime_seconds": time.time() - self._start_time
        }


class ResourceManager:
    """Manage system resources and prevent resource exhaustion."""
    
    def __init__(self):
        self.memory_limit = 1024 * 1024 * 1024  # 1GB default
        self.cpu_limit = 80.0  # 80% CPU usage limit
        self.task_limit = 50   # Maximum concurrent tasks
        self._resource_locks = {}
    
    async def check_resources(self) -> Dict[str, Any]:
        """Check current resource usage against limits."""
        status = {"ok": True, "warnings": [], "critical": []}
        
        try:
            # Memory check
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                status["critical"].append("Memory usage critical (>90%)")
                status["ok"] = False
            elif memory.percent > self.cpu_limit:
                status["warnings"].append(f"Memory usage high ({memory.percent:.1f}%)")
            
            # CPU check  
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 95:
                status["critical"].append("CPU usage critical (>95%)")
                status["ok"] = False
            elif cpu_percent > self.cpu_limit:
                status["warnings"].append(f"CPU usage high ({cpu_percent:.1f}%)")
            
            # Task count check
            active_tasks = len([t for t in asyncio.all_tasks() if not t.done()])
            if active_tasks > self.task_limit:
                status["warnings"].append(f"High task count ({active_tasks})")
            
        except ImportError:
            status["warnings"].append("Resource monitoring not available (psutil missing)")
        
        return status
    
    @asynccontextmanager
    async def resource_guard(self, operation_name: str):
        """Context manager to guard resource-intensive operations."""
        start_time = time.time()
        
        # Check resources before starting
        resource_status = await self.check_resources()
        if not resource_status["ok"]:
            logger.warning(f"Starting {operation_name} with resource warnings: {resource_status['critical']}")
        
        try:
            yield
        finally:
            # Track operation duration
            duration = time.time() - start_time
            if duration > 30:  # Log long-running operations
                logger.info(f"Long-running operation {operation_name}: {duration:.2f}s")
    
    async def garbage_collect_if_needed(self):
        """Trigger garbage collection if memory usage is high."""
        try:
            memory = psutil.virtual_memory()
            if memory.percent > 80:
                logger.info("High memory usage detected, running garbage collection")
                collected = gc.collect()
                logger.info(f"Garbage collection freed {collected} objects")
        except ImportError:
            # Fallback: always run gc periodically
            gc.collect()


class AdaptiveThrottling:
    """Adaptive throttling based on system load."""
    
    def __init__(self):
        self.base_delay = 0.1  # Base delay in seconds
        self.max_delay = 5.0   # Maximum delay
        self.load_threshold = 0.8  # CPU/memory threshold
        self._current_delay = self.base_delay
    
    async def throttle(self):
        """Apply adaptive throttling delay."""
        try:
            # Get system load
            cpu_percent = psutil.cpu_percent(interval=None)
            memory_percent = psutil.virtual_memory().percent
            
            max_load = max(cpu_percent, memory_percent) / 100.0
            
            # Adjust delay based on load
            if max_load > self.load_threshold:
                # Increase delay exponentially with load
                load_factor = (max_load - self.load_threshold) / (1.0 - self.load_threshold)
                self._current_delay = min(
                    self.base_delay * (1 + load_factor * 10),
                    self.max_delay
                )
            else:
                # Gradually reduce delay when load is low
                self._current_delay = max(
                    self._current_delay * 0.9,
                    self.base_delay
                )
            
            if self._current_delay > self.base_delay:
                logger.debug(f"Throttling: delay={self._current_delay:.2f}s, load={max_load:.1%}")
                await asyncio.sleep(self._current_delay)
                
        except ImportError:
            # Fallback: use base delay
            await asyncio.sleep(self.base_delay)
    
    def get_current_delay(self) -> float:
        """Get current throttling delay."""
        return self._current_delay


class BatchProcessor:
    """Batch processing for improved throughput."""
    
    def __init__(self, batch_size: int = 10, flush_interval: float = 1.0):
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self._pending_items = []
        self._last_flush = time.time()
        self._lock = asyncio.Lock()
    
    async def add_item(self, item: Any, processor: Callable[[List], Any]) -> Optional[Any]:
        """Add item to batch for processing."""
        async with self._lock:
            self._pending_items.append((item, processor))
            
            # Check if batch is ready
            current_time = time.time()
            should_flush = (
                len(self._pending_items) >= self.batch_size or
                current_time - self._last_flush >= self.flush_interval
            )
            
            if should_flush:
                return await self._flush_batch()
            
        return None
    
    async def _flush_batch(self) -> List[Any]:
        """Process the current batch."""
        if not self._pending_items:
            return []
        
        batch = self._pending_items.copy()
        self._pending_items.clear()
        self._last_flush = time.time()
        
        # Group items by processor
        processor_groups = {}
        for item, processor in batch:
            processor_id = id(processor)
            if processor_id not in processor_groups:
                processor_groups[processor_id] = (processor, [])
            processor_groups[processor_id][1].append(item)
        
        # Process each group
        results = []
        for processor, items in processor_groups.values():
            try:
                if asyncio.iscoroutinefunction(processor):
                    result = await processor(items)
                else:
                    result = processor(items)
                results.extend(result if isinstance(result, list) else [result])
            except Exception as e:
                logger.exception(f"Batch processing error: {e}")
        
        return results
    
    async def flush(self) -> List[Any]:
        """Flush any remaining items."""
        async with self._lock:
            return await self._flush_batch()


class PriorityQueue:
    """Priority queue for task scheduling."""
    
    def __init__(self):
        self._queues = {
            "critical": asyncio.Queue(),
            "high": asyncio.Queue(),
            "normal": asyncio.Queue(),
            "low": asyncio.Queue()
        }
        self._priority_order = ["critical", "high", "normal", "low"]
    
    async def put(self, item: Any, priority: str = "normal"):
        """Add item to queue with priority."""
        if priority not in self._queues:
            priority = "normal"
        
        await self._queues[priority].put(item)
    
    async def get(self) -> Any:
        """Get next item from highest priority non-empty queue."""
        while True:
            for priority in self._priority_order:
                queue = self._queues[priority]
                if not queue.empty():
                    return await queue.get()
            
            # All queues empty, wait a bit
            await asyncio.sleep(0.1)
    
    def qsize(self) -> Dict[str, int]:
        """Get size of each priority queue."""
        return {priority: queue.qsize() for priority, queue in self._queues.items()}
    
    def total_size(self) -> int:
        """Get total items across all queues."""
        return sum(queue.qsize() for queue in self._queues.values())


# Global instances
performance_monitor = PerformanceMonitor()
resource_manager = ResourceManager()
adaptive_throttling = AdaptiveThrottling()


def performance_optimized(monitor_performance: bool = True):
    """Decorator to add performance optimization to functions."""
    def decorator(func: Callable) -> Callable:
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                # Apply throttling if system load is high
                await adaptive_throttling.throttle()
                
                # Execute function with resource management
                async with resource_manager.resource_guard(func.__name__):
                    if asyncio.iscoroutinefunction(func):
                        result = await func(*args, **kwargs)
                    else:
                        result = func(*args, **kwargs)
                
                # Record successful execution
                if monitor_performance:
                    execution_time = time.time() - start_time
                    performance_monitor.record_request(execution_time, success=True)
                
                return result
                
            except Exception as e:
                # Record failed execution
                if monitor_performance:
                    execution_time = time.time() - start_time
                    performance_monitor.record_request(execution_time, success=False)
                
                logger.exception(f"Performance-optimized function {func.__name__} failed: {e}")
                raise
        
        return wrapper
    return decorator