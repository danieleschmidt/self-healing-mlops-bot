"""Concurrent processing and resource management."""

import asyncio
import time
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
from collections import defaultdict, deque
import resource
import psutil

from ..monitoring.logging import get_logger, performance_logger
from ..monitoring.metrics import metrics

logger = get_logger(__name__)


@dataclass
class TaskMetrics:
    """Metrics for tracking task performance."""
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    avg_execution_time: float = 0.0
    max_execution_time: float = 0.0
    min_execution_time: float = float('inf')
    total_execution_time: float = 0.0
    
    def update(self, execution_time: float, success: bool):
        """Update metrics with new task completion."""
        self.total_tasks += 1
        self.total_execution_time += execution_time
        
        if success:
            self.completed_tasks += 1
        else:
            self.failed_tasks += 1
        
        self.max_execution_time = max(self.max_execution_time, execution_time)
        self.min_execution_time = min(self.min_execution_time, execution_time)
        self.avg_execution_time = self.total_execution_time / self.total_tasks


class AdaptiveExecutor:
    """Adaptive executor that adjusts concurrency based on system resources."""
    
    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or min(32, (psutil.cpu_count() or 1) + 4)
        self.current_workers = max(1, self.max_workers // 4)  # Start conservative
        self.thread_executor = ThreadPoolExecutor(max_workers=self.current_workers)
        self.process_executor = ProcessPoolExecutor(max_workers=max(1, psutil.cpu_count() or 1))
        
        self.task_metrics: Dict[str, TaskMetrics] = defaultdict(TaskMetrics)
        self.resource_monitor = ResourceMonitor()
        self.last_adjustment = time.time()
        self.min_adjustment_interval = 30  # seconds
        
    async def submit_io_task(self, func: Callable, *args, task_type: str = "io", **kwargs) -> Any:
        """Submit I/O bound task to thread pool."""
        start_time = time.time()
        
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(self.thread_executor, func, *args)
            
            execution_time = time.time() - start_time
            self.task_metrics[task_type].update(execution_time, True)
            
            performance_logger.log_execution_time(
                f"async_io_task_{task_type}", execution_time, True
            )
            
            # Adjust concurrency if needed
            await self._maybe_adjust_concurrency()
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.task_metrics[task_type].update(execution_time, False)
            
            performance_logger.log_execution_time(
                f"async_io_task_{task_type}", execution_time, False
            )
            
            logger.error(f"I/O task {task_type} failed: {e}")
            raise
    
    async def submit_cpu_task(self, func: Callable, *args, task_type: str = "cpu", **kwargs) -> Any:
        """Submit CPU bound task to process pool."""
        start_time = time.time()
        
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(self.process_executor, func, *args)
            
            execution_time = time.time() - start_time
            self.task_metrics[task_type].update(execution_time, True)
            
            performance_logger.log_execution_time(
                f"async_cpu_task_{task_type}", execution_time, True
            )
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.task_metrics[task_type].update(execution_time, False)
            
            performance_logger.log_execution_time(
                f"async_cpu_task_{task_type}", execution_time, False
            )
            
            logger.error(f"CPU task {task_type} failed: {e}")
            raise
    
    async def _maybe_adjust_concurrency(self):
        """Adjust concurrency based on system performance."""
        current_time = time.time()
        
        if current_time - self.last_adjustment < self.min_adjustment_interval:
            return
        
        # Get current resource usage
        resource_info = await self.resource_monitor.get_current_usage()
        
        # Decide whether to scale up or down
        should_scale_up = (
            resource_info["cpu_percent"] < 70 and
            resource_info["memory_percent"] < 80 and
            self.current_workers < self.max_workers
        )
        
        should_scale_down = (
            resource_info["cpu_percent"] > 85 or
            resource_info["memory_percent"] > 90 or
            self._has_high_task_failure_rate()
        )
        
        if should_scale_up:
            new_workers = min(self.max_workers, int(self.current_workers * 1.5))
            await self._adjust_thread_pool(new_workers)
            logger.info(f"Scaled up thread pool to {new_workers} workers")
            
        elif should_scale_down:
            new_workers = max(1, int(self.current_workers * 0.7))
            await self._adjust_thread_pool(new_workers)
            logger.info(f"Scaled down thread pool to {new_workers} workers")
        
        self.last_adjustment = current_time
    
    def _has_high_task_failure_rate(self) -> bool:
        """Check if task failure rate is too high."""
        for task_type, metrics in self.task_metrics.items():
            if metrics.total_tasks > 10:  # Minimum sample size
                failure_rate = metrics.failed_tasks / metrics.total_tasks
                if failure_rate > 0.1:  # More than 10% failure rate
                    return True
        return False
    
    async def _adjust_thread_pool(self, new_size: int):
        """Adjust thread pool size."""
        if new_size != self.current_workers:
            # Shutdown old executor
            self.thread_executor.shutdown(wait=False)
            
            # Create new executor with adjusted size
            self.thread_executor = ThreadPoolExecutor(max_workers=new_size)
            self.current_workers = new_size
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get executor metrics."""
        return {
            "current_workers": self.current_workers,
            "max_workers": self.max_workers,
            "task_metrics": {
                task_type: {
                    "total_tasks": metrics.total_tasks,
                    "completed_tasks": metrics.completed_tasks,
                    "failed_tasks": metrics.failed_tasks,
                    "success_rate": metrics.completed_tasks / metrics.total_tasks if metrics.total_tasks > 0 else 0,
                    "avg_execution_time": metrics.avg_execution_time,
                    "max_execution_time": metrics.max_execution_time,
                    "min_execution_time": metrics.min_execution_time if metrics.min_execution_time != float('inf') else 0
                }
                for task_type, metrics in self.task_metrics.items()
            }
        }
    
    async def shutdown(self):
        """Shutdown executors gracefully."""
        self.thread_executor.shutdown(wait=True)
        self.process_executor.shutdown(wait=True)


class ResourceMonitor:
    """Monitor system resources for adaptive scaling."""
    
    def __init__(self):
        self.history_size = 60  # Keep 60 samples (5 minutes at 5-second intervals)
        self.cpu_history = deque(maxlen=self.history_size)
        self.memory_history = deque(maxlen=self.history_size)
        self.monitoring_task: Optional[asyncio.Task] = None
        self.running = False
    
    async def start_monitoring(self):
        """Start continuous resource monitoring."""
        self.running = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
    
    async def stop_monitoring(self):
        """Stop resource monitoring."""
        self.running = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
    
    async def _monitoring_loop(self):
        """Continuous monitoring loop."""
        while self.running:
            try:
                cpu_percent = psutil.cpu_percent(interval=1)
                memory_info = psutil.virtual_memory()
                
                self.cpu_history.append(cpu_percent)
                self.memory_history.append(memory_info.percent)
                
                # Log resource usage
                performance_logger.log_resource_usage(
                    "system",
                    cpu_percent=cpu_percent,
                    memory_mb=memory_info.used / 1024 / 1024
                )
                
                await asyncio.sleep(5)  # Sample every 5 seconds
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                await asyncio.sleep(10)  # Longer delay on error
    
    async def get_current_usage(self) -> Dict[str, float]:
        """Get current resource usage."""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_info = psutil.virtual_memory()
            
            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory_info.percent,
                "memory_available_mb": memory_info.available / 1024 / 1024,
                "memory_used_mb": memory_info.used / 1024 / 1024
            }
        except Exception as e:
            logger.error(f"Failed to get resource usage: {e}")
            return {
                "cpu_percent": 0.0,
                "memory_percent": 0.0,
                "memory_available_mb": 0.0,
                "memory_used_mb": 0.0
            }
    
    def get_average_usage(self, window_minutes: int = 5) -> Dict[str, float]:
        """Get average resource usage over a time window."""
        samples = min(window_minutes * 12, len(self.cpu_history))  # 12 samples per minute
        
        if samples == 0:
            return {"cpu_percent": 0.0, "memory_percent": 0.0}
        
        recent_cpu = list(self.cpu_history)[-samples:]
        recent_memory = list(self.memory_history)[-samples:]
        
        return {
            "cpu_percent": sum(recent_cpu) / len(recent_cpu),
            "memory_percent": sum(recent_memory) / len(recent_memory)
        }


class ConcurrentBatchProcessor:
    """Process batches of similar tasks concurrently."""
    
    def __init__(self, executor: AdaptiveExecutor, batch_size: int = 10):
        self.executor = executor
        self.batch_size = batch_size
        self.pending_batches: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.batch_locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
    
    async def add_task(self, batch_type: str, task_data: Dict[str, Any], process_func: Callable):
        """Add task to batch for processing."""
        async with self.batch_locks[batch_type]:
            self.pending_batches[batch_type].append({
                "data": task_data,
                "process_func": process_func,
                "timestamp": time.time()
            })
            
            # Process batch if it's full or if oldest task is too old
            if (len(self.pending_batches[batch_type]) >= self.batch_size or
                self._should_process_batch(batch_type)):
                await self._process_batch(batch_type)
    
    def _should_process_batch(self, batch_type: str) -> bool:
        """Check if batch should be processed based on age."""
        if not self.pending_batches[batch_type]:
            return False
        
        oldest_task = self.pending_batches[batch_type][0]
        age = time.time() - oldest_task["timestamp"]
        return age > 30  # Process if oldest task is more than 30 seconds old
    
    async def _process_batch(self, batch_type: str):
        """Process a batch of tasks concurrently."""
        if not self.pending_batches[batch_type]:
            return
        
        batch = self.pending_batches[batch_type].copy()
        self.pending_batches[batch_type].clear()
        
        logger.info(f"Processing batch of {len(batch)} {batch_type} tasks")
        
        # Create tasks for concurrent execution
        tasks = []
        for task_info in batch:
            task = asyncio.create_task(
                self._execute_task(task_info, batch_type)
            )
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Log batch completion
        successful = sum(1 for r in results if not isinstance(r, Exception))
        failed = len(results) - successful
        
        logger.info(f"Batch {batch_type} completed: {successful} successful, {failed} failed")
        
        return results
    
    async def _execute_task(self, task_info: Dict[str, Any], batch_type: str) -> Any:
        """Execute individual task within a batch."""
        try:
            process_func = task_info["process_func"]
            task_data = task_info["data"]
            
            if asyncio.iscoroutinefunction(process_func):
                return await process_func(task_data)
            else:
                return await self.executor.submit_io_task(
                    process_func, task_data, task_type=batch_type
                )
        except Exception as e:
            logger.error(f"Task in batch {batch_type} failed: {e}")
            return e
    
    async def flush_all_batches(self):
        """Process all pending batches."""
        batch_types = list(self.pending_batches.keys())
        
        for batch_type in batch_types:
            async with self.batch_locks[batch_type]:
                if self.pending_batches[batch_type]:
                    await self._process_batch(batch_type)


class CircuitBreaker:
    """Circuit breaker pattern for handling failures gracefully."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = "closed"  # closed, open, half-open
        self.lock = asyncio.Lock()
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Call function through circuit breaker."""
        async with self.lock:
            if self.state == "open":
                if self._should_attempt_reset():
                    self.state = "half-open"
                else:
                    raise Exception("Circuit breaker is open")
            
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Success - reset failure count
                if self.state == "half-open":
                    self.state = "closed"
                self.failure_count = 0
                
                return result
                
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = "open"
                    logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
                
                raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        if not self.last_failure_time:
            return True
        
        return time.time() - self.last_failure_time >= self.recovery_timeout
    
    def get_state(self) -> Dict[str, Any]:
        """Get circuit breaker state."""
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "last_failure_time": self.last_failure_time,
            "time_since_last_failure": time.time() - self.last_failure_time if self.last_failure_time else None
        }


# Global instances
adaptive_executor = AdaptiveExecutor()
resource_monitor = ResourceMonitor()
batch_processor = ConcurrentBatchProcessor(adaptive_executor)